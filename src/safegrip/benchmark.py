from __future__ import annotations
import json, time, hashlib, platform
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from joblib import dump

from .literature import LITERATURE_BASELINES, LITERATURE_ONLY, PAPER_BASELINES, QUICK_BASELINES, validate_paper_baselines
from .models import make_literature_baseline, SafeGripNet, SafeGripNoTemporal, DeterministicTCN
from .physics import (
    project_torch, project_numpy, project_interval_numpy, gaussian_interval,
    conformal_lower_correction, apply_lower_correction,
)
from .utils import ensure_dir, seed_everything, device

META={
    "time","distance","lat","lon","route_s_m","gps_heading_deg","match_distance_m","match_ref_index",
    "mu_ref","physics_lower_mechanics","physics_lower_raw","split","split_position","source_file","ref_source_file",
    "route_id","direction","trip_id","trajectory_id","segment_id","sample_uid","resampled",
}
PROPOSAL_VARIANTS=(
    "safegrip_data_only",
    "safegrip_no_projection",
    "safegrip_no_uq",
    "safegrip_no_physics_loss",
    "safegrip_no_calibration",
    "safegrip_no_temporal",
    "safegrip",
)


@dataclass
class Bundle:
    features: list[str]
    scaler: object
    scaler_kind: str
    sequence_length: int
    eval_start: int
    q: float
    Xtr: np.ndarray; ytr: np.ndarray; lotr: np.ndarray; raw_lotr: np.ndarray; idtr: np.ndarray
    Xc: np.ndarray; yc: np.ndarray; loc: np.ndarray; raw_loc: np.ndarray; idc: np.ndarray
    Xv: np.ndarray; yv: np.ndarray; lov: np.ndarray; raw_lov: np.ndarray; idv: np.ndarray
    Xt: np.ndarray; yt: np.ndarray; lot: np.ndarray; raw_lot: np.ndarray; idt: np.ndarray


def common_eval_start(cfg) -> int:
    """Return a common endpoint warm-up so different history lengths share labels."""
    b=cfg.get("benchmark",{})
    if "common_warmup_samples" in b:
        return max(0,int(b["common_warmup_samples"])-1)
    values=[int(cfg.get("sequence_length",64)),100]
    values.extend(int(x) for x in cfg.get("tuning",{}).get("space",{}).get("sequence_length",[]))
    for x in cfg.get("baseline_tuning",{}).get("sequence_length",[]):
        values.append(int(x))
    return max(values)-1


def windows_for_split(df, features, split, L, stride, eval_start=None, physics_window=1):
    """Build windows strictly inside one trip and one split.

    This prevents the subtle leakage/error where filtering all rows by split and
    then resetting the index creates a sequence that bridges two independent
    trips or discontinuous matched trajectory segments.  The physics lower
    endpoint is the maximum over a fixed trailing physics window, matching the
    partial-identification theorem used in the paper. Stable endpoint IDs are
    returned for exact cross-model parity checks.
    """
    z=df[df.split==split].copy()
    xs=[]; ys=[]; ls=[]; ids=[]
    if z.empty:
        return (np.empty((0,L,len(features)),np.float32), np.empty(0,np.float32),
                np.empty(0,np.float32), np.empty(0,dtype=str))
    group_cols=[c for c in ("segment_id",) if c in z.columns]
    if not group_cols:
        group_cols=[c for c in ("trip_id",) if c in z.columns]
    groups=z.groupby(group_cols,sort=False,dropna=False) if group_cols else [("all",z)]
    pw=max(1,int(physics_window))
    first=max(L-1,pw-1,int(eval_start) if eval_start is not None else L-1)
    for gkey,g in groups:
        g=g.copy()
        sort_cols=[c for c in ("time","route_s_m","distance") if c in g.columns and g[c].notna().any()]
        if sort_cols:
            g=g.sort_values(sort_cols[0],kind="stable")
        else:
            g=g.sort_index(kind="stable")
        X=g[features].to_numpy(np.float32)
        y=g.mu_ref.to_numpy(np.float32)
        lo=g.physics_lower_raw.to_numpy(np.float32)
        if "sample_uid" in g.columns:
            uid=g.sample_uid.astype(str).to_numpy()
        else:
            prefix=str(gkey[0] if isinstance(gkey,tuple) else gkey)
            uid=np.asarray([f"{prefix}:{split}:{i}" for i in range(len(g))],dtype=str)
        for end in range(first,len(g),stride):
            xs.append(X[end-L+1:end+1]); ys.append(y[end])
            ls.append(float(np.nanmax(lo[end-pw+1:end+1])))
            ids.append(uid[end])
    if not xs:
        return (np.empty((0,L,len(features)),np.float32), np.empty(0,np.float32),
                np.empty(0,np.float32), np.empty(0,dtype=str))
    return np.stack(xs),np.asarray(ys,np.float32),np.asarray(ls,np.float32),np.asarray(ids,dtype=str)


def _make_scaler(kind: str):
    if kind=="standard": return StandardScaler()
    if kind=="minmax": return MinMaxScaler()
    raise ValueError(f"Unknown scaler: {kind}")


def make_bundle(csv_path, cfg, sequence_length=None, scaler_kind="standard", eval_start=None):
    df=pd.read_csv(csv_path)
    features=[c for c in df.columns if c not in META and pd.api.types.is_numeric_dtype(df[c])]
    features=[c for c in features if c not in ("mu_ref","physics_lower_raw")]
    L=int(sequence_length or cfg["sequence_length"]); stride=int(cfg["stride"])
    start=common_eval_start(cfg) if eval_start is None else int(eval_start)
    physics_window=int(cfg.get("physics",{}).get("window_samples",1))
    splits={sp:windows_for_split(df,features,sp,L,stride,start,physics_window=physics_window) for sp in ("train","calibration","validation","test")}
    Xtr,ytr,raw_lotr,idtr=splits["train"]; Xc,yc,raw_loc,idc=splits["calibration"]
    Xv,yv,raw_lov,idv=splits["validation"]; Xt,yt,raw_lot,idt=splits["test"]
    if min(len(Xtr),len(Xv),len(Xt))==0:
        raise RuntimeError("One split has no common-evaluation windows; lower benchmark.common_warmup_samples or inspect prepared data")
    scaler=_make_scaler(scaler_kind).fit(Xtr.reshape(-1,len(features)))
    def sc(X):
        if len(X)==0: return X.astype(np.float32)
        return scaler.transform(X.reshape(-1,len(features))).reshape(X.shape).astype(np.float32)
    Xtr,Xc,Xv,Xt=map(sc,(Xtr,Xc,Xv,Xt))
    q=conformal_lower_correction(raw_loc,yc,cfg["alpha"]) if len(yc) else 0.0
    lotr=apply_lower_correction(raw_lotr,q).astype(np.float32)
    loc=apply_lower_correction(raw_loc,q).astype(np.float32)
    lov=apply_lower_correction(raw_lov,q).astype(np.float32)
    lot=apply_lower_correction(raw_lot,q).astype(np.float32)
    return Bundle(features,scaler,scaler_kind,L,start,q,
                  Xtr,ytr,lotr,raw_lotr,idtr,Xc,yc,loc,raw_loc,idc,
                  Xv,yv,lov,raw_lov,idv,Xt,yt,lot,raw_lot,idt)


def regression_metrics(y,p,lo=None,mu_upper=1.3,sigma=None,raw_mean=None,project_uncertainty=False):
    y=np.asarray(y); p=np.asarray(p)
    out={
        "mae":float(mean_absolute_error(y,p)),
        "rmse":float(mean_squared_error(y,p)**.5),
        "r2":float(r2_score(y,p)),
        "unsafe_overestimate_mean":float(np.maximum(0,p-y).mean()),
        "unsafe_overestimate_rate_005":float(np.mean(p>y+0.05)),
    }
    if lo is not None:
        lo=np.asarray(lo)
        out.update({
            "lower_violation_rate":float(np.mean(lo>y)),
            "mean_identified_width":float(np.mean(np.maximum(0,mu_upper-lo))),
            "point_outside_physics_set_rate":float(np.mean((p<lo)|(p>mu_upper))),
            "projection_correction_rate":float(np.mean(np.abs(p-np.asarray(raw_mean if raw_mean is not None else p))>1e-8)),
        })
    if sigma is not None:
        sigma=np.maximum(np.asarray(sigma),1e-5)
        center=np.asarray(raw_mean if raw_mean is not None else p)
        low_raw,high_raw=gaussian_interval(center,sigma)
        out.update({
            "picp95_raw":float(np.mean((y>=low_raw)&(y<=high_raw))),
            "mpiw95_raw":float(np.mean(high_raw-low_raw)),
            # The Gaussian head predicts a distribution around ``raw_mean``.
            # Projection is a non-Gaussian post-processing map, so NLL must be
            # evaluated at the raw Gaussian center rather than the projected
            # point prediction.
            "gaussian_nll_raw":float(np.mean(.5*((y-center)/sigma)**2+np.log(sigma))),
        })
        if project_uncertainty and lo is not None:
            low,high=project_interval_numpy(low_raw,high_raw,np.asarray(lo),float(mu_upper))
            out.update({
                "picp95":float(np.mean((y>=low)&(y<=high))),
                "mpiw95":float(np.mean(high-low)),
                "interval_outside_physics_set_rate":float(np.mean((low<np.asarray(lo)-1e-9)|(high>float(mu_upper)+1e-9))),
            })
        else:
            out.update({"picp95":out["picp95_raw"],"mpiw95":out["mpiw95_raw"]})
    return out


def _deep_loader(X,y,lo,batch):
    return DataLoader(TensorDataset(torch.from_numpy(X),torch.from_numpy(y),torch.from_numpy(lo)),batch_size=batch,shuffle=True)


def _optimizer(name, params, lr, weight_decay):
    if str(name).lower()=="adam":
        return torch.optim.Adam(params,lr=lr,weight_decay=weight_decay)
    if str(name).lower()=="adamw":
        return torch.optim.AdamW(params,lr=lr,weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer: {name}")


def _fit_deterministic(model,Xtr,ytr,Xv,yv,hp,epochs):
    dev=device(); model=model.to(dev)
    opt=_optimizer(hp.get("optimizer","adamw"),model.parameters(),float(hp["lr"]),float(hp["weight_decay"]))
    dl=DataLoader(TensorDataset(torch.from_numpy(Xtr),torch.from_numpy(ytr)),batch_size=int(hp["batch_size"]),shuffle=True)
    xv=torch.from_numpy(Xv).to(dev); yv_t=torch.from_numpy(yv).to(dev)
    best=None; bestloss=float("inf"); bad=0; patience=int(hp.get("patience",10))
    for _ in range(int(epochs)):
        model.train()
        for xb,yb in dl:
            xb=xb.to(dev); yb=yb.to(dev); opt.zero_grad(); pred=model(xb)
            loss=nn.functional.mse_loss(pred,yb); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5.0); opt.step()
        model.eval()
        with torch.no_grad(): vl=nn.functional.mse_loss(model(xv),yv_t).item()
        if vl<bestloss-1e-7:
            bestloss=vl; best={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}; bad=0
        else: bad+=1
        if bad>=patience: break
    if best: model.load_state_dict(best)
    return model


def literature_hparams(name,cfg,overrides=None,preset="paper"):
    tr=cfg["training"]
    base={
        "sequence_length":int(cfg.get("sequence_length",64)), "scaler":"standard",
        "optimizer":"adamw", "lr":float(tr["lr"]), "weight_decay":float(tr["weight_decay"]),
        "batch_size":int(tr["batch_size"]), "dropout":float(tr.get("dropout",.1)),
        "patience":int(tr.get("patience",10)),
        "epochs":int(tr["epochs_quick" if preset=="quick" else "epochs_paper"]),
    }
    if preset in ("trust", "paper"):
        # Trust/paper modes use the literature-specific preprocessing/context and
        # full architecture.  Trust mode caps only the training duration so a
        # Kaggle validation run remains practical.
        base.update(cfg.get("baseline",{}).get(name,{}))
        if preset == "trust":
            base["epochs"] = int(cfg["training"].get("epochs_trust", 30))
            base["patience"] = int(cfg["training"].get("patience_trust", min(int(base.get("patience",10)), 10)))
    elif name=="todorovic2022_cnn":
        # Quick mode stays lightweight while preserving the layer pattern only;
        # it is a software smoke test, not a scientific comparison.
        base["sequence_length"]=max(16,min(100,int(cfg.get("sequence_length",64))))
    if overrides: base.update(overrides)
    return base


def fit_literature(name,b:Bundle,cfg,epochs=None,preset="paper",hp_overrides=None):
    hp=literature_hparams(name,cfg,hp_overrides,preset)
    epochs=int(epochs or hp["epochs"])
    if name=="chen2025_svdkl":
        from .svdkl import fit_svdkl
        return fit_svdkl(b.Xtr,b.ytr,b.Xv,b.yv,hidden=int(hp.get("hidden",64)),feature_dim=int(hp.get("feature_dim",16)),
                         inducing=int(hp.get("inducing",128)),dropout=float(hp["dropout"]),lr=float(hp["lr"]),
                         weight_decay=float(hp["weight_decay"]),batch_size=int(hp["batch_size"]),epochs=epochs,
                         patience=int(hp["patience"]),device=device(),optimizer=hp.get("optimizer","adamw"))
    model=make_literature_baseline(name,len(b.features),sequence_length=b.sequence_length,
                                   debug_scale=(preset=="quick"),dropout=float(hp["dropout"]),
                                   hidden=hp.get("hidden"),layers=int(hp.get("layers",2)),heads=int(hp.get("heads",4)),
                                   ff_mult=int(hp.get("ff_mult",2)))
    return _fit_deterministic(model,b.Xtr,b.ytr,b.Xv,b.yv,hp,epochs)


def predict_literature(model,name,X,batch=1024):
    if name=="chen2025_svdkl":
        from .svdkl import predict_svdkl
        return predict_svdkl(model,X,batch_size=batch,device=device())
    dev=device(); model.eval(); ps=[]
    with torch.no_grad():
        for i in range(0,len(X),batch): ps.append(model(torch.from_numpy(X[i:i+batch]).to(dev)).cpu().numpy())
    return np.concatenate(ps),None


def _proposal_model(variant,d,hp):
    common=dict(hidden=int(hp["hidden"]),dropout=float(hp["dropout"]),blocks=int(hp["tcn_blocks"]),kernel_size=int(hp["kernel_size"]))
    if variant=="safegrip_no_uq": return DeterministicTCN(d,**common)
    if variant=="safegrip_no_temporal": return SafeGripNoTemporal(d,hidden=common["hidden"],dropout=common["dropout"])
    return SafeGripNet(d,**common)


def proposal_hparams(cfg, overrides=None):
    p=dict(cfg.get("proposal",{})); tr=cfg["training"]
    defaults={
        "hidden":tr.get("hidden",64),"dropout":tr.get("dropout",.1),"tcn_blocks":4,"kernel_size":3,
        "lr":tr.get("lr",1e-3),"weight_decay":tr.get("weight_decay",1e-4),"batch_size":tr.get("batch_size",256),
        "lambda_mse":.20,"lambda_physics":.05,
    }
    defaults.update(p); defaults.update(overrides or {}); return defaults


def fit_proposal(variant,b:Bundle,cfg,epochs,hp_overrides=None):
    if variant not in PROPOSAL_VARIANTS: raise ValueError(variant)
    hp=proposal_hparams(cfg,hp_overrides); dev=device(); model=_proposal_model(variant,len(b.features),hp).to(dev)
    opt=torch.optim.AdamW(model.parameters(),lr=float(hp["lr"]),weight_decay=float(hp["weight_decay"]))
    # Calibration labels are never used in gradient training.  The soft physics
    # penalty always uses the uncalibrated mechanics-derived lower endpoint from
    # the training split.  The calibrated bound is applied only for validation
    # model selection and final inference.
    lo_train=b.raw_lotr
    lo_val=b.raw_lov if variant=="safegrip_no_calibration" else b.lov
    dl=_deep_loader(b.Xtr,b.ytr,lo_train,int(hp["batch_size"]))
    xv=torch.from_numpy(b.Xv).to(dev); yv=torch.from_numpy(b.yv).to(dev); lov=torch.from_numpy(lo_val).to(dev)
    mu_u=float(cfg["mu_upper"]); best=None; bestloss=float("inf"); bad=0; patience=int(cfg["training"]["patience"])
    for _ in range(int(epochs)):
        model.train()
        for xb,yb,lb in dl:
            xb=xb.to(dev); yb=yb.to(dev); lb=lb.to(dev); ub=torch.full_like(lb,mu_u); opt.zero_grad()
            if variant=="safegrip_no_uq":
                raw=model(xb)
                # Controlled deterministic ablation: train on labels only; hard
                # projection is an inference-time safety map, not a gradient path.
                loss=nn.functional.mse_loss(raw,yb)
            else:
                raw,logsig=model(xb)
                sig=torch.exp(logsig)
                # The Gaussian head is centered at the raw network mean.  Hard
                # projection is non-Gaussian and is therefore kept out of NLL.
                nll=(.5*((yb-raw)/sig)**2+logsig).mean()
                loss=nll+float(hp["lambda_mse"])*nn.functional.mse_loss(raw,yb)
            if variant not in ("safegrip_no_physics_loss","safegrip_data_only"):
                phys=torch.relu(lb-raw).pow(2).mean()+.25*torch.relu(raw-mu_u).pow(2).mean()
                loss=loss+float(hp["lambda_physics"])*phys
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5.0); opt.step()
        model.eval()
        with torch.no_grad():
            if variant=="safegrip_no_uq":
                raw=model(xv); pv=project_torch(raw,lov,torch.full_like(lov,mu_u))
            else:
                raw,_=model(xv); pv=raw if variant in ("safegrip_no_projection","safegrip_data_only") else project_torch(raw,lov,torch.full_like(lov,mu_u))
            vl=nn.functional.mse_loss(pv,yv).item()
        if vl<bestloss-1e-7:
            bestloss=vl; best={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}; bad=0
        else: bad+=1
        if bad>=patience: break
    if best: model.load_state_dict(best)
    return model,hp


def predict_proposal_details(model,variant,X,lo,raw_lo,mu_upper,batch=1024):
    dev=device(); model.eval(); ps=[]; raws=[]; ss=[]
    bound=np.asarray(raw_lo if variant=="safegrip_no_calibration" else lo,dtype=np.float32)
    with torch.no_grad():
        for i in range(0,len(X),batch):
            xb=torch.from_numpy(X[i:i+batch]).to(dev)
            lb=torch.from_numpy(bound[i:i+batch]).to(dev)
            ub=torch.full_like(lb,float(mu_upper))
            if variant=="safegrip_no_uq":
                raw=model(xb); p=project_torch(raw,lb,ub)
            else:
                raw,logsig=model(xb)
                p=raw if variant in ("safegrip_no_projection","safegrip_data_only") else project_torch(raw,lb,ub)
                ss.append(torch.exp(logsig).cpu().numpy())
            raws.append(raw.cpu().numpy()); ps.append(p.cpu().numpy())
    prediction=np.concatenate(ps); raw_mean=np.concatenate(raws)
    sigma=np.concatenate(ss) if ss else None
    details={"prediction":prediction,"raw_mean":raw_mean,"sigma":sigma,"bound":bound}
    if sigma is not None:
        low_raw,high_raw=gaussian_interval(raw_mean,sigma)
        low_phys,high_phys=project_interval_numpy(low_raw,high_raw,bound,float(mu_upper))
        details.update({"pi95_low_raw":low_raw,"pi95_high_raw":high_raw,
                        "pi95_low_physics":low_phys,"pi95_high_physics":high_phys})
    return details


def predict_proposal(model,variant,X,lo,raw_lo,mu_upper,batch=1024):
    d=predict_proposal_details(model,variant,X,lo,raw_lo,mu_upper,batch=batch)
    return d["prediction"],d["sigma"],d["bound"]


def _export_literature_manifest(out:Path,names):
    rows=[]
    for n in names:
        x={"model":n,**LITERATURE_BASELINES[n]}; rows.append(x)
    pd.DataFrame(rows).to_csv(out/"baseline_manifest.csv",index=False)
    (out/"literature_only.json").write_text(json.dumps(LITERATURE_ONLY,indent=2),encoding="utf-8")


def _save_bundle_meta(out,b,cfg):
    dump(b.scaler,out/"scaler.joblib")
    (out/"features.json").write_text(json.dumps(b.features,indent=2),encoding="utf-8")
    (out/"calibration.json").write_text(json.dumps({"one_sided_q":b.q,"alpha":cfg["alpha"],"test_used_for_calibration":False,
                                                     "sequence_length":b.sequence_length,"eval_start":b.eval_start,
                                                     "physics_window_samples":int(cfg.get("physics",{}).get("window_samples",1)),
                                                     "calibration_labels_used_in_gradient_training":False,
                                                     "scaler":b.scaler_kind},indent=2),encoding="utf-8")


def _assert_same_targets(reference: Bundle, candidate: Bundle, name: str):
    if reference.idt.shape!=candidate.idt.shape or not np.array_equal(reference.idt,candidate.idt):
        raise RuntimeError(f"{name} is not evaluated on the same test endpoint IDs; check trip grouping/common warm-up/stride")
    if reference.idv.shape!=candidate.idv.shape or not np.array_equal(reference.idv,candidate.idv):
        raise RuntimeError(f"{name} is not evaluated on the same validation endpoint IDs; check trip grouping/common warm-up/stride")
    if not np.allclose(reference.yt,candidate.yt,equal_nan=True) or not np.allclose(reference.yv,candidate.yv,equal_nan=True):
        raise RuntimeError(f"{name} endpoint IDs match but targets differ; prepared data are inconsistent")


def _aggregate_seed_metrics(rows):
    df=pd.DataFrame(rows)
    out=[]
    id_cols=[c for c in ("model","kind","doi") if c in df]
    for keys,g in df.groupby(id_cols,dropna=False,sort=False):
        if not isinstance(keys,tuple): keys=(keys,)
        row=dict(zip(id_cols,keys)); row["n_seeds"]=int(g["seed"].nunique()) if "seed" in g else len(g)
        for c in g.columns:
            if c in id_cols or c=="seed" or not pd.api.types.is_numeric_dtype(g[c]): continue
            row[c]=float(g[c].mean())
            row[c+"_std"]=float(g[c].std(ddof=1)) if len(g)>1 else 0.0
        out.append(row)
    return pd.DataFrame(out)


def _sanity_baselines(bundle: Bundle):
    """Train-only constant baselines used to detect misleading negative-R2 wins."""
    train_mean = float(np.mean(bundle.ytr))
    train_median = float(np.median(bundle.ytr))
    rows = []
    for name, value in (("train_mean", train_mean), ("train_median", train_median)):
        pred = np.full_like(bundle.yt, value, dtype=float)
        rows.append({"model": name, "kind": "sanity_constant_baseline", **regression_metrics(bundle.yt, pred)})
    return pd.DataFrame(rows)


def _result_health(bundle: Bundle, metrics: pd.DataFrame, preds: pd.DataFrame, preset: str, cfg: dict) -> dict:
    """Machine-readable scientific sanity gates.

    These gates do not manufacture a positive result.  They flag runs whose
    apparent error is dominated by a constant projection, too few endpoints, or
    performance no better than a train-only constant baseline.
    """
    row = metrics.loc[metrics["model"] == "safegrip"].iloc[0]
    y = np.asarray(bundle.yt, float)
    p = np.asarray(preds["safegrip"], float)
    raw = np.asarray(preds.get("safegrip_raw", p), float)
    target_std = float(np.std(y))
    pred_std = float(np.std(p))
    sanity = _sanity_baselines(bundle)
    mean_rmse = float(sanity.loc[sanity.model == "train_mean", "rmse"].iloc[0])
    min_n = int(cfg.get("health",{}).get("min_test_endpoints_trust", 200 if preset != "quick" else 50))
    projection_rate = float(row.get("projection_correction_rate", np.nan))
    raw_bound = np.asarray(bundle.raw_lot, float)
    alpha=float(cfg.get("alpha",0.05))
    lower_violation=float(row.get("lower_violation_rate",np.nan))
    # Finite-sample tolerance for a one-sided nominal alpha violation rate.
    coverage_tol=max(0.02,2.0*np.sqrt(max(alpha*(1-alpha),1e-12)/max(len(y),1)))
    checks = {
        "finite_predictions": bool(np.isfinite(p).all()),
        "enough_test_endpoints": bool(len(y) >= min_n),
        "final_prediction_not_constant": bool(pred_std > max(1e-4, 0.05 * target_std)),
        "not_projection_dominated": bool(not np.isfinite(projection_rate) or projection_rate < float(cfg.get("health",{}).get("max_projection_correction_rate", 0.95))),
        "beats_train_mean_rmse": bool(float(row["rmse"]) < mean_rmse),
        "positive_r2": bool(float(row["r2"]) > 0.0),
        "raw_bound_finite": bool(np.isfinite(raw_bound).all()),
        "calibrated_lower_coverage_consistent": bool(np.isfinite(lower_violation) and lower_violation <= alpha + coverage_tol),
    }
    return {
        "status": "PASS" if all(checks.values()) else "REVIEW",
        "preset": preset,
        "checks": checks,
        "diagnostics": {
            "n_test_endpoints": int(len(y)),
            "target_std": target_std,
            "prediction_std": pred_std,
            "raw_prediction_std": float(np.std(raw)),
            "projection_correction_rate": projection_rate,
            "safegrip_rmse": float(row["rmse"]),
            "safegrip_r2": float(row["r2"]),
            "train_mean_rmse": mean_rmse,
            "physics_lower_median": float(np.median(raw_bound)),
            "physics_lower_p99": float(np.quantile(raw_bound, 0.99)),
            "calibrated_lower_violation_rate": lower_violation,
            "nominal_alpha": alpha,
            "coverage_tolerance": coverage_tol,
        },
        "interpretation": "PASS means the run clears automatic degeneracy/sanity gates; it does not replace multi-seed statistical analysis or external validation.",
    }


def _sha256_file(path: Path) -> str:
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()


def _write_reproducibility_manifest(out: Path, csv_path, cfg: dict, preset: str):
    csv_path=Path(csv_path)
    source_file=csv_path.parent.parent.parent/"raw"/csv_path.parent.name/"SOURCE.json"
    source_meta=None
    if source_file.exists():
        try: source_meta=json.loads(source_file.read_text(encoding="utf-8"))
        except Exception: source_meta={"path":str(source_file)}
    manifest={
        "preset":preset,
        "processed_csv":str(csv_path),
        "processed_csv_sha256":_sha256_file(csv_path),
        "source_metadata":source_meta,
        "python":platform.python_version(),
        "torch":torch.__version__,
        "numpy":np.__version__,
        "pandas":pd.__version__,
        "config":cfg,
    }
    (out/"reproducibility_manifest.json").write_text(json.dumps(manifest,indent=2,default=str),encoding="utf-8")


def run_benchmark(csv_path,out_dir,cfg,preset="quick",models=None,hp_overrides=None,baseline_hparams=None):
    out=ensure_dir(out_dir)
    _write_reproducibility_manifest(out,csv_path,cfg,preset)
    baseline_names=list(models) if models is not None else list(QUICK_BASELINES if preset=="quick" else PAPER_BASELINES)
    validate_paper_baselines(baseline_names); _export_literature_manifest(out,baseline_names)
    proposal_seq=int((hp_overrides or {}).get("sequence_length",cfg["sequence_length"]))
    baseline_plan = {
        name: literature_hparams(name, cfg, (baseline_hparams or {}).get(name, {}), preset)
        for name in baseline_names
    }
    # Never allow an explicit warm-up that is shorter than one comparator's
    # context; otherwise different methods silently evaluate different labels.
    start=max(
        common_eval_start(cfg),
        proposal_seq - 1,
        max([int(hp["sequence_length"]) - 1 for hp in baseline_plan.values()] or [0]),
    )
    proposal_bundle=make_bundle(csv_path,cfg,sequence_length=proposal_seq,scaler_kind="standard",eval_start=start)
    _save_bundle_meta(out,proposal_bundle,cfg)
    if preset == "paper":
        seed_list=list(cfg.get("evaluation",{}).get("seeds",[cfg["seed"]]))
    elif preset == "trust":
        seed_list=list(cfg.get("evaluation",{}).get("trust_seeds",[0,1,2]))
    else:
        seed_list=[int(cfg["seed"])]
    include_projection=bool(cfg.get("evaluation",{}).get("projection_parity_controls",True)) and preset in ("trust","paper")
    epoch_key = "epochs_quick" if preset=="quick" else ("epochs_trust" if preset=="trust" else "epochs_paper")
    epochs_proposal=int(cfg["training"].get(epoch_key, cfg["training"].get("epochs_paper",60)))

    per_seed=[]; control_rows=[]
    preds=pd.DataFrame({"endpoint_id":proposal_bundle.idt,"y_true":proposal_bundle.yt,
                        "physics_lower":proposal_bundle.lot,"physics_lower_raw":proposal_bundle.raw_lot})
    control_preds=pd.DataFrame({"endpoint_id":proposal_bundle.idt,"y_true":proposal_bundle.yt,"physics_lower":proposal_bundle.lot})
    selected={}; scaler_dir=ensure_dir(out/"baseline_scalers")

    for name in baseline_names:
        hp=dict(baseline_plan[name]); selected[name]=hp
        b=make_bundle(csv_path,cfg,sequence_length=int(hp["sequence_length"]),scaler_kind=str(hp["scaler"]),eval_start=start)
        _assert_same_targets(proposal_bundle,b,name); dump(b.scaler,scaler_dir/f"{name}.joblib")
        print(f"[baseline/adapted] {name} DOI={LITERATURE_BASELINES[name]['doi']} L={b.sequence_length} scaler={b.scaler_kind}")
        seed_preds=[]; seed_sigmas=[]; seed_control=[]
        for seed in seed_list:
            seed_everything(int(seed)); t0=time.time()
            model=fit_literature(name,b,cfg,None,preset,hp); p,sig=predict_literature(model,name,b.Xt)
            per_seed.append({"model":name,"kind":"literature_adapted_baseline","doi":LITERATURE_BASELINES[name]["doi"],"seed":int(seed),
                             **regression_metrics(b.yt,p,b.lot,cfg["mu_upper"],sig,raw_mean=p,project_uncertainty=False),
                             "seconds":time.time()-t0})
            seed_preds.append(p)
            if sig is not None: seed_sigmas.append(sig)
            if include_projection:
                pp=project_numpy(p,b.lot,float(cfg["mu_upper"])); seed_control.append(pp)
                control_rows.append({"model":name+"__projection_control","source_model":name,"kind":"postprocessing_parity_control",
                                     "doi":LITERATURE_BASELINES[name]["doi"],"seed":int(seed),
                                     **regression_metrics(b.yt,pp,b.lot,cfg["mu_upper"],sig,raw_mean=p,project_uncertainty=True)})
        preds[name]=np.mean(np.stack(seed_preds),axis=0)
        if seed_sigmas: preds[name+"_sigma"]=np.mean(np.stack(seed_sigmas),axis=0)
        if seed_control: control_preds[name+"__projection_control"]=np.mean(np.stack(seed_control),axis=0)

    name="safegrip"; print(f"[proposal] safegrip L={proposal_bundle.sequence_length}")
    proposal_preds=[]; proposal_raw=[]; proposal_sigmas=[]; final_hp=None
    seed_pi_low=[]; seed_pi_high=[]
    for seed in seed_list:
        seed_everything(int(seed)); t0=time.time(); model,final_hp=fit_proposal(name,proposal_bundle,cfg,epochs_proposal,hp_overrides)
        d=predict_proposal_details(model,name,proposal_bundle.Xt,proposal_bundle.lot,proposal_bundle.raw_lot,cfg["mu_upper"])
        per_seed.append({"model":name,"kind":"proposal","doi":"","seed":int(seed),
                         **regression_metrics(proposal_bundle.yt,d["prediction"],d["bound"],cfg["mu_upper"],d["sigma"],
                                              raw_mean=d["raw_mean"],project_uncertainty=True),
                         "seconds":time.time()-t0})
        proposal_preds.append(d["prediction"]); proposal_raw.append(d["raw_mean"]); proposal_sigmas.append(d["sigma"])
        seed_pi_low.append(d["pi95_low_physics"]); seed_pi_high.append(d["pi95_high_physics"])
    preds[name]=np.mean(np.stack(proposal_preds),axis=0)
    preds[name+"_raw"]=np.mean(np.stack(proposal_raw),axis=0)
    preds[name+"_sigma"]=np.mean(np.stack(proposal_sigmas),axis=0)
    preds[name+"_pi95_low_physics"]=np.mean(np.stack(seed_pi_low),axis=0)
    preds[name+"_pi95_high_physics"]=np.mean(np.stack(seed_pi_high),axis=0)

    by_seed=pd.DataFrame(per_seed); metrics=_aggregate_seed_metrics(per_seed)
    metrics.to_csv(out/"metrics.csv",index=False); by_seed.to_csv(out/"metrics_by_seed.csv",index=False); preds.to_csv(out/"predictions.csv",index=False)
    (out/"proposal_hparams.json").write_text(json.dumps(final_hp,indent=2),encoding="utf-8")
    (out/"baseline_selected_hparams.json").write_text(json.dumps(selected,indent=2),encoding="utf-8")
    (out/"evaluation_protocol.json").write_text(json.dumps({
        "seeds":seed_list,"common_eval_start":start,"common_eval_warmup_samples":start+1,
        "same_validation_test_endpoints":True,"endpoint_identity_checked":True,
        "windows_grouped_by_segment":True,"primary_selection_metric":"validation RMSE",
        "physics_window_samples":int(cfg.get("physics",{}).get("window_samples",1)),
        "calibration_labels_used_in_gradient_training":False,
        "physical_claim":"conditional on configured bounded-error and mu_upper assumptions",
    },indent=2),encoding="utf-8")
    sanity = _sanity_baselines(proposal_bundle)
    sanity.to_csv(out/"sanity_baselines.csv", index=False)
    health = _result_health(proposal_bundle, metrics, preds, preset, cfg)
    (out/"result_health.json").write_text(json.dumps(health, indent=2), encoding="utf-8")
    if control_rows:
        _aggregate_seed_metrics(control_rows).to_csv(out/"projection_control_metrics.csv",index=False)
        pd.DataFrame(control_rows).to_csv(out/"projection_control_metrics_by_seed.csv",index=False)
        control_preds.to_csv(out/"projection_control_predictions.csv",index=False)
    return metrics


def run_ablation(csv_path,out_dir,cfg,preset="paper",variants=None,hp_overrides=None):
    out=ensure_dir(out_dir)
    seq=(hp_overrides or {}).get("sequence_length")
    b=make_bundle(csv_path,cfg,sequence_length=seq,scaler_kind="standard",eval_start=common_eval_start(cfg)); _save_bundle_meta(out,b,cfg)
    variants=list(variants or PROPOSAL_VARIANTS)
    epoch_key = "epochs_quick" if preset=="quick" else ("epochs_trust" if preset=="trust" else "epochs_paper")
    epochs=int(cfg["training"].get(epoch_key, cfg["training"].get("epochs_paper",60)))
    if preset == "paper":
        seed_list=list(cfg.get("evaluation",{}).get("seeds",[cfg["seed"]]))
    elif preset == "trust":
        seed_list=list(cfg.get("evaluation",{}).get("trust_seeds",[0,1,2]))
    else:
        seed_list=[int(cfg["seed"])]
    results=[]; preds=pd.DataFrame({"endpoint_id":b.idt,"y_true":b.yt,"physics_lower":b.lot,"physics_lower_raw":b.raw_lot})
    for variant in variants:
        print(f"[ablation] {variant}")
        variant_preds=[]; variant_raw=[]; variant_sigma=[]
        for seed in seed_list:
            seed_everything(int(seed)); t0=time.time(); model,hp=fit_proposal(variant,b,cfg,epochs,hp_overrides)
            d=predict_proposal_details(model,variant,b.Xt,b.lot,b.raw_lot,cfg["mu_upper"])
            row={"model":variant,"kind":"ablation" if variant!="safegrip" else "proposal","seed":int(seed),
                 **regression_metrics(b.yt,d["prediction"],d["bound"],cfg["mu_upper"],d["sigma"],
                                      raw_mean=d["raw_mean"],
                                      project_uncertainty=(variant not in ("safegrip_no_projection","safegrip_data_only"))),
                 "seconds":time.time()-t0}
            results.append(row); variant_preds.append(d["prediction"]); variant_raw.append(d["raw_mean"])
            if d["sigma"] is not None: variant_sigma.append(d["sigma"])
        preds[variant]=np.mean(np.stack(variant_preds),axis=0)
        preds[variant+"_raw"]=np.mean(np.stack(variant_raw),axis=0)
        if variant_sigma: preds[variant+"_sigma"]=np.mean(np.stack(variant_sigma),axis=0)
    by_seed=pd.DataFrame(results); summary=_aggregate_seed_metrics(results)
    summary.to_csv(out/"ablation_metrics.csv",index=False); by_seed.to_csv(out/"ablation_metrics_by_seed.csv",index=False); preds.to_csv(out/"ablation_predictions.csv",index=False)
    (out/"ablation_design.json").write_text(json.dumps({
        "safegrip_data_only":"remove hard projection and soft physics loss; retain the same TCN + UQ backbone",
        "safegrip_no_projection":"remove hard identified-set projection; keep UQ + soft physics",
        "safegrip_no_uq":"replace heteroscedastic NLL by one deterministic MSE objective; keep hard projection",
        "safegrip_no_physics_loss":"remove soft physics penalty; keep hard projection + UQ",
        "safegrip_no_calibration":"use raw mechanics lower bound; remove one-sided statistical relaxation",
        "safegrip_no_temporal":"replace TCN by last-state MLP; keep projection + UQ",
        "safegrip":"full calibrated physics-constrained partial-identification estimator",
        "seeds":seed_list,"same_hyperparameters_across_variants":True,
    },indent=2),encoding="utf-8")
    return summary

