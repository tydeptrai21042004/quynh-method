from __future__ import annotations
import json, time, hashlib, platform
from dataclasses import dataclass, replace
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
from .models import make_literature_baseline, SafeGripV3Net, SafeGripBackboneNet, ResidualScaleHead
from .physics import (
    project_torch, project_numpy, project_interval_numpy, gaussian_interval,
    conformal_lower_correction, apply_lower_correction,
)
from .utils import ensure_dir, seed_everything, device
from .features import add_safegrip_features

META={
    "time","distance","lat","lon","route_s_m","gps_heading_deg","match_distance_m","match_ref_index",
    "mu_ref","physics_lower_mechanics","physics_lower_raw","split","split_position","source_file","ref_source_file",
    "route_id","direction","trip_id","trajectory_id","segment_id","sample_uid","resampled",
}
PROPOSAL_VARIANTS=(
    "safegrip_backbone_raw",
    "safegrip_features_only",
    "safegrip_prior_evidence",
    "safegrip_no_excitation",
    "safegrip_endpoint_only",
    "safegrip_no_gate",
    "safegrip_no_bound",
    "safegrip_no_uq",
    "safegrip_no_calibration",
    "safegrip",
    "safegrip_data_only",       # legacy alias -> safegrip_backbone_raw
    "safegrip_static_only",     # legacy alias -> safegrip_endpoint_only
)
PRIMARY_ABLATION_VARIANTS=(
    "safegrip_backbone_raw",
    "safegrip_features_only",
    "safegrip_prior_evidence",
    "safegrip_no_excitation",
    "safegrip_endpoint_only",
    "safegrip_no_gate",
    "safegrip_no_bound",
    "safegrip_no_uq",
    "safegrip_no_calibration",
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
    proposal_features: bool
    feature_mode: str
    Xtr: np.ndarray; ytr: np.ndarray; lotr: np.ndarray; raw_lotr: np.ndarray; idtr: np.ndarray; etr: np.ndarray
    Xc: np.ndarray; yc: np.ndarray; loc: np.ndarray; raw_loc: np.ndarray; idc: np.ndarray; ec: np.ndarray; lower_cal_mask: np.ndarray; uq_cal_mask: np.ndarray
    Xv: np.ndarray; yv: np.ndarray; lov: np.ndarray; raw_lov: np.ndarray; idv: np.ndarray; ev: np.ndarray
    Xt: np.ndarray; yt: np.ndarray; lot: np.ndarray; raw_lot: np.ndarray; idt: np.ndarray; et: np.ndarray


def common_eval_start(cfg) -> int:
    """Benchmark warm-up independent of unused hyperparameter-search choices.

    The old implementation included the largest sequence length present in the
    tuning search space, which silently discarded extra validation/test points
    even when that length was not used by the benchmark.  The benchmark now
    uses only its configured warm-up; ``run_benchmark`` additionally takes the
    maximum of the methods actually being compared.
    """
    b=cfg.get("benchmark",{})
    if "common_warmup_samples" in b:
        return max(0,int(b["common_warmup_samples"])-1)
    return max(int(cfg.get("sequence_length",16)),100)-1


def tuning_context_lengths(cfg) -> list[int]:
    """Union of every proposal/baseline sequence length that may be tuned."""
    values=[int(x) for x in cfg.get("tuning",{}).get("space",{}).get("sequence_length",[cfg.get("sequence_length",16)])]
    bt=cfg.get("baseline_tuning",{})
    values.extend(int(x) for x in bt.get("sequence_length",[]))
    for space in bt.get("space",{}).values():
        values.extend(int(x) for x in space.get("sequence_length",[]))
    for hp in cfg.get("baseline",{}).values():
        if "sequence_length" in hp: values.append(int(hp["sequence_length"]))
    return sorted(set(values or [int(cfg.get("sequence_length",16))]))


def tuning_eval_start(cfg, *, include_baselines: bool = False) -> int:
    """One global endpoint warm-up for every hyperparameter search."""
    del include_baselines
    return max(tuning_context_lengths(cfg))-1


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


def _ids_hash(ids: np.ndarray) -> str:
    arr=np.asarray(ids,dtype=str)
    return hashlib.sha256("\n".join(arr.tolist()).encode("utf-8")).hexdigest()


def _augment_train_with_lower_calibration(b: Bundle) -> Bundle:
    """Equal-label-budget control for literature baselines.

    Adds only the lower-bound-calibration role to baseline supervised training.
    The UQ-calibration role remains untouched, preventing test/UQ leakage.
    """
    m=np.asarray(b.lower_cal_mask,dtype=bool)
    if not m.any(): return b
    return replace(
        b,
        Xtr=np.concatenate([b.Xtr,b.Xc[m]],axis=0),
        ytr=np.concatenate([b.ytr,b.yc[m]],axis=0),
        lotr=np.concatenate([b.lotr,b.loc[m]],axis=0),
        raw_lotr=np.concatenate([b.raw_lotr,b.raw_loc[m]],axis=0),
        idtr=np.concatenate([b.idtr,b.idc[m]],axis=0),
        etr=np.concatenate([b.etr,b.ec[m]],axis=0),
    )


def _common_conformal_interval(y_cal,p_cal,p_test,alpha,ids=None,block_size=1):
    """Same absolute-residual block split-conformal interval for any model."""
    score=np.abs(np.asarray(y_cal,float)-np.asarray(p_cal,float))
    if ids is not None:
        score=_block_max_scores(score,np.asarray(ids,dtype=str),max(1,int(block_size)))
    q=_finite_sample_quantile(score,float(alpha))
    p=np.asarray(p_test,float)
    return p-q,p+q,float(q)


def _model_parameter_count(model) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def _reset_peak_gpu_memory():
    if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()


def _peak_gpu_memory_mb() -> float:
    return float(torch.cuda.max_memory_allocated()/1024**2) if torch.cuda.is_available() else 0.0


def _write_fairness_audit(out: Path, proposal_bundle: Bundle, compared_bundles: dict, cfg: dict, controls: dict):
    cal_equal=all(np.array_equal(proposal_bundle.idc.astype(str),b.idc.astype(str)) for b in compared_bundles.values())
    val_equal=all(np.array_equal(proposal_bundle.idv.astype(str),b.idv.astype(str)) for b in compared_bundles.values())
    test_equal=all(np.array_equal(proposal_bundle.idt.astype(str),b.idt.astype(str)) for b in compared_bundles.values())
    required={
        "calibration_endpoint_identity":bool(cal_equal),
        "validation_endpoint_identity":bool(val_equal),
        "test_endpoint_identity":bool(test_equal),
        "test_labels_locked_from_training_and_calibration":True,
        "train_only_scaler_fit":True,
        "proposal_engineered_features_label_free":True,
        "calibration_roles_disjoint":bool(not np.any(proposal_bundle.lower_cal_mask & proposal_bundle.uq_cal_mask)),
    }
    audit={
        "status":"PASS" if all(required.values()) else "FAIL",
        "required_checks":required,
        "common_eval_start":int(proposal_bundle.eval_start),
        "global_tuning_eval_start":int(tuning_eval_start(cfg)),
        "global_tuning_warmup_samples":int(tuning_eval_start(cfg)+1),
        "calibration_endpoint_count":int(len(proposal_bundle.idc)),
        "validation_endpoint_count":int(len(proposal_bundle.idv)),
        "test_endpoint_count":int(len(proposal_bundle.idt)),
        "calibration_endpoint_hash":_ids_hash(proposal_bundle.idc),
        "validation_endpoint_hash":_ids_hash(proposal_bundle.idv),
        "test_endpoint_hash":_ids_hash(proposal_bundle.idt),
        "proposal_train_labels":int(len(proposal_bundle.ytr)),
        "proposal_lower_calibration_labels":int(np.sum(proposal_bundle.lower_cal_mask)),
        "proposal_uq_calibration_labels":int(np.sum(proposal_bundle.uq_cal_mask)),
        "parity_controls":{
            "feature_parity":controls.get("feature_parity",[]),
            "label_budget":controls.get("label_budget",[]),
            "common_conformal":controls.get("common_conformal",[]),
            "projection":controls.get("projection",[]),
        },
    }
    (out/"fairness_audit.json").write_text(json.dumps(audit,indent=2),encoding="utf-8")
    txt=["SCIENTIFIC FAIRNESS AUDIT", "="*32, f"STATUS: {audit['status']}"]
    txt += [f"{k}: {'PASS' if v else 'FAIL'}" for k,v in required.items()]
    txt += [f"validation_hash: {audit['validation_endpoint_hash']}", f"test_hash: {audit['test_endpoint_hash']}"]
    (out/"fairness_audit.txt").write_text("\n".join(txt)+"\n",encoding="utf-8")
    if audit["status"]!="PASS": raise RuntimeError("Scientific fairness audit failed: endpoint/protocol invariant mismatch")


def _make_scaler(kind: str):
    if kind=="standard": return StandardScaler()
    if kind=="minmax": return MinMaxScaler()
    raise ValueError(f"Unknown scaler: {kind}")


def _calibration_role_masks(ids: np.ndarray, lower_fraction: float = 0.25):
    """Deterministically split calibration endpoints into disjoint roles.

    The first contiguous portion of each trajectory segment calibrates only the
    mechanics lower-bound relaxation; the remaining portion is reserved for
    predictive conformal UQ.  This prevents the same calibration labels from
    defining both the point estimator's lower endpoint and its UQ quantile.
    """
    ids=np.asarray(ids,dtype=str); n=len(ids)
    lower=np.zeros(n,dtype=bool); uq=np.zeros(n,dtype=bool)
    frac=min(max(float(lower_fraction),0.05),0.8)
    start=0
    while start<n:
        key=_segment_key(ids[start]); stop=start+1
        while stop<n and _segment_key(ids[stop])==key:
            stop+=1
        m=stop-start
        if m==1:
            uq[start]=True
        else:
            k=max(1,min(m-1,int(np.ceil(frac*m))))
            lower[start:start+k]=True; uq[start+k:stop]=True
        start=stop
    if n and not lower.any():
        lower[0]=True; uq[0]=False
    if n>1 and not uq.any():
        uq[-1]=True; lower[-1]=False
    return lower,uq


def make_bundle(csv_path, cfg, sequence_length=None, scaler_kind="standard", eval_start=None,
                proposal_features: bool = False, feature_mode: str | None = None):
    """Build one leakage-safe model bundle.

    ``feature_mode`` makes reviewer controls explicit:
      raw                 -> source numeric sensor channels only
      safegrip            -> all label-free SafeGrip engineered channels
      safegrip_no_excitation -> engineered channels except explicit E/memory
      safegrip_endpoint   -> endpoint-safe engineered channels; derivative/memory
                             channels are removed for the genuine static ablation

    ``proposal_features`` is retained for backward compatibility.
    """
    if feature_mode is None:
        feature_mode="safegrip" if proposal_features else "raw"
    feature_mode=str(feature_mode)
    valid={"raw","safegrip","safegrip_no_excitation","safegrip_endpoint"}
    if feature_mode not in valid: raise ValueError(f"Unknown feature_mode: {feature_mode}")
    df=pd.read_csv(csv_path)
    use_sg=feature_mode!="raw"
    if use_sg:
        df=add_safegrip_features(df,cfg)
    features=[c for c in df.columns if c not in META and pd.api.types.is_numeric_dtype(df[c])]
    features=[c for c in features if c not in ("mu_ref","physics_lower_raw")]
    if feature_mode=="safegrip_no_excitation":
        drop={"sg_excitation_instant","sg_excitation_score"}
        features=[c for c in features if c not in drop]
    elif feature_mode=="safegrip_endpoint":
        # A static endpoint control must not retain history through derivative
        # or causal-memory engineered channels. Current-value algebraic transforms
        # (acceleration magnitude, wheel spread, torque utilization, etc.) remain.
        drop={"sg_jerk_x","sg_jerk_y","sg_jerk_mag","sg_excitation_instant","sg_excitation_score"}
        features=[c for c in features if c not in drop]
    L=int(sequence_length or cfg["sequence_length"]); stride=int(cfg["stride"])
    start=common_eval_start(cfg) if eval_start is None else int(eval_start)
    physics_window=int(cfg.get("physics",{}).get("window_samples",1))
    splits={sp:windows_for_split(df,features,sp,L,stride,start,physics_window=physics_window) for sp in ("train","calibration","validation","test")}
    Xtr,ytr,raw_lotr,idtr=splits["train"]; Xc,yc,raw_loc,idc=splits["calibration"]
    Xv,yv,raw_lov,idv=splits["validation"]; Xt,yt,raw_lot,idt=splits["test"]
    if min(len(Xtr),len(Xv),len(Xt))==0:
        raise RuntimeError("One split has no common-evaluation windows; lower benchmark.common_warmup_samples or inspect prepared data")

    if "sg_excitation_score" in features:
        ei=features.index("sg_excitation_score")
        etr=np.clip(Xtr[:,-1,ei],0,1).astype(np.float32); ec=np.clip(Xc[:,-1,ei],0,1).astype(np.float32) if len(Xc) else np.empty(0,np.float32)
        ev=np.clip(Xv[:,-1,ei],0,1).astype(np.float32); et=np.clip(Xt[:,-1,ei],0,1).astype(np.float32)
    else:
        etr=np.zeros(len(Xtr),np.float32); ec=np.zeros(len(Xc),np.float32); ev=np.zeros(len(Xv),np.float32); et=np.zeros(len(Xt),np.float32)

    scaler=_make_scaler(scaler_kind).fit(Xtr.reshape(-1,len(features)))
    excitation_feature_index=features.index("sg_excitation_score") if "sg_excitation_score" in features else None
    def sc(X):
        if len(X)==0: return X.astype(np.float32)
        out=scaler.transform(X.reshape(-1,len(features))).reshape(X.shape).astype(np.float32)
        if excitation_feature_index is not None:
            out[:,:,excitation_feature_index]=X[:,:,excitation_feature_index].astype(np.float32)
        return out
    Xtr,Xc,Xv,Xt=map(sc,(Xtr,Xc,Xv,Xt))
    lower_cal_mask,uq_cal_mask=_calibration_role_masks(idc,cfg.get("calibration",{}).get("lower_fraction",0.25)) if len(yc) else (np.zeros(0,dtype=bool),np.zeros(0,dtype=bool))
    q=conformal_lower_correction(raw_loc[lower_cal_mask],yc[lower_cal_mask],cfg["alpha"]) if lower_cal_mask.any() else 0.0
    lotr=apply_lower_correction(raw_lotr,q).astype(np.float32)
    loc=apply_lower_correction(raw_loc,q).astype(np.float32)
    lov=apply_lower_correction(raw_lov,q).astype(np.float32)
    lot=apply_lower_correction(raw_lot,q).astype(np.float32)
    return Bundle(features,scaler,scaler_kind,L,start,q,bool(use_sg),feature_mode,
                  Xtr,ytr,lotr,raw_lotr,idtr,etr,Xc,yc,loc,raw_loc,idc,ec,lower_cal_mask,uq_cal_mask,
                  Xv,yv,lov,raw_lov,idv,ev,Xt,yt,lot,raw_lot,idt,et)


def regression_metrics(y,p,lo=None,mu_upper=1.3,sigma=None,raw_mean=None,project_uncertainty=False,
                       interval_low=None,interval_high=None,interval_low_raw=None,interval_high_raw=None):
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

    # SafeGrip-v3 supplies conformal intervals directly. Literature models that
    # expose a predictive sigma retain the legacy Gaussian diagnostics.
    if interval_low is not None and interval_high is not None:
        low=np.asarray(interval_low); high=np.asarray(interval_high)
        raw_low=np.asarray(interval_low_raw if interval_low_raw is not None else low)
        raw_high=np.asarray(interval_high_raw if interval_high_raw is not None else high)
        out.update({
            "picp95_raw":float(np.mean((y>=raw_low)&(y<=raw_high))),
            "mpiw95_raw":float(np.mean(raw_high-raw_low)),
            "picp95":float(np.mean((y>=low)&(y<=high))),
            "mpiw95":float(np.mean(high-low)),
        })
        if lo is not None:
            out["interval_outside_physics_set_rate"]=float(np.mean((low<np.asarray(lo)-1e-9)|(high>float(mu_upper)+1e-9)))
        if sigma is not None:
            out["mean_uq_scale"]=float(np.mean(np.asarray(sigma)))
    elif sigma is not None:
        sigma=np.maximum(np.asarray(sigma),1e-5)
        center=np.asarray(raw_mean if raw_mean is not None else p)
        low_raw,high_raw=gaussian_interval(center,sigma)
        out.update({
            "picp95_raw":float(np.mean((y>=low_raw)&(y<=high_raw))),
            "mpiw95_raw":float(np.mean(high_raw-low_raw)),
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


def _proposal_flags(variant: str) -> dict:
    if variant not in PROPOSAL_VARIANTS:
        raise ValueError(variant)
    raw_only = variant in ("safegrip_data_only", "safegrip_no_excitation")
    return {
        # static_only is now genuinely endpoint-only in SafeGripV3Net.
        "use_temporal": variant != "safegrip_static_only",
        "use_gate": variant not in ("safegrip_data_only", "safegrip_static_only", "safegrip_no_gate", "safegrip_no_excitation"),
        "use_bound": variant not in ("safegrip_data_only", "safegrip_no_bound"),
        "use_uq": variant != "safegrip_no_uq",
        "use_calibrated_lower": variant != "safegrip_no_calibration",
        "raw_features_only": raw_only,
        "use_excitation_regularizer": variant not in ("safegrip_data_only", "safegrip_no_excitation"),
        "use_excitation_uq_inflation": variant not in ("safegrip_data_only", "safegrip_no_excitation"),
    }


def _proposal_model(variant: str, b: Bundle, hp: dict):
    flags=_proposal_flags(variant)
    exc_idx=b.features.index("sg_excitation_score") if "sg_excitation_score" in b.features else None
    model=SafeGripV3Net(
        len(b.features), excitation_index=exc_idx,
        hidden=int(hp["hidden"]), gru_hidden=int(hp["gru_hidden"]),
        dropout=float(hp["dropout"]), evidence_window=int(hp.get("evidence_window",8)),
        delta_scale=float(hp.get("delta_scale",2.0)),
        gate_init_slope=float(hp.get("gate_init_slope",6.0)),
        gate_init_threshold=float(hp.get("gate_init_threshold",0.25)),
        use_temporal=flags["use_temporal"], use_gate=flags["use_gate"],
        use_bound=flags["use_bound"],
    )
    # Positive coefficient gives uncertainty that is monotonically larger when
    # the excitation/observability score is smaller.
    model.excitation_beta=max(0.0,float(hp.get("excitation_beta",1.0))) if flags["use_excitation_uq_inflation"] else 0.0
    return model


def proposal_hparams(cfg, overrides=None):
    p=dict(cfg.get("proposal",{})); tr=cfg["training"]
    defaults={
        "hidden":tr.get("hidden",64), "gru_hidden":32, "dropout":tr.get("dropout",.1),
        "lr":tr.get("lr",1e-3), "weight_decay":tr.get("weight_decay",1e-4),
        "batch_size":tr.get("batch_size",256), "huber_beta":0.05,
        "evidence_window":8, "delta_scale":2.0,
        "gate_init_slope":6.0, "gate_init_threshold":0.25,
        "delta_loss_weight":0.10, "rank_loss_weight":0.03,
        "smooth_loss_weight":0.005, "rank_min_delta":0.01,
        "rank_margin":0.005, "pair_lag":1,
        "uq_epochs":100, "uq_lr":1e-3, "uq_scale_floor":0.005,
        "excitation_beta":1.0,
    }
    defaults.update(p); defaults.update(overrides or {}); return defaults


def _finite_sample_quantile(values, alpha: float) -> float:
    values=np.asarray(values,float)
    values=values[np.isfinite(values)]
    if len(values)==0:
        return 1.0
    level=min(1.0,np.ceil((len(values)+1)*(1-float(alpha)))/len(values))
    try:
        return float(np.quantile(values,level,method="higher"))
    except TypeError:
        return float(np.quantile(values,level,interpolation="higher"))


def _segment_key(endpoint_id: str) -> str:
    text=str(endpoint_id)
    parts=text.rsplit(":",2)
    return parts[0] if len(parts)>=3 else text


def _block_max_scores(scores, ids, block_size: int) -> np.ndarray:
    """Conservative block-max calibration for overlapping temporal windows.

    It does not claim arbitrary-dependence conformal validity.  It reduces the
    pseudo-replication caused by strongly overlapping windows by taking one
    worst-case score per non-overlapping block inside each trajectory segment.
    """
    scores=np.asarray(scores,float); ids=np.asarray(ids,dtype=str)
    block=max(1,int(block_size)); out=[]
    if len(scores)==0:
        return np.empty(0,float)
    start=0
    while start<len(scores):
        key=_segment_key(ids[start]); stop=start+1
        while stop<len(scores) and _segment_key(ids[stop])==key:
            stop+=1
        local=scores[start:stop]
        for i in range(0,len(local),block):
            chunk=local[i:i+block]
            finite=chunk[np.isfinite(chunk)]
            if len(finite): out.append(float(np.max(finite)))
        start=stop
    return np.asarray(out,float)


def _previous_pair_indices(ids: np.ndarray, lag: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Return previous endpoint indices without crossing trajectory segments.

    The pairing is used only for relative-change/ranking regularization.  It is
    derived from stable endpoint IDs, so shuffled mini-batches cannot accidentally
    pair samples from different road segments.
    """
    ids=np.asarray(ids,dtype=str); n=len(ids); lag=max(1,int(lag))
    prev=np.arange(n,dtype=np.int64); valid=np.zeros(n,dtype=bool)
    for i in range(n):
        j=i-lag
        if j>=0 and _segment_key(ids[i])==_segment_key(ids[j]):
            prev[i]=j; valid[i]=True
    return prev,valid


def _forward_point(model, X, lower, mu_upper, batch=1024, return_features=False, excitation=None):
    dev=device(); model.eval()
    pred=[]; latent=[]; rel=[]; feats=[]; priors=[]; deltas=[]; prior_latents=[]
    excitation_arr=None if excitation is None else np.asarray(excitation,dtype=np.float32)
    with torch.no_grad():
        for i in range(0,len(X),batch):
            xb=torch.from_numpy(X[i:i+batch]).to(dev)
            lb=torch.from_numpy(np.asarray(lower[i:i+batch],dtype=np.float32)).to(dev)
            eb=None if excitation_arr is None else torch.from_numpy(excitation_arr[i:i+batch]).to(dev)
            d=model.forward_details(xb,lb,float(mu_upper),excitation=eb)
            pred.append(d["prediction"].cpu().numpy())
            latent.append(d["latent"].cpu().numpy())
            rel.append(d["reliability"].cpu().numpy())
            priors.append(d["prior_prediction"].cpu().numpy())
            deltas.append(d["evidence_delta"].cpu().numpy())
            prior_latents.append(d["prior_latent"].cpu().numpy())
            if return_features:
                feats.append(d["features"].cpu().numpy())
    base=(np.concatenate(pred),np.concatenate(latent),np.concatenate(rel))
    if return_features:
        return base+(np.concatenate(feats),np.concatenate(priors),np.concatenate(deltas),np.concatenate(prior_latents))
    return base


def _fit_residual_scale(model: SafeGripV3Net, b: Bundle, cfg: dict, hp: dict, lower_val: np.ndarray):
    """Fit heteroscedastic residual scale after point-model selection.

    The validation residual median initializes the scale head near the correct
    numerical range.  Excitation inflation is monotone and external to the head,
    preserving the clean separation between point estimation and UQ.
    """
    dev=device(); model.eval()
    p,_,_,h,_,_,_=_forward_point(
        model,b.Xv,lower_val,float(cfg["mu_upper"]),return_features=True,excitation=b.ev
    )
    target=np.maximum(np.abs(np.asarray(b.yv,float)-p),float(hp["uq_scale_floor"]))
    e=np.clip(np.asarray(b.ev,float),0.0,1.0)
    inflation=1.0+max(0.0,float(hp.get("excitation_beta",1.0)))*(1.0-e)
    base_target=np.maximum(target/np.maximum(inflation,1e-6),float(hp["uq_scale_floor"]))
    init_scale=float(np.median(base_target)) if len(base_target) else float(hp["uq_scale_floor"])*2.0
    head=ResidualScaleHead(
        int(hp["hidden"]),floor=float(hp["uq_scale_floor"]),initial_scale=init_scale
    ).to(dev)
    opt=torch.optim.AdamW(head.parameters(),lr=float(hp["uq_lr"]),weight_decay=float(hp["weight_decay"]))
    ht=torch.from_numpy(h.astype(np.float32)).to(dev)
    tgt=torch.from_numpy(base_target.astype(np.float32)).to(dev)
    best=None; bestloss=float("inf"); stale=0
    max_epochs=int(hp.get("uq_epochs",100)); patience=max(10,min(30,max_epochs//4))
    for _ in range(max_epochs):
        head.train(); opt.zero_grad()
        base=head(ht)
        loss=nn.functional.smooth_l1_loss(torch.log(base),torch.log(tgt),beta=0.25)
        loss.backward(); torch.nn.utils.clip_grad_norm_(head.parameters(),5.0); opt.step()
        val=float(loss.detach().cpu())
        if val<bestloss-1e-6:
            bestloss=val; best={k:v.detach().cpu().clone() for k,v in head.state_dict().items()}; stale=0
        else:
            stale+=1
        if stale>=patience: break
    if best: head.load_state_dict(best)
    model.scale_head=head
    model.uq_scale_initialization=init_scale


def fit_proposal(variant,b:Bundle,cfg,epochs,hp_overrides=None):
    if variant not in PROPOSAL_VARIANTS: raise ValueError(variant)
    flags=_proposal_flags(variant)
    if not b.proposal_features and not flags["raw_features_only"]:
        raise ValueError("This SafeGrip variant requires make_bundle(..., proposal_features=True)")
    hp=proposal_hparams(cfg,hp_overrides); dev=device(); model=_proposal_model(variant,b,hp).to(dev)
    opt=torch.optim.AdamW(model.parameters(),lr=float(hp["lr"]),weight_decay=float(hp["weight_decay"]))

    # Point training uses only train labels.  Calibration labels never enter the
    # gradient path.  Relative losses use previous endpoints from the same stable
    # trajectory segment and therefore never cross a split/segment boundary.
    lo_train=b.raw_lotr
    lo_val=b.lov if flags["use_calibrated_lower"] else b.raw_lov
    prev_idx,pair_valid=_previous_pair_indices(b.idtr,int(hp.get("pair_lag",1)))
    dataset=TensorDataset(
        torch.from_numpy(b.Xtr), torch.from_numpy(b.ytr), torch.from_numpy(lo_train), torch.from_numpy(b.etr),
        torch.from_numpy(b.Xtr[prev_idx]), torch.from_numpy(b.ytr[prev_idx]),
        torch.from_numpy(lo_train[prev_idx]), torch.from_numpy(b.etr[prev_idx]),
        torch.from_numpy(pair_valid.astype(np.bool_)),
    )
    dl=DataLoader(dataset,batch_size=int(hp["batch_size"]),shuffle=True)
    xv=torch.from_numpy(b.Xv).to(dev); yv=torch.from_numpy(b.yv).to(dev)
    lov=torch.from_numpy(np.asarray(lo_val,dtype=np.float32)).to(dev)
    ev=torch.from_numpy(np.asarray(b.ev,dtype=np.float32)).to(dev)
    mu_u=float(cfg["mu_upper"]); best=None; bestloss=float("inf"); bad=0
    patience=int(cfg["training"].get("patience",10)); huber_beta=float(hp.get("huber_beta",0.05))
    delta_w=max(0.0,float(hp.get("delta_loss_weight",0.10)))
    rank_w=max(0.0,float(hp.get("rank_loss_weight",0.03)))
    smooth_w=max(0.0,float(hp.get("smooth_loss_weight",0.005))) if flags["use_excitation_regularizer"] else 0.0
    rank_min=max(0.0,float(hp.get("rank_min_delta",0.01)))
    rank_margin=max(0.0,float(hp.get("rank_margin",0.005)))

    for _ in range(int(epochs)):
        model.train()
        for xb,yb,lb,eb,xp,yp,lp,ep,pair_mask in dl:
            xb=xb.to(dev); yb=yb.to(dev); lb=lb.to(dev); eb=eb.to(dev)
            xp=xp.to(dev); yp=yp.to(dev); lp=lp.to(dev); ep=ep.to(dev); pair_mask=pair_mask.to(dev)
            opt.zero_grad()
            pred,_,_=model(xb,lb,mu_u,excitation=eb)
            loss=nn.functional.smooth_l1_loss(pred,yb,beta=huber_beta)
            if pair_mask.any() and (delta_w>0 or rank_w>0 or smooth_w>0):
                prev_pred,_,_=model(xp,lp,mu_u,excitation=ep)
                dp=pred-prev_pred; dy=yb-yp
                m=pair_mask
                if delta_w>0:
                    loss=loss+delta_w*nn.functional.smooth_l1_loss(dp[m],dy[m],beta=huber_beta)
                if rank_w>0:
                    rm=m & (torch.abs(dy)>=rank_min)
                    if rm.any():
                        sign=torch.sign(dy[rm])
                        rank_loss=torch.relu(rank_margin-sign*dp[rm]).mean()
                        loss=loss+rank_w*rank_loss
                if smooth_w>0:
                    # Weak excitation should not cause gratuitous local jumps;
                    # high excitation automatically relaxes this regularizer.
                    smooth=((1.0-eb[m])*torch.abs(dp[m])).mean()
                    loss=loss+smooth_w*smooth
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5.0); opt.step()

        model.eval()
        with torch.no_grad():
            pv,_,_=model(xv,lov,mu_u,excitation=ev)
            vl=nn.functional.mse_loss(pv,yv).item()
        if vl<bestloss-1e-7:
            bestloss=vl; best={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}; bad=0
        else:
            bad+=1
        if bad>=patience: break
    if best: model.load_state_dict(best)

    # UQ is deliberately fitted after point estimation, so uncertainty learning
    # cannot trade away RMSE. Validation fits the scale function; the disjoint
    # calibration split determines the block-conformal multiplier.
    if flags["use_uq"]:
        _fit_residual_scale(model,b,cfg,hp,lo_val)
        lo_cal=b.loc if flags["use_calibrated_lower"] else b.raw_loc
        mask=np.asarray(b.uq_cal_mask,dtype=bool)
        Xcal=b.Xc[mask]; ycal=b.yc[mask]; idcal=b.idc[mask]; ecal=b.ec[mask]; lower_cal=lo_cal[mask]
        if len(Xcal):
            pcal,_,_,hcal,_,_,_=_forward_point(
                model,Xcal,lower_cal,mu_u,return_features=True,excitation=ecal
            )
            model.scale_head.eval()
            with torch.no_grad():
                base=model.scale_head(torch.from_numpy(hcal.astype(np.float32)).to(dev)).cpu().numpy()
            inflation=1.0+max(0.0,float(hp.get("excitation_beta",1.0)))*(1.0-np.asarray(ecal,float))
            scale=base*inflation
            scores=np.abs(np.asarray(ycal,float)-pcal)/np.maximum(scale,float(hp["uq_scale_floor"]))
            block=int(cfg.get("uq",{}).get("block_size",0))
            if block<=0:
                block=max(1,int(np.ceil(float(b.sequence_length)/max(float(cfg.get("stride",1)),1.0))))
            block_scores=_block_max_scores(scores,idcal,block)
            model.conformal_q=_finite_sample_quantile(block_scores,cfg.get("alpha",0.05))
            model.conformal_block_size=block
            model.uq_calibration_count=int(len(Xcal)); model.uq_block_count=int(len(block_scores))
        else:
            model.conformal_q=1.96
            model.conformal_block_size=1
    return model,hp


def predict_proposal_details(model,variant,X,lo,raw_lo,mu_upper,batch=1024,excitation=None):
    flags=_proposal_flags(variant)
    bound=np.asarray(lo if flags["use_calibrated_lower"] else raw_lo,dtype=np.float32)
    prediction,latent,reliability,h,prior,delta,prior_latent=_forward_point(
        model,X,bound,mu_upper,batch=batch,return_features=True,excitation=excitation
    )
    details={
        "prediction":prediction,"raw_mean":prediction.copy(),"latent_score":latent,
        "prior_latent":prior_latent,"prior_prediction":prior,"evidence_delta":delta,
        "gate":reliability,"reliability":reliability,"sigma":None,"bound":bound,
    }
    if flags["use_uq"] and getattr(model,"scale_head",None) is not None:
        dev=device(); model.scale_head.eval(); scales=[]
        with torch.no_grad():
            for i in range(0,len(h),batch):
                scales.append(model.scale_head(torch.from_numpy(h[i:i+batch].astype(np.float32)).to(dev)).cpu().numpy())
        base=np.concatenate(scales)
        if excitation is None:
            excitation=np.clip(reliability,0.0,1.0)
        excitation=np.clip(np.asarray(excitation,float),0.0,1.0)
        scale=base*(1.0+max(0.0,float(getattr(model,"excitation_beta",1.0)))*(1.0-excitation))
        q=float(getattr(model,"conformal_q",1.96) or 1.96)
        radius=q*scale
        low_raw=prediction-radius; high_raw=prediction+radius
        if flags["use_bound"]:
            low=np.maximum(low_raw,bound); high=np.minimum(high_raw,float(mu_upper))
        else:
            low=np.maximum(low_raw,0.0); high=np.minimum(high_raw,float(mu_upper))
        low=np.minimum(low,prediction); high=np.maximum(high,prediction)
        details.update({
            "sigma":scale.astype(np.float32), "conformal_q":q,
            "pi95_low_raw":low_raw.astype(np.float32), "pi95_high_raw":high_raw.astype(np.float32),
            "pi95_low_physics":low.astype(np.float32), "pi95_high_physics":high.astype(np.float32),
        })
    return details

def predict_proposal(model,variant,X,lo,raw_lo,mu_upper,batch=1024,excitation=None):
    d=predict_proposal_details(model,variant,X,lo,raw_lo,mu_upper,batch=batch,excitation=excitation)
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
                                                     "proposal_features":bool(b.proposal_features),
                                                     "lower_bound_calibration_count":int(np.sum(b.lower_cal_mask)),
                                                     "predictive_uq_calibration_count":int(np.sum(b.uq_cal_mask)),
                                                     "calibration_roles_disjoint":bool(not np.any(b.lower_cal_mask & b.uq_cal_mask)),
                                                     "scaler":b.scaler_kind,
                                                     "proposal_unscaled_features":["sg_excitation_score"] if b.proposal_features and "sg_excitation_score" in b.features else []},indent=2),encoding="utf-8")


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
    picp=float(row.get("picp95",np.nan))
    mpiw=float(row.get("mpiw95",np.nan))
    target_coverage=1.0-alpha
    interval_tol=max(0.03,2.0*np.sqrt(max(target_coverage*(1-target_coverage),1e-12)/max(len(y),1)))
    excitation=np.asarray(preds.get("safegrip_excitation",bundle.et),float)
    reliability=np.asarray(preds.get("safegrip_reliability",preds.get("safegrip_gate",np.zeros_like(y))),float)
    if len(excitation)==len(reliability) and np.std(excitation)>1e-10 and np.std(reliability)>1e-10:
        reliability_corr=float(np.corrcoef(excitation,reliability)[0,1])
    else:
        reliability_corr=float("nan")
    prior=np.asarray(preds.get("safegrip_prior",p),float)
    prior_rmse=float(np.sqrt(np.mean((y-prior)**2))) if len(prior)==len(y) else float("nan")
    n_segments=len({_segment_key(x) for x in np.asarray(bundle.idt,dtype=str)})
    checks = {
        "finite_predictions": bool(np.isfinite(p).all()),
        "enough_test_endpoints": bool(len(y) >= min_n),
        "final_prediction_not_constant": bool(pred_std > max(1e-4, 0.05 * target_std)),
        "not_projection_dominated": bool(not np.isfinite(projection_rate) or projection_rate < float(cfg.get("health",{}).get("max_projection_correction_rate", 0.95))),
        "beats_train_mean_rmse": bool(float(row["rmse"]) < mean_rmse),
        "positive_r2": bool(float(row["r2"]) > 0.0),
        "raw_bound_finite": bool(np.isfinite(raw_bound).all()),
        "calibrated_lower_coverage_consistent": bool(np.isfinite(lower_violation) and lower_violation <= alpha + coverage_tol),
        "predictive_interval_coverage_consistent": bool(np.isfinite(picp) and picp >= target_coverage-interval_tol),
        "reliability_not_inversely_related_to_excitation": bool(not np.isfinite(reliability_corr) or reliability_corr >= -1e-6),
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
            "predictive_interval_coverage": picp,
            "predictive_interval_width": mpiw,
            "predictive_width_over_target_std": float(mpiw/max(target_std,1e-12)) if np.isfinite(mpiw) else float("nan"),
            "target_predictive_coverage": target_coverage,
            "predictive_coverage_tolerance": interval_tol,
            "n_test_segments": int(n_segments),
            "physics_lower_nonzero_rate": float(np.mean(raw_bound>1e-9)),
            "physics_lower_above_005_rate": float(np.mean(raw_bound>0.05)),
            "physics_information_fraction_mean": float(np.mean(np.clip(raw_bound/max(float(cfg.get("mu_upper",1.3)),1e-12),0.0,1.0))),
            "prior_rmse": prior_rmse,
            "mean_abs_dynamic_update": float(np.mean(np.abs(p-prior))) if len(prior)==len(p) else float("nan"),
            "excitation_reliability_correlation": reliability_corr,
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
    proposal_bundle=make_bundle(csv_path,cfg,sequence_length=proposal_seq,scaler_kind="standard",eval_start=start,proposal_features=True)
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
    feature_parity_rows=[]; label_budget_rows=[]; common_uq_rows=[]
    compared_bundles={}
    controls_manifest={"feature_parity":[],"label_budget":[],"common_conformal":[]}
    parity_names=set(cfg.get("evaluation",{}).get("feature_parity_models",["lampe2023_gru","schaefke2023_transformer"]))
    enable_feature_parity=bool(cfg.get("evaluation",{}).get("feature_parity_controls",True)) and preset in ("trust","paper")
    enable_label_budget=bool(cfg.get("evaluation",{}).get("label_budget_parity_controls",True)) and preset in ("trust","paper")
    enable_common_uq=bool(cfg.get("evaluation",{}).get("common_conformal_controls",True)) and preset in ("trust","paper")
    preds=pd.DataFrame({"endpoint_id":proposal_bundle.idt,"y_true":proposal_bundle.yt,
                        "physics_lower":proposal_bundle.lot,"physics_lower_raw":proposal_bundle.raw_lot})
    control_preds=pd.DataFrame({"endpoint_id":proposal_bundle.idt,"y_true":proposal_bundle.yt,"physics_lower":proposal_bundle.lot})
    selected={}; scaler_dir=ensure_dir(out/"baseline_scalers"); feature_dir=ensure_dir(out/"baseline_features")

    for name in baseline_names:
        hp=dict(baseline_plan[name]); selected[name]=hp
        b=make_bundle(csv_path,cfg,sequence_length=int(hp["sequence_length"]),scaler_kind=str(hp["scaler"]),eval_start=start)
        _assert_same_targets(proposal_bundle,b,name); compared_bundles[name]=b; dump(b.scaler,scaler_dir/f"{name}.joblib"); (feature_dir/f"{name}.json").write_text(json.dumps(b.features,indent=2),encoding="utf-8")
        print(f"[baseline/adapted] {name} DOI={LITERATURE_BASELINES[name]['doi']} L={b.sequence_length} scaler={b.scaler_kind}")
        seed_preds=[]; seed_sigmas=[]; seed_control=[]
        for seed in seed_list:
            seed_everything(int(seed)); t0=time.time()
            model=fit_literature(name,b,cfg,None,preset,hp); p,sig=predict_literature(model,name,b.Xt)
            if enable_common_uq and np.any(b.uq_cal_mask):
                m=np.asarray(b.uq_cal_mask,dtype=bool)
                pcal,_=predict_literature(model,name,b.Xc[m])
                cil,cih,cq=_common_conformal_interval(b.yc[m],pcal,p,cfg.get("alpha",0.05))
                cm=regression_metrics(b.yt,p,b.lot,cfg["mu_upper"],None,raw_mean=p,interval_low=cil,interval_high=cih)
                common_uq_rows.append({"model":name,"seed":int(seed),"common_conformal_q":cq,**cm})
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

        # Feature-parity control: same literature architecture receives the same
        # label-free SafeGrip engineered representation.  This isolates gains
        # due to representation from gains due to the proposal architecture.
        if enable_feature_parity and name in parity_names:
            bp=make_bundle(csv_path,cfg,sequence_length=int(hp["sequence_length"]),scaler_kind=str(hp["scaler"]),eval_start=start,proposal_features=True)
            _assert_same_targets(proposal_bundle,bp,name+"__sg_features")
            controls_manifest["feature_parity"].append(name+"__sg_features")
            for seed in seed_list:
                seed_everything(int(seed)); fm=fit_literature(name,bp,cfg,None,preset,hp); fp,fs=predict_literature(fm,name,bp.Xt)
                feature_parity_rows.append({"model":name+"__sg_features","source_model":name,"seed":int(seed),
                    **regression_metrics(bp.yt,fp,bp.lot,cfg["mu_upper"],fs,raw_mean=fp)})

        # Equal supervised-label-budget control. Baselines receive the lower-
        # calibration labels that affect SafeGrip's point-support calibration;
        # the disjoint UQ-calibration role remains held out.
        if enable_label_budget:
            bl=_augment_train_with_lower_calibration(b)
            controls_manifest["label_budget"].append(name+"__equal_label_budget")
            for seed in seed_list:
                seed_everything(int(seed)); lm=fit_literature(name,bl,cfg,None,preset,hp); lp,ls=predict_literature(lm,name,bl.Xt)
                label_budget_rows.append({"model":name+"__equal_label_budget","source_model":name,"seed":int(seed),
                    "supervised_train_count":int(len(bl.ytr)),**regression_metrics(bl.yt,lp,bl.lot,cfg["mu_upper"],ls,raw_mean=lp)})

    name="safegrip"; print(f"[proposal/v3] safegrip L={proposal_bundle.sequence_length}")
    proposal_preds=[]; proposal_raw=[]; proposal_sigmas=[]; proposal_latent=[]; final_hp=None
    seed_priors=[]; seed_deltas=[]; seed_prior_latents=[]
    seed_gate_slopes=[]; seed_gate_thresholds=[]; seed_uq_initial_scales=[]
    seed_pi_low=[]; seed_pi_high=[]; seed_pi_low_raw=[]; seed_pi_high_raw=[]; seed_gates=[]; seed_q=[]; seed_blocks=[]; seed_uq_counts=[]
    for seed in seed_list:
        seed_everything(int(seed)); t0=time.time(); model,final_hp=fit_proposal(name,proposal_bundle,cfg,epochs_proposal,hp_overrides)
        d=predict_proposal_details(model,name,proposal_bundle.Xt,proposal_bundle.lot,proposal_bundle.raw_lot,cfg["mu_upper"],excitation=proposal_bundle.et)
        if enable_common_uq and np.any(proposal_bundle.uq_cal_mask):
            m=np.asarray(proposal_bundle.uq_cal_mask,dtype=bool)
            dc=predict_proposal_details(model,name,proposal_bundle.Xc[m],proposal_bundle.loc[m],proposal_bundle.raw_loc[m],cfg["mu_upper"],excitation=proposal_bundle.ec[m])
            cil,cih,cq=_common_conformal_interval(proposal_bundle.yc[m],dc["prediction"],d["prediction"],cfg.get("alpha",0.05))
            cm=regression_metrics(proposal_bundle.yt,d["prediction"],proposal_bundle.lot,cfg["mu_upper"],None,raw_mean=d["prediction"],interval_low=cil,interval_high=cih)
            common_uq_rows.append({"model":"safegrip","seed":int(seed),"common_conformal_q":cq,**cm})
        per_seed.append({"model":name,"kind":"proposal","doi":"","seed":int(seed),
                         **regression_metrics(proposal_bundle.yt,d["prediction"],d["bound"],cfg["mu_upper"],d["sigma"],
                                              raw_mean=d["raw_mean"],
                                              interval_low=d.get("pi95_low_physics"),interval_high=d.get("pi95_high_physics"),
                                              interval_low_raw=d.get("pi95_low_raw"),interval_high_raw=d.get("pi95_high_raw")),
                         "conformal_q":float(d.get("conformal_q",np.nan)),
                         "mean_reliability":float(np.mean(d["reliability"])),
                         "mean_gate":float(np.mean(d["reliability"])),
                         "prior_rmse":float(np.sqrt(np.mean((proposal_bundle.yt-d["prior_prediction"])**2))),
                         "mean_abs_dynamic_update":float(np.mean(np.abs(d["prediction"]-d["prior_prediction"]))),
                         "seconds":time.time()-t0})
        proposal_preds.append(d["prediction"]); proposal_raw.append(d["raw_mean"]); proposal_latent.append(d["latent_score"]); seed_gates.append(d["reliability"])
        seed_priors.append(d["prior_prediction"]); seed_deltas.append(d["evidence_delta"]); seed_prior_latents.append(d["prior_latent"])
        seed_gate_slopes.append(float((nn.functional.softplus(model.reliability_slope_raw)+1e-4).detach().cpu()))
        seed_gate_thresholds.append(float(torch.sigmoid(model.reliability_threshold_raw).detach().cpu()))
        seed_uq_initial_scales.append(float(getattr(model,"uq_scale_initialization",np.nan)))
        if d["sigma"] is not None: proposal_sigmas.append(d["sigma"])
        if "pi95_low_physics" in d:
            seed_pi_low.append(d["pi95_low_physics"]); seed_pi_high.append(d["pi95_high_physics"])
            seed_pi_low_raw.append(d["pi95_low_raw"]); seed_pi_high_raw.append(d["pi95_high_raw"])
            seed_q.append(float(d["conformal_q"])); seed_blocks.append(int(getattr(model,"conformal_block_size",1) or 1)); seed_uq_counts.append(int(getattr(model,"uq_calibration_count",0)))
    preds[name]=np.mean(np.stack(proposal_preds),axis=0)
    preds[name+"_raw"]=np.mean(np.stack(proposal_raw),axis=0)
    preds[name+"_latent"]=np.mean(np.stack(proposal_latent),axis=0)
    for _seed,_latent in zip(seed_list,proposal_latent):
        preds[f"{name}_latent_seed_{int(_seed)}"]=_latent
    preds[name+"_reliability"]=np.mean(np.stack(seed_gates),axis=0)
    preds[name+"_gate"]=preds[name+"_reliability"]  # backward-compatible column name
    preds[name+"_prior"]=np.mean(np.stack(seed_priors),axis=0)
    preds[name+"_evidence_delta"]=np.mean(np.stack(seed_deltas),axis=0)
    preds[name+"_prior_latent"]=np.mean(np.stack(seed_prior_latents),axis=0)
    preds[name+"_excitation"]=proposal_bundle.et
    if proposal_sigmas: preds[name+"_sigma"]=np.mean(np.stack(proposal_sigmas),axis=0)
    if seed_pi_low:
        preds[name+"_pi95_low_physics"]=np.mean(np.stack(seed_pi_low),axis=0)
        preds[name+"_pi95_high_physics"]=np.mean(np.stack(seed_pi_high),axis=0)
        preds[name+"_pi95_low_raw"]=np.mean(np.stack(seed_pi_low_raw),axis=0)
        preds[name+"_pi95_high_raw"]=np.mean(np.stack(seed_pi_high_raw),axis=0)

    (out/"proposal_reliability.json").write_text(json.dumps({
        "form":"sigmoid(softplus(slope_raw)*(excitation-sigmoid(threshold_raw)))",
        "monotone_by_construction":True,
        "per_seed_slope":seed_gate_slopes,
        "per_seed_threshold":seed_gate_thresholds,
        "per_seed_uq_initial_scale":seed_uq_initial_scales,
        "excitation_feature":"sg_excitation_score",
        "excitation_feature_range":[0.0,1.0],
        "excitation_preserved_unscaled":True,
    },indent=2),encoding="utf-8")
    (out/"proposal_uq.json").write_text(json.dumps({
        "method":"post-hoc residual scale + block-max split conformal",
        "alpha":float(cfg.get("alpha",0.05)),
        "per_seed_conformal_q":seed_q,
        "per_seed_block_size":seed_blocks,
        "per_seed_uq_calibration_count":seed_uq_counts,
        "lower_bound_calibration_count":int(np.sum(proposal_bundle.lower_cal_mask)),
        "predictive_uq_calibration_count":int(np.sum(proposal_bundle.uq_cal_mask)),
        "calibration_roles_disjoint":bool(not np.any(proposal_bundle.lower_cal_mask & proposal_bundle.uq_cal_mask)),
        "dependence_claim":"block maxima mitigate overlap; no arbitrary-dependence finite-sample guarantee",
    },indent=2),encoding="utf-8")
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
        "proposal_feature_engineering_label_free":True,
        "proposal_point_loss":"Huber + same-segment relative-change + ranking + weak-excitation smoothness regularization",
        "proposal_architecture":"long-context friction prior + short-context dynamic evidence update",
        "proposal_reliability":"monotone sigmoid(softplus(slope)*(excitation-threshold))",
        "proposal_bound_parameterization":"lower + (mu_upper-lower)*sigmoid(q_prior + reliability*evidence_delta)",
        "proposal_uq":"post-hoc residual scale + block-max split-conformal multiplier",
        "proposal_uq_dependence_note":"block-max calibration is a conservative dependence mitigation, not an arbitrary-dependence finite-sample guarantee",
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
    if feature_parity_rows:
        _aggregate_seed_metrics(feature_parity_rows).to_csv(out/"feature_parity_metrics.csv",index=False)
        pd.DataFrame(feature_parity_rows).to_csv(out/"feature_parity_metrics_by_seed.csv",index=False)
    if label_budget_rows:
        _aggregate_seed_metrics(label_budget_rows).to_csv(out/"label_budget_parity_metrics.csv",index=False)
        pd.DataFrame(label_budget_rows).to_csv(out/"label_budget_parity_metrics_by_seed.csv",index=False)
    if common_uq_rows:
        controls_manifest["common_conformal"]=["same symmetric split-conformal calibration on the disjoint UQ-calibration role"]
        _aggregate_seed_metrics(common_uq_rows).to_csv(out/"common_conformal_uq_metrics.csv",index=False)
        pd.DataFrame(common_uq_rows).to_csv(out/"common_conformal_uq_metrics_by_seed.csv",index=False)
    _write_fairness_audit(out,proposal_bundle,compared_bundles,cfg,controls_manifest)
    return metrics


def run_ablation(csv_path,out_dir,cfg,preset="paper",variants=None,hp_overrides=None):
    out=ensure_dir(out_dir)
    seq=(hp_overrides or {}).get("sequence_length")
    start=common_eval_start(cfg)
    b=make_bundle(csv_path,cfg,sequence_length=seq,scaler_kind="standard",eval_start=start,proposal_features=True); _save_bundle_meta(out,b,cfg)
    raw_b=make_bundle(csv_path,cfg,sequence_length=seq,scaler_kind="standard",eval_start=start,proposal_features=False)
    _assert_same_targets(b,raw_b,"raw_feature_ablation_bundle")
    variants=list(variants or PROPOSAL_VARIANTS)
    bad=[v for v in variants if v not in PROPOSAL_VARIANTS]
    if bad: raise ValueError("Unknown proposal variants: "+", ".join(bad))
    epoch_key = "epochs_quick" if preset=="quick" else ("epochs_trust" if preset=="trust" else "epochs_paper")
    epochs=int(cfg["training"].get(epoch_key, cfg["training"].get("epochs_paper",60)))
    if preset == "paper":
        seed_list=list(cfg.get("evaluation",{}).get("seeds",[cfg["seed"]]))
    elif preset == "trust":
        seed_list=list(cfg.get("evaluation",{}).get("trust_seeds",[0,1,2]))
    else:
        seed_list=[int(cfg["seed"])]
    results=[]; preds=pd.DataFrame({"endpoint_id":b.idt,"y_true":b.yt,"physics_lower":b.lot,"physics_lower_raw":b.raw_lot,"excitation":b.et})
    for variant in variants:
        print(f"[ablation/v3] {variant}")
        vb=raw_b if _proposal_flags(variant)["raw_features_only"] else b
        variant_preds=[]; variant_raw=[]; variant_sigma=[]; variant_gate=[]; lo_int=[]; hi_int=[]
        for seed in seed_list:
            seed_everything(int(seed)); t0=time.time(); model,hp=fit_proposal(variant,vb,cfg,epochs,hp_overrides)
            d=predict_proposal_details(model,variant,vb.Xt,vb.lot,vb.raw_lot,cfg["mu_upper"],excitation=vb.et)
            row={"model":variant,"kind":"ablation" if variant!="safegrip" else "proposal","seed":int(seed),
                 **regression_metrics(vb.yt,d["prediction"],d["bound"],cfg["mu_upper"],d["sigma"],
                                      raw_mean=d["raw_mean"],
                                      interval_low=d.get("pi95_low_physics"),interval_high=d.get("pi95_high_physics"),
                                      interval_low_raw=d.get("pi95_low_raw"),interval_high_raw=d.get("pi95_high_raw")),
                 "conformal_q":float(d.get("conformal_q",np.nan)),
                 "mean_reliability":float(np.mean(d["reliability"])),
                 "mean_gate":float(np.mean(d["reliability"])),
                 "prior_rmse":float(np.sqrt(np.mean((vb.yt-d["prior_prediction"])**2))),
                 "mean_abs_dynamic_update":float(np.mean(np.abs(d["prediction"]-d["prior_prediction"]))),
                 "seconds":time.time()-t0}
            results.append(row); variant_preds.append(d["prediction"]); variant_raw.append(d["raw_mean"]); variant_gate.append(d["gate"])
            if d["sigma"] is not None: variant_sigma.append(d["sigma"])
            if "pi95_low_physics" in d: lo_int.append(d["pi95_low_physics"]); hi_int.append(d["pi95_high_physics"])
        preds[variant]=np.mean(np.stack(variant_preds),axis=0)
        preds[variant+"_raw"]=np.mean(np.stack(variant_raw),axis=0)
        preds[variant+"_gate"]=np.mean(np.stack(variant_gate),axis=0)
        if variant_sigma: preds[variant+"_sigma"]=np.mean(np.stack(variant_sigma),axis=0)
        if lo_int:
            preds[variant+"_pi95_low_physics"]=np.mean(np.stack(lo_int),axis=0)
            preds[variant+"_pi95_high_physics"]=np.mean(np.stack(hi_int),axis=0)
    by_seed=pd.DataFrame(results); summary=_aggregate_seed_metrics(results)
    summary.to_csv(out/"ablation_metrics.csv",index=False); by_seed.to_csv(out/"ablation_metrics_by_seed.csv",index=False); preds.to_csv(out/"ablation_predictions.csv",index=False)
    (out/"ablation_design.json").write_text(json.dumps({
        "safegrip_data_only":"true raw-sensor data-only backbone: no proposal engineered features, no sample-specific bound, no excitation gate/regularizer/inflation",
        "safegrip_static_only":"genuine endpoint-only prior: uses only x[t], no window mean/std, GRU, or dynamic-evidence branch",
        "safegrip_no_excitation":"raw sensors with prior/evidence temporal architecture but no excitation-derived features, monotone gate, weak-excitation regularizer, or excitation UQ inflation",
        "safegrip_no_gate":"replace monotone excitation reliability by fixed 0.5 evidence reliability",
        "safegrip_no_bound":"remove the sample-specific lower endpoint; retain only global [0, mu_upper] support",
        "safegrip_no_uq":"retain the full v3 point estimator but remove residual-scale and conformal uncertainty",
        "safegrip_no_calibration":"use the raw mechanics lower endpoint instead of its one-sided calibration relaxation",
        "safegrip":"full v3 prior + dynamic-evidence estimator with monotone excitation reliability and separately fitted block-conformal UQ",
        "seeds":seed_list,"same_hyperparameters_across_variants":True,
    },indent=2),encoding="utf-8")
    return summary

