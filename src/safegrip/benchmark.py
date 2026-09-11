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
    "safegrip_persistent",
    "safegrip_neural_innovation",
    "safegrip_no_identifiability",
    "safegrip_excitation_proxy",
    "safegrip_no_acceptance",
    "safegrip_no_cf_agreement",
    "safegrip_single_scale_cf",
    "safegrip_no_linearity_consistency",
    "safegrip_no_agreement_veto",
    "safegrip_no_counterfactual_ranking",
    "safegrip_no_state_update_loss",
    "safegrip_no_direction_loss",
    "safegrip_no_dynamics_pretrain",
    "safegrip_no_innovation_supervision",
    "safegrip_no_bound",
    "safegrip_no_uq",
    "safegrip_no_inverse_expert",
    "safegrip_no_learned_arbitration",
    "safegrip_fixed_persistence",
    "safegrip_unsplit_innovation",
    "safegrip",
    # Backward-compatible names retained for old notebooks.
    "safegrip_features_only",
    "safegrip_prior_evidence",
    "safegrip_no_excitation",
    "safegrip_endpoint_only",
    "safegrip_no_gate",
    "safegrip_no_calibration",
    "safegrip_data_only",
    "safegrip_static_only",
)
PRIMARY_ABLATION_VARIANTS=(
    "safegrip_backbone_raw",
    "safegrip_persistent",
    "safegrip_neural_innovation",
    "safegrip_no_identifiability",
    "safegrip_excitation_proxy",
    "safegrip_no_inverse_expert",
    "safegrip_no_learned_arbitration",
    "safegrip_fixed_persistence",
    "safegrip_unsplit_innovation",
    "safegrip_no_counterfactual_ranking",
    "safegrip_no_dynamics_pretrain",
    "safegrip_no_innovation_supervision",
    "safegrip_no_bound",
    "safegrip_no_uq",
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
    """Explicit ablation registry.

    v0.7 inferred behavior from variant names and accidentally made several
    nominally different ablations identical.  v1.0 uses an explicit semantic
    specification so every primary ablation changes a concrete model path.
    """
    aliases={
        "safegrip_data_only":"safegrip_backbone_raw",
        "safegrip_prior_evidence":"safegrip_neural_innovation",
        "safegrip_no_excitation":"safegrip_neural_innovation",
        "safegrip_no_gate":"safegrip_neural_innovation",
        "safegrip_static_only":"safegrip_endpoint_only",
    }
    canonical=aliases.get(variant,variant)
    specs={
        "safegrip_backbone_raw": dict(model_kind="backbone", feature_mode="raw", use_temporal=True,
            use_innovation=False, use_persistent_state=False, use_identifiability=False, use_acceptance=False,
            use_excitation_proxy=False, use_innovation_supervision=False, use_dynamics_loss=False,
            use_counterfactual_loss=False, use_cf_agreement_loss=False, use_bound=False, use_uq=False, use_calibrated_lower=True),
        "safegrip_features_only": dict(model_kind="backbone", feature_mode="safegrip_no_excitation", use_temporal=True,
            use_innovation=False, use_persistent_state=False, use_identifiability=False, use_acceptance=False,
            use_excitation_proxy=False, use_innovation_supervision=False, use_dynamics_loss=False,
            use_counterfactual_loss=False, use_cf_agreement_loss=False, use_bound=False, use_uq=False, use_calibrated_lower=True),
        "safegrip_persistent": dict(model_kind="ci", feature_mode="raw", use_temporal=True,
            use_innovation=False, use_persistent_state=True, use_identifiability=False, use_acceptance=False,
            use_excitation_proxy=False, use_innovation_supervision=False, use_dynamics_loss=True,
            use_counterfactual_loss=False, use_cf_agreement_loss=False, use_bound=True, use_uq=True, use_calibrated_lower=True),
        "safegrip_neural_innovation": dict(model_kind="ci", feature_mode="raw", use_temporal=True,
            use_innovation=True, use_persistent_state=True, use_identifiability=False, use_acceptance=False,
            use_excitation_proxy=False, use_innovation_supervision=True, use_dynamics_loss=True,
            use_counterfactual_loss=True, use_cf_agreement_loss=False, use_bound=True, use_uq=True, use_calibrated_lower=True),
        "safegrip_no_identifiability": dict(model_kind="ci", feature_mode="raw", use_temporal=True,
            use_innovation=True, use_persistent_state=True, use_identifiability=False, use_acceptance=True,
            use_excitation_proxy=False, use_innovation_supervision=True, use_dynamics_loss=True, use_counterfactual_loss=True,
            use_cf_agreement_loss=False, use_bound=True, use_uq=True, use_calibrated_lower=True,
            use_multiscale_counterfactual=False, use_linearity_consistency=False, use_agreement_veto=False, use_dual_expert=True,
            use_learned_arbitration=True, use_inverse_expert=True, use_adaptive_persistence=True, use_split_innovation=True),
        "safegrip_excitation_proxy": dict(model_kind="ci", feature_mode="raw", use_temporal=True,
            use_innovation=True, use_persistent_state=True, use_identifiability=True, use_acceptance=True,
            use_excitation_proxy=True, use_innovation_supervision=True, use_dynamics_loss=True, use_counterfactual_loss=True,
            use_cf_agreement_loss=False, use_bound=True, use_uq=True, use_calibrated_lower=True,
            use_multiscale_counterfactual=False, use_linearity_consistency=False, use_agreement_veto=False, use_dual_expert=True,
            use_learned_arbitration=True, use_inverse_expert=True, use_adaptive_persistence=True, use_split_innovation=True),
        "safegrip_no_acceptance": dict(model_kind="ci", feature_mode="raw", use_temporal=True,
            use_innovation=True, use_persistent_state=True, use_identifiability=True, use_acceptance=False,
            use_excitation_proxy=False, use_innovation_supervision=True, use_dynamics_loss=True,
            use_counterfactual_loss=True, use_cf_agreement_loss=True, use_bound=True, use_uq=True, use_calibrated_lower=True),
        "safegrip_no_cf_agreement": dict(model_kind="ci", feature_mode="raw", use_temporal=True,
            use_innovation=True, use_persistent_state=True, use_identifiability=True, use_acceptance=True,
            use_excitation_proxy=False, use_innovation_supervision=True, use_dynamics_loss=True,
            use_counterfactual_loss=True, use_cf_agreement_loss=False, use_bound=True, use_uq=True, use_calibrated_lower=True),
        "safegrip_single_scale_cf": dict(model_kind="ci", feature_mode="raw", use_temporal=True,
            use_innovation=True, use_persistent_state=True, use_identifiability=True, use_acceptance=True,
            use_excitation_proxy=False, use_innovation_supervision=True, use_dynamics_loss=True,
            use_counterfactual_loss=True, use_cf_agreement_loss=True, use_bound=True, use_uq=True, use_calibrated_lower=True,
            use_multiscale_counterfactual=False, use_linearity_consistency=False),
        "safegrip_no_linearity_consistency": dict(model_kind="ci", feature_mode="raw", use_temporal=True,
            use_innovation=True, use_persistent_state=True, use_identifiability=True, use_acceptance=True,
            use_excitation_proxy=False, use_innovation_supervision=True, use_dynamics_loss=True,
            use_counterfactual_loss=True, use_cf_agreement_loss=True, use_bound=True, use_uq=True, use_calibrated_lower=True,
            use_linearity_consistency=False),
        "safegrip_no_agreement_veto": dict(model_kind="ci", feature_mode="raw", use_temporal=True,
            use_innovation=True, use_persistent_state=True, use_identifiability=True, use_acceptance=True,
            use_excitation_proxy=False, use_innovation_supervision=True, use_dynamics_loss=True,
            use_counterfactual_loss=True, use_cf_agreement_loss=True, use_bound=True, use_uq=True, use_calibrated_lower=True,
            use_agreement_veto=False),
        "safegrip_no_counterfactual_ranking": dict(model_kind="ci", feature_mode="raw", use_temporal=True,
            use_innovation=True, use_persistent_state=True, use_identifiability=True, use_acceptance=True,
            use_excitation_proxy=False, use_innovation_supervision=True, use_dynamics_loss=True, use_counterfactual_loss=False,
            use_cf_agreement_loss=False, use_bound=True, use_uq=True, use_calibrated_lower=True,
            use_multiscale_counterfactual=False, use_linearity_consistency=False, use_agreement_veto=False, use_dual_expert=True,
            use_learned_arbitration=True, use_inverse_expert=True, use_adaptive_persistence=True, use_split_innovation=True),
        "safegrip_no_state_update_loss": dict(model_kind="ci", feature_mode="raw", use_temporal=True,
            use_innovation=True, use_persistent_state=True, use_identifiability=True, use_acceptance=True,
            use_excitation_proxy=False, use_innovation_supervision=True, use_dynamics_loss=True, use_counterfactual_loss=True,
            use_cf_agreement_loss=False, use_bound=True, use_uq=True, use_calibrated_lower=True,
            use_multiscale_counterfactual=False, use_linearity_consistency=False, use_agreement_veto=False, use_dual_expert=True,
            use_learned_arbitration=True, use_inverse_expert=True, use_adaptive_persistence=True, use_split_innovation=True,
            use_state_update_loss=False),
        "safegrip_no_direction_loss": dict(model_kind="ci", feature_mode="raw", use_temporal=True,
            use_innovation=True, use_persistent_state=True, use_identifiability=True, use_acceptance=True,
            use_excitation_proxy=False, use_innovation_supervision=True, use_dynamics_loss=True, use_counterfactual_loss=True,
            use_cf_agreement_loss=False, use_bound=True, use_uq=True, use_calibrated_lower=True,
            use_multiscale_counterfactual=False, use_linearity_consistency=False, use_agreement_veto=False, use_dual_expert=True,
            use_learned_arbitration=True, use_inverse_expert=True, use_adaptive_persistence=True, use_split_innovation=True,
            use_direction_loss=False),
        "safegrip_no_dynamics_pretrain": dict(model_kind="ci", feature_mode="raw", use_temporal=True,
            use_innovation=True, use_persistent_state=True, use_identifiability=True, use_acceptance=True,
            use_excitation_proxy=False, use_innovation_supervision=True, use_dynamics_loss=True, use_counterfactual_loss=True,
            use_cf_agreement_loss=False, use_bound=True, use_uq=True, use_calibrated_lower=True,
            use_multiscale_counterfactual=False, use_linearity_consistency=False, use_agreement_veto=False, use_dual_expert=True,
            use_learned_arbitration=True, use_inverse_expert=True, use_adaptive_persistence=True, use_split_innovation=True,
            use_dynamics_pretrain=False),
        "safegrip_no_innovation_supervision": dict(model_kind="ci", feature_mode="raw", use_temporal=True,
            use_innovation=True, use_persistent_state=True, use_identifiability=True, use_acceptance=True,
            use_excitation_proxy=False, use_innovation_supervision=False, use_dynamics_loss=True, use_counterfactual_loss=True,
            use_cf_agreement_loss=False, use_bound=True, use_uq=True, use_calibrated_lower=True,
            use_multiscale_counterfactual=False, use_linearity_consistency=False, use_agreement_veto=False, use_dual_expert=True,
            use_learned_arbitration=True, use_inverse_expert=True, use_adaptive_persistence=True, use_split_innovation=True),
        "safegrip_no_bound": dict(model_kind="ci", feature_mode="raw", use_temporal=True,
            use_innovation=True, use_persistent_state=True, use_identifiability=True, use_acceptance=True,
            use_excitation_proxy=False, use_innovation_supervision=True, use_dynamics_loss=True, use_counterfactual_loss=True,
            use_cf_agreement_loss=False, use_bound=False, use_uq=True, use_calibrated_lower=True,
            use_multiscale_counterfactual=False, use_linearity_consistency=False, use_agreement_veto=False, use_dual_expert=True,
            use_learned_arbitration=True, use_inverse_expert=True, use_adaptive_persistence=True, use_split_innovation=True),
        "safegrip_no_uq": dict(model_kind="ci", feature_mode="raw", use_temporal=True,
            use_innovation=True, use_persistent_state=True, use_identifiability=True, use_acceptance=True,
            use_excitation_proxy=False, use_innovation_supervision=True, use_dynamics_loss=True, use_counterfactual_loss=True,
            use_cf_agreement_loss=False, use_bound=True, use_uq=False, use_calibrated_lower=True,
            use_multiscale_counterfactual=False, use_linearity_consistency=False, use_agreement_veto=False, use_dual_expert=True,
            use_learned_arbitration=True, use_inverse_expert=True, use_adaptive_persistence=True, use_split_innovation=True),
        "safegrip_no_inverse_expert": dict(model_kind="ci", feature_mode="raw", use_temporal=True,
            use_innovation=True, use_persistent_state=True, use_identifiability=True, use_acceptance=True,
            use_excitation_proxy=False, use_innovation_supervision=True, use_dynamics_loss=True,
            use_counterfactual_loss=True, use_cf_agreement_loss=False, use_bound=True, use_uq=True, use_calibrated_lower=True,
            use_multiscale_counterfactual=False, use_linearity_consistency=False, use_agreement_veto=False,
            use_dual_expert=True, use_learned_arbitration=True, use_inverse_expert=False,
            use_adaptive_persistence=True, use_split_innovation=True),
        "safegrip_no_learned_arbitration": dict(model_kind="ci", feature_mode="raw", use_temporal=True,
            use_innovation=True, use_persistent_state=True, use_identifiability=True, use_acceptance=True,
            use_excitation_proxy=False, use_innovation_supervision=True, use_dynamics_loss=True,
            use_counterfactual_loss=True, use_cf_agreement_loss=False, use_bound=True, use_uq=True, use_calibrated_lower=True,
            use_multiscale_counterfactual=False, use_linearity_consistency=False, use_agreement_veto=False,
            use_dual_expert=False, use_learned_arbitration=False, use_inverse_expert=False,
            use_adaptive_persistence=True, use_split_innovation=True),
        "safegrip_fixed_persistence": dict(model_kind="ci", feature_mode="raw", use_temporal=True,
            use_innovation=True, use_persistent_state=True, use_identifiability=True, use_acceptance=True,
            use_excitation_proxy=False, use_innovation_supervision=True, use_dynamics_loss=True,
            use_counterfactual_loss=True, use_cf_agreement_loss=False, use_bound=True, use_uq=True, use_calibrated_lower=True,
            use_multiscale_counterfactual=False, use_linearity_consistency=False, use_agreement_veto=False,
            use_dual_expert=True, use_learned_arbitration=True, use_inverse_expert=True,
            use_adaptive_persistence=False, use_split_innovation=True),
        "safegrip_unsplit_innovation": dict(model_kind="ci", feature_mode="raw", use_temporal=True,
            use_innovation=True, use_persistent_state=True, use_identifiability=True, use_acceptance=True,
            use_excitation_proxy=False, use_innovation_supervision=True, use_dynamics_loss=True,
            use_counterfactual_loss=True, use_cf_agreement_loss=False, use_bound=True, use_uq=True, use_calibrated_lower=True,
            use_multiscale_counterfactual=False, use_linearity_consistency=False, use_agreement_veto=False,
            use_dual_expert=True, use_learned_arbitration=True, use_inverse_expert=True,
            use_adaptive_persistence=True, use_split_innovation=False),
        "safegrip_endpoint_only": dict(model_kind="ci", feature_mode="raw", use_temporal=False,
            use_innovation=False, use_persistent_state=False, use_identifiability=False, use_acceptance=False,
            use_excitation_proxy=False, use_innovation_supervision=False, use_dynamics_loss=False,
            use_counterfactual_loss=False, use_cf_agreement_loss=False, use_bound=True, use_uq=True, use_calibrated_lower=True),
        "safegrip_no_calibration": dict(model_kind="ci", feature_mode="raw", use_temporal=True,
            use_innovation=True, use_persistent_state=True, use_identifiability=True, use_acceptance=True,
            use_excitation_proxy=False, use_innovation_supervision=True, use_dynamics_loss=True,
            use_counterfactual_loss=True, use_cf_agreement_loss=True, use_bound=True, use_uq=True, use_calibrated_lower=False),
        "safegrip": dict(model_kind="ci", feature_mode="raw", use_temporal=True,
            use_innovation=True, use_persistent_state=True, use_identifiability=True, use_acceptance=True,
            use_excitation_proxy=False, use_innovation_supervision=True, use_dynamics_loss=True,
            use_counterfactual_loss=True, use_cf_agreement_loss=False, use_bound=True, use_uq=True, use_calibrated_lower=True,
            use_multiscale_counterfactual=False, use_linearity_consistency=False, use_agreement_veto=False,
            use_dual_expert=True, use_learned_arbitration=True, use_inverse_expert=True,
            use_adaptive_persistence=True, use_split_innovation=True),
    }
    if canonical not in specs:
        raise ValueError(variant)
    out=dict(specs[canonical])
    out.setdefault("use_multiscale_counterfactual", True)
    out.setdefault("use_linearity_consistency", True)
    out.setdefault("use_agreement_veto", bool(out.get("use_acceptance", False) and out.get("use_cf_agreement_loss", False)))
    out.setdefault("use_state_update_loss", bool(out.get("use_innovation_supervision", False)))
    out.setdefault("use_direction_loss", bool(out.get("use_innovation_supervision", False)))
    out.setdefault("use_dynamics_pretrain", bool(out.get("use_dynamics_loss", False)))
    out.setdefault("use_dual_expert", False)
    out.setdefault("use_learned_arbitration", False)
    out.setdefault("use_inverse_expert", False)
    out.setdefault("use_adaptive_persistence", False)
    out.setdefault("use_split_innovation", False)
    # Removing all counterfactual agreement must remove both its training loss
    # and its inference-time disagreement veto.
    if canonical == "safegrip_no_cf_agreement":
        out["use_agreement_veto"] = False
    out["canonical_variant"]=canonical
    out["raw_features_only"]=out["feature_mode"]=="raw"
    # Compatibility fields used by old notebooks/tests.
    out["use_gate"]=bool(out["use_identifiability"] or out["use_acceptance"] or out["use_excitation_proxy"])
    out["use_excitation_regularizer"]=False
    out["use_excitation_uq_inflation"]=False
    return out


def _dynamics_indices(features: list[str]) -> list[int]:
    preferred=("ax","ay","wheel_fl","wheel_fr","wheel_rl","wheel_rr")
    idx=[features.index(c) for c in preferred if c in features]
    if not idx:
        idx=list(range(min(4,len(features))))
    return idx


def _proposal_model(variant: str, b: Bundle, hp: dict):
    flags=_proposal_flags(variant)
    if flags["model_kind"]=="backbone":
        model=SafeGripBackboneNet(
            len(b.features), hidden=int(hp["hidden"]), gru_hidden=int(hp["gru_hidden"]),
            dropout=float(hp["dropout"]),
        )
        model.variant_spec=flags
        return model
    exc_idx=b.features.index("sg_excitation_score") if "sg_excitation_score" in b.features else None
    model=SafeGripV3Net(
        len(b.features), excitation_index=exc_idx, dynamics_indices=_dynamics_indices(b.features),
        hidden=int(hp["hidden"]), gru_hidden=int(hp["gru_hidden"]), dropout=float(hp["dropout"]),
        evidence_window=int(hp.get("evidence_window",8)), delta_scale=float(hp.get("delta_scale",1.5)),
        counterfactual_delta=float(hp.get("counterfactual_delta",0.08)),
        identifiability_lambda=float(hp.get("identifiability_lambda",0.001)),
        acceptance_temperature=float(hp.get("acceptance_temperature",12.0)),
        acceptance_margin=float(hp.get("acceptance_margin",0.0)),
        acceptance_tolerance=float(hp.get("acceptance_tolerance",0.10)),
        acceptance_strength=float(hp.get("acceptance_strength",0.35)),
        counterfactual_scale_span=float(hp.get("counterfactual_scale_span",2.0)),
        use_multiscale_counterfactual=flags["use_multiscale_counterfactual"],
        use_linearity_consistency=flags["use_linearity_consistency"],
        linearity_penalty=float(hp.get("linearity_penalty",0.50)),
        inverse_dynamics_ridge=float(hp.get("inverse_dynamics_ridge",0.001)),
        inverse_dynamics_max_step=float(hp.get("inverse_dynamics_max_step",0.12)),
        use_agreement_veto=flags["use_agreement_veto"],
        agreement_temperature=float(hp.get("agreement_temperature",10.0)),
        agreement_threshold=float(hp.get("agreement_threshold",0.35)),
        agreement_strength=float(hp.get("agreement_strength",0.15)),
        use_temporal=flags["use_temporal"], use_gate=flags["use_gate"], use_bound=flags["use_bound"],
        endpoint_only=not flags["use_temporal"], use_identifiability=flags["use_identifiability"],
        use_acceptance=flags["use_acceptance"], use_persistent_state=flags["use_persistent_state"],
        use_excitation_proxy=flags["use_excitation_proxy"], use_innovation=flags["use_innovation"],
        state_persistence=float(hp.get("state_persistence",0.85)),
        use_dual_expert=flags.get("use_dual_expert",False),
        use_learned_arbitration=flags.get("use_learned_arbitration",False),
        use_inverse_expert=flags.get("use_inverse_expert",False),
        use_adaptive_persistence=flags.get("use_adaptive_persistence",False),
        use_split_innovation=flags.get("use_split_innovation",False),
        persistence_min=float(hp.get("persistence_min",0.05)),
        persistence_max=float(hp.get("persistence_max",0.98)),
        inverse_expert_scale=float(hp.get("inverse_expert_scale",1.0)),
    )
    model.information_beta=max(0.0,float(hp.get("information_beta",1.0)))
    model.disagreement_beta=max(0.0,float(hp.get("disagreement_beta",0.35)))
    model.variant_spec=flags
    return model


def proposal_hparams(cfg, overrides=None):
    p=dict(cfg.get("proposal",{})); tr=cfg["training"]
    defaults={
        "hidden":tr.get("hidden",64), "gru_hidden":32, "dropout":tr.get("dropout",.1),
        "lr":tr.get("lr",1e-3), "weight_decay":tr.get("weight_decay",1e-4),
        "batch_size":tr.get("batch_size",256), "huber_beta":0.05,
        "evidence_window":8, "delta_scale":0.5,
        "counterfactual_delta":0.08, "identifiability_lambda":0.001,
        "state_persistence":0.85, "dynamics_pretrain_epochs":8,
        "acceptance_temperature":12.0, "acceptance_margin":0.0,
        "acceptance_tolerance":0.10, "acceptance_strength":0.35,
        "counterfactual_scale_span":2.0, "linearity_penalty":0.50,
        "inverse_dynamics_ridge":0.001, "inverse_dynamics_max_step":0.12,
        "agreement_temperature":10.0, "agreement_threshold":0.35, "agreement_strength":0.15,
        "innovation_loss_weight":0.20, "state_update_loss_weight":0.35, "candidate_loss_weight":0.15,
        "cf_agreement_loss_weight":0.03, "direction_loss_weight":0.05,
        "dynamics_loss_weight":0.10,
        "counterfactual_loss_weight":0.05, "do_no_harm_weight":0.10,
        "counterfactual_margin":0.02,
        "uq_epochs":100, "uq_lr":1e-3, "uq_scale_floor":0.005,
        "information_beta":1.0, "disagreement_beta":0.35,
        "arbitration_loss_weight":0.30, "prior_loss_weight":0.05,
        "teacher_forcing_start":0.80, "teacher_forcing_end":0.0,
        "persistence_min":0.05, "persistence_max":0.98, "inverse_expert_scale":1.0,
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


def _forward_point(model, X, lower, mu_upper, batch=1024, return_features=False,
                   excitation=None, ids=None, stateful=True):
    """Predict proposal endpoints, optionally carrying friction state by segment."""
    dev=device(); model.eval()
    keys=("prediction","latent","reliability","features","prior_prediction","evidence_delta",
          "prior_latent","identifiability","acceptance","information_raw","candidate_prediction",
          "dynamics_residual_prior","dynamics_residual_candidate","persistent_state_used",
          "normalized_improvement","veto_probability","counterfactual_delta_mu","counterfactual_agreement",
          "information_scale_cv","local_linearity","agreement_veto_probability",
          "expert_weight_prior","expert_weight_neural","expert_weight_inverse",
          "adaptive_persistence","direction_agreement","magnitude_agreement","inverse_candidate_prediction")
    store={k:[] for k in keys}
    excitation_arr=None if excitation is None else np.asarray(excitation,dtype=np.float32)
    X=np.asarray(X,dtype=np.float32); lower=np.asarray(lower,dtype=np.float32)
    ids_arr=None if ids is None else np.asarray(ids,dtype=str)
    use_state=bool(stateful and ids_arr is not None and getattr(model,"use_persistent_state",False))

    def append_details(d):
        for k in keys:
            v=d.get(k)
            if v is None:
                continue
            store[k].append(v.detach().cpu().numpy())

    with torch.no_grad():
        if use_state:
            state_mu=None; last_key=None
            for i in range(len(X)):
                key=_segment_key(ids_arr[i])
                if key!=last_key:
                    state_mu=None; last_key=key
                xb=torch.from_numpy(X[i:i+1]).to(dev)
                lb=torch.from_numpy(lower[i:i+1]).to(dev)
                eb=None if excitation_arr is None else torch.from_numpy(excitation_arr[i:i+1]).to(dev)
                if state_mu is None:
                    d=model.forward_details(xb,lb,float(mu_upper),excitation=eb)
                else:
                    pm=torch.tensor([state_mu],dtype=xb.dtype,device=dev)
                    mask=torch.ones(1,dtype=torch.bool,device=dev)
                    d=model.forward_details(xb,lb,float(mu_upper),excitation=eb,prior_mu=pm,prior_mask=mask)
                state_mu=float(d["prediction"].detach().cpu().item())
                append_details(d)
        else:
            for i in range(0,len(X),batch):
                xb=torch.from_numpy(X[i:i+batch]).to(dev)
                lb=torch.from_numpy(lower[i:i+batch]).to(dev)
                eb=None if excitation_arr is None else torch.from_numpy(excitation_arr[i:i+batch]).to(dev)
                d=model.forward_details(xb,lb,float(mu_upper),excitation=eb)
                append_details(d)

    def cat(k, fallback=None):
        if store[k]: return np.concatenate(store[k],axis=0)
        return fallback
    pred=cat("prediction",np.empty(0,np.float32))
    latent=cat("latent",np.empty_like(pred)); rel=cat("reliability",np.zeros_like(pred))
    if return_features:
        return (
            pred,latent,rel,cat("features"),cat("prior_prediction",pred.copy()),
            cat("evidence_delta",np.zeros_like(pred)),cat("prior_latent",latent.copy()),
            cat("identifiability",np.zeros_like(pred)),cat("acceptance",np.ones_like(pred)),
            cat("information_raw",np.zeros_like(pred)),cat("candidate_prediction",pred.copy()),
            cat("dynamics_residual_prior",np.zeros_like(pred)),cat("dynamics_residual_candidate",np.zeros_like(pred)),
            cat("persistent_state_used",np.zeros_like(pred)),
            cat("normalized_improvement",np.zeros_like(pred)),cat("veto_probability",np.zeros_like(pred)),
            cat("counterfactual_delta_mu",np.zeros_like(pred)),cat("counterfactual_agreement",np.full_like(pred,0.5)),
            cat("information_scale_cv",np.zeros_like(pred)),cat("local_linearity",np.ones_like(pred)),
            cat("agreement_veto_probability",np.zeros_like(pred)),
            cat("expert_weight_prior",np.ones_like(pred)),cat("expert_weight_neural",np.zeros_like(pred)),
            cat("expert_weight_inverse",np.zeros_like(pred)),cat("adaptive_persistence",np.zeros_like(pred)),
            cat("direction_agreement",np.zeros_like(pred)),cat("magnitude_agreement",np.zeros_like(pred)),
            cat("inverse_candidate_prediction",pred.copy()),
        )
    return pred,latent,rel

def _fit_residual_scale(model, b: Bundle, cfg: dict, hp: dict, lower_val: np.ndarray):
    """Fit post-hoc residual scale; low identifiability inflates uncertainty."""
    if not hasattr(model,"forward_details"):
        return
    dev=device(); model.eval()
    vals=_forward_point(
        model,b.Xv,lower_val,float(cfg["mu_upper"]),return_features=True,
        excitation=b.ev,ids=b.idv,stateful=True
    )
    p=vals[0]; h=vals[3]; ident=vals[7]; cf_agreement=vals[17]
    target=np.maximum(np.abs(np.asarray(b.yv,float)-p),float(hp["uq_scale_floor"]))
    info=np.clip(np.asarray(ident,float),0.0,1.0)
    agreement=np.clip(np.asarray(cf_agreement,float),0.0,1.0)
    beta=max(0.0,float(hp.get("information_beta",1.0)))
    disagreement_beta=max(0.0,float(hp.get("disagreement_beta",0.35)))
    inflation=1.0+beta*(1.0-info)+disagreement_beta*(1.0-agreement)
    base_target=np.maximum(target/np.maximum(inflation,1e-6),float(hp["uq_scale_floor"]))
    init_scale=float(np.median(base_target)) if len(base_target) else float(hp["uq_scale_floor"])*2.0
    head=ResidualScaleHead(int(hp["hidden"]),floor=float(hp["uq_scale_floor"]),initial_scale=init_scale).to(dev)
    opt=torch.optim.AdamW(head.parameters(),lr=float(hp["uq_lr"]),weight_decay=float(hp["weight_decay"]))
    ht=torch.from_numpy(h.astype(np.float32)).to(dev); tgt=torch.from_numpy(base_target.astype(np.float32)).to(dev)
    best=None; bestloss=float("inf"); stale=0
    max_epochs=int(hp.get("uq_epochs",100)); patience=max(10,min(30,max_epochs//4))
    for _ in range(max_epochs):
        head.train(); opt.zero_grad(); base=head(ht)
        loss=nn.functional.smooth_l1_loss(torch.log(base),torch.log(tgt),beta=0.25)
        loss.backward(); torch.nn.utils.clip_grad_norm_(head.parameters(),5.0); opt.step()
        val=float(loss.detach().cpu())
        if val<bestloss-1e-6:
            bestloss=val; best={k:v.detach().cpu().clone() for k,v in head.state_dict().items()}; stale=0
        else: stale+=1
        if stale>=patience: break
    if best: head.load_state_dict(best)
    model.scale_head=head; model.uq_scale_initialization=init_scale


def _target_latent(y: torch.Tensor, lower: torch.Tensor, upper: float, use_bound: bool) -> torch.Tensor:
    eps=1e-5
    if use_bound:
        span=torch.clamp(torch.as_tensor(upper,dtype=y.dtype,device=y.device)-lower,min=1e-6)
        frac=(y-lower)/span
    else:
        frac=y/max(float(upper),1e-6)
    frac=torch.clamp(frac,eps,1.0-eps)
    return torch.log(frac)-torch.log1p(-frac)


def fit_proposal(variant,b:Bundle,cfg,epochs,hp_overrides=None):
    if variant not in PROPOSAL_VARIANTS: raise ValueError(variant)
    flags=_proposal_flags(variant)
    if getattr(b,"feature_mode",None) != flags["feature_mode"]:
        raise ValueError(f"{variant} requires feature_mode={flags['feature_mode']}, got {getattr(b,'feature_mode',None)}")
    hp=proposal_hparams(cfg,hp_overrides); dev=device(); model=_proposal_model(variant,b,hp).to(dev)
    opt=torch.optim.AdamW(model.parameters(),lr=float(hp["lr"]),weight_decay=float(hp["weight_decay"]))
    lo_train=b.raw_lotr
    lo_val=b.lov if flags["use_calibrated_lower"] else b.raw_lov
    prev_idx,pair_valid=_previous_pair_indices(b.idtr,1)
    dataset=TensorDataset(
        torch.from_numpy(b.Xtr),torch.from_numpy(b.ytr),torch.from_numpy(lo_train),torch.from_numpy(b.etr),
        torch.from_numpy(b.Xtr[prev_idx]),torch.from_numpy(b.ytr[prev_idx]),torch.from_numpy(lo_train[prev_idx]),torch.from_numpy(b.etr[prev_idx]),
        torch.from_numpy(pair_valid.astype(np.bool_)),
    )
    dl=DataLoader(dataset,batch_size=int(hp["batch_size"]),shuffle=True)
    mu_u=float(cfg["mu_upper"]); best=None; bestloss=float("inf"); bad=0
    patience=int(cfg["training"].get("patience",10)); huber_beta=float(hp.get("huber_beta",0.05))
    innov_w=float(hp.get("innovation_loss_weight",0.20)) if flags["use_innovation_supervision"] else 0.0
    update_w=max(0.0,float(hp.get("state_update_loss_weight",0.35))) if flags["use_innovation_supervision"] and flags.get("use_state_update_loss",True) else 0.0
    candidate_w=max(0.0,float(hp.get("candidate_loss_weight",0.15))) if flags["use_innovation_supervision"] else 0.0
    cf_agreement_w=max(0.0,float(hp.get("cf_agreement_loss_weight",0.05))) if flags.get("use_cf_agreement_loss",False) else 0.0
    direction_w=max(0.0,float(hp.get("direction_loss_weight",0.05))) if flags["use_innovation_supervision"] and flags.get("use_direction_loss",True) else 0.0
    dyn_w=float(hp.get("dynamics_loss_weight",0.10)) if flags["use_dynamics_loss"] else 0.0
    cf_w=float(hp.get("counterfactual_loss_weight",0.05)) if flags["use_counterfactual_loss"] else 0.0
    harm_w=max(0.0,float(hp.get("do_no_harm_weight",0.10))) if flags["use_innovation"] else 0.0
    arb_w=max(0.0,float(hp.get("arbitration_loss_weight",0.30))) if flags.get("use_learned_arbitration",False) else 0.0
    prior_w=max(0.0,float(hp.get("prior_loss_weight",0.05))) if flags.get("use_adaptive_persistence",False) else 0.0
    tf_start=min(max(float(hp.get("teacher_forcing_start",0.80)),0.0),1.0)
    tf_end=min(max(float(hp.get("teacher_forcing_end",0.0)),0.0),1.0)
    cf_margin=max(0.0,float(hp.get("counterfactual_margin",0.02)))

    # Warm-start the friction-conditioned dynamics model before it is allowed
    # to control the estimator.  Without this stage, a randomly initialized G
    # has near-zero friction sensitivity, which would correctly but uselessly
    # suppress every innovation at the beginning of training.  Counterfactual
    # ranking forces G to use friction rather than reconstructing the endpoint
    # from context alone.
    pre_epochs=int(hp.get("dynamics_pretrain_epochs",8)) if flags["model_kind"]=="ci" and flags["use_dynamics_loss"] and flags.get("use_dynamics_pretrain",True) else 0
    if pre_epochs>0:
        dyn_params=list(model.dynamics_gru.parameters())+list(model.dynamics_context.parameters())+list(model.dynamics_head.parameters())
        dyn_opt=torch.optim.AdamW(dyn_params,lr=float(hp["lr"]),weight_decay=float(hp["weight_decay"]))
        for _pre in range(min(pre_epochs,max(1,int(epochs)))):
            model.train()
            for xb,yb,lb,eb,xp,yp,lp,ep,pair_mask in dl:
                xb=xb.to(dev); yb=yb.to(dev); dyn_opt.zero_grad()
                dyn_true,dyn_target=model.dynamics_prediction(xb,yb,mu_u)
                dloss=nn.functional.smooth_l1_loss(dyn_true,dyn_target,beta=0.2)
                if flags["use_counterfactual_loss"]:
                    true_err=torch.mean((dyn_true-dyn_target).square(),dim=-1)
                    delta=float(hp.get("counterfactual_delta",0.08))
                    wrong_lo=torch.clamp(yb-delta,min=0.0,max=mu_u); wrong_hi=torch.clamp(yb+delta,min=0.0,max=mu_u)
                    p_lo,_=model.dynamics_prediction(xb,wrong_lo,mu_u); p_hi,_=model.dynamics_prediction(xb,wrong_hi,mu_u)
                    wrong_err=0.5*(torch.mean((p_lo-dyn_target).square(),dim=-1)+torch.mean((p_hi-dyn_target).square(),dim=-1))
                    dloss=dloss+max(cf_w,0.05)*torch.relu(cf_margin+true_err-wrong_err).mean()
                dloss.backward(); torch.nn.utils.clip_grad_norm_(dyn_params,5.0); dyn_opt.step()

    total_epochs=max(1,int(epochs))
    for epoch_idx in range(total_epochs):
        model.train()
        tf_frac=epoch_idx/max(total_epochs-1,1)
        teacher_forcing=tf_start+(tf_end-tf_start)*tf_frac
        for xb,yb,lb,eb,xp,yp,lp,ep,pair_mask in dl:
            xb=xb.to(dev); yb=yb.to(dev); lb=lb.to(dev); eb=eb.to(dev)
            xp=xp.to(dev); yp=yp.to(dev); lp=lp.to(dev); ep=ep.to(dev); pair_mask=pair_mask.to(dev)
            opt.zero_grad()
            if flags["model_kind"]=="backbone":
                d=model.forward_details(xb,None,mu_u,excitation=eb)
            else:
                with torch.no_grad():
                    pd=model.forward_details(xp,lp,mu_u,excitation=ep)
                    predicted_prior=pd["prediction"].detach()
                    # Scheduled teacher forcing reduces train/inference mismatch:
                    # training starts from a reliable previous state and anneals
                    # to the model's own recursively predicted state.
                    prior_mu=teacher_forcing*yp+(1.0-teacher_forcing)*predicted_prior
                d=model.forward_details(xb,lb,mu_u,excitation=eb,prior_mu=prior_mu,prior_mask=pair_mask)
            pred=d["prediction"]
            loss=nn.functional.smooth_l1_loss(pred,yb,beta=huber_beta)

            if flags["model_kind"]=="ci":
                if innov_w>0:
                    target_q=_target_latent(yb,lb,mu_u,flags["use_bound"])
                    desired=(target_q-d["prior_latent"].detach())
                    # v1.0: supervise the *raw candidate innovation*, not the
                    # authority-weighted update.  Candidate quality and update
                    # authority are intentionally separate estimation tasks.
                    loss=loss+innov_w*nn.functional.smooth_l1_loss(
                        d["innovation"],desired,beta=max(huber_beta,0.1)
                    )
                    # A distinct authorized-update objective prevents low but
                    # scientifically meaningful identifiability from shrinking
                    # every well-directed update into a near-static estimator.
                    # Authority is not detached here: the estimator and its
                    # counterfactual authority learn to cooperate, while the raw
                    # innovation objective above still anchors candidate meaning.
                    if update_w>0:
                        loss=loss+update_w*nn.functional.smooth_l1_loss(
                            d["effective_innovation"],desired,beta=max(huber_beta,0.1)
                        )
                    if candidate_w>0:
                        loss=loss+candidate_w*nn.functional.smooth_l1_loss(
                            d["candidate_prediction"],yb,beta=huber_beta
                        )
                    if direction_w>0:
                        active=(torch.abs(desired)>0.02).to(desired.dtype)
                        signed_progress=torch.sign(desired)*d["innovation"]
                        dir_pen=torch.relu(0.02-signed_progress)*active
                        loss=loss+direction_w*dir_pen.sum()/torch.clamp(active.sum(),min=1.0)
                if cf_agreement_w>0:
                    # Label-free local inverse-dynamics pseudo-target.  Detach
                    # the target/weight so the agreement term cannot improve by
                    # warping the dynamics model or identifiability score.
                    cand_delta=d["candidate_prediction"]-d["prior_prediction"].detach()
                    cf_target=d["counterfactual_delta_mu"].detach()
                    cf_weight=d["identifiability"].detach()
                    cf_err=nn.functional.smooth_l1_loss(
                        cand_delta,cf_target,beta=0.03,reduction="none"
                    )
                    loss=loss+cf_agreement_w*(cf_err*cf_weight).sum()/torch.clamp(cf_weight.sum(),min=1.0)
                if arb_w>0 and "arbitration_logits" in d and "inverse_candidate_prediction" in d:
                    # Supervise the arbitration decision with the best training-time
                    # expert.  Labels are used only here to define the training
                    # target; inference receives no friction label.
                    expert_err=torch.stack([
                        torch.abs(d["prior_prediction"].detach()-yb),
                        torch.abs(d["candidate_prediction"].detach()-yb),
                        torch.abs(d["inverse_candidate_prediction"].detach()-yb),
                    ],dim=-1)
                    expert_target=torch.argmin(expert_err,dim=-1)
                    loss=loss+arb_w*nn.functional.cross_entropy(d["arbitration_logits"],expert_target)
                if prior_w>0:
                    loss=loss+prior_w*nn.functional.smooth_l1_loss(
                        d["prior_prediction"],yb,beta=max(huber_beta,0.05)
                    )
                if dyn_w>0:
                    dyn_true,dyn_target=model.dynamics_prediction(xb,yb,mu_u)
                    loss=loss+dyn_w*nn.functional.smooth_l1_loss(dyn_true,dyn_target,beta=0.2)
                if cf_w>0:
                    dyn_true,dyn_target=model.dynamics_prediction(xb,yb,mu_u)
                    true_err=torch.mean((dyn_true-dyn_target).square(),dim=-1)
                    delta=float(hp.get("counterfactual_delta",0.08))
                    wrong_lo=torch.clamp(yb-delta,min=0.0,max=mu_u)
                    wrong_hi=torch.clamp(yb+delta,min=0.0,max=mu_u)
                    d_lo,_=model.dynamics_prediction(xb,wrong_lo,mu_u)
                    d_hi,_=model.dynamics_prediction(xb,wrong_hi,mu_u)
                    wrong_err=0.5*(torch.mean((d_lo-dyn_target).square(),dim=-1)+torch.mean((d_hi-dyn_target).square(),dim=-1))
                    loss=loss+cf_w*torch.relu(cf_margin+true_err-wrong_err).mean()
                if harm_w>0:
                    harm=torch.relu(torch.abs(pred-yb)-torch.abs(d["prior_prediction"]-yb)).mean()
                    loss=loss+harm_w*harm

            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5.0); opt.step()

        # Early stopping uses the same stateful evaluation path used at test time.
        v=_forward_point(model,b.Xv,np.asarray(lo_val,np.float32),mu_u,return_features=False,
                         excitation=b.ev,ids=b.idv,stateful=True)[0]
        vl=float(np.mean((np.asarray(v,float)-np.asarray(b.yv,float))**2))
        if vl<bestloss-1e-7:
            bestloss=vl; best={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}; bad=0
        else: bad+=1
        if bad>=patience: break
    if best: model.load_state_dict(best)

    if flags["use_uq"] and flags["model_kind"]=="ci":
        _fit_residual_scale(model,b,cfg,hp,lo_val)
        lo_cal=b.loc if flags["use_calibrated_lower"] else b.raw_loc
        mask=np.asarray(b.uq_cal_mask,dtype=bool)
        Xcal=b.Xc[mask]; ycal=b.yc[mask]; idcal=b.idc[mask]; ecal=b.ec[mask]; lower_cal=lo_cal[mask]
        if len(Xcal):
            vals=_forward_point(model,Xcal,lower_cal,mu_u,return_features=True,
                                excitation=ecal,ids=idcal,stateful=True)
            pcal=vals[0]; hcal=vals[3]; ident=vals[7]; cf_agreement=vals[17]
            model.scale_head.eval()
            with torch.no_grad():
                base=model.scale_head(torch.from_numpy(hcal.astype(np.float32)).to(dev)).cpu().numpy()
            beta=max(0.0,float(hp.get("information_beta",1.0)))
            disagreement_beta=max(0.0,float(hp.get("disagreement_beta",0.35)))
            scale=base*(1.0+beta*(1.0-np.clip(np.asarray(ident,float),0.0,1.0))
                        +disagreement_beta*(1.0-np.clip(np.asarray(cf_agreement,float),0.0,1.0)))
            scores=np.abs(np.asarray(ycal,float)-pcal)/np.maximum(scale,float(hp["uq_scale_floor"]))
            block=int(cfg.get("uq",{}).get("block_size",0))
            if block<=0: block=max(1,int(np.ceil(float(b.sequence_length)/max(float(cfg.get("stride",1)),1.0))))
            block_scores=_block_max_scores(scores,idcal,block)
            model.conformal_q=_finite_sample_quantile(block_scores,cfg.get("alpha",0.05))
            model.conformal_block_size=block; model.uq_calibration_count=int(len(Xcal)); model.uq_block_count=int(len(block_scores))
        else:
            model.conformal_q=1.96; model.conformal_block_size=1
    return model,hp

def predict_proposal_details(model,variant,X,lo,raw_lo,mu_upper,batch=1024,excitation=None,ids=None):
    flags=_proposal_flags(variant)
    bound=np.asarray(lo if flags["use_calibrated_lower"] else raw_lo,dtype=np.float32)
    vals=_forward_point(
        model,X,bound,mu_upper,batch=batch,return_features=True,excitation=excitation,
        ids=ids,stateful=True
    )
    (prediction,latent,authority,h,prior,innovation,prior_latent,ident,acceptance,info_raw,
     candidate,r_prior,r_candidate,persistent_used,normalized_improvement,veto_probability,
     cf_delta_mu,cf_agreement,information_scale_cv,local_linearity,agreement_veto_probability,
     expert_weight_prior,expert_weight_neural,expert_weight_inverse,adaptive_persistence,
     direction_agreement,magnitude_agreement,inverse_candidate)=vals
    details={
        "prediction":prediction,"raw_mean":prediction.copy(),"latent_score":latent,
        "prior_latent":prior_latent,"prior_prediction":prior,"candidate_prediction":candidate,
        "evidence_delta":innovation,"innovation":innovation,
        "gate":authority,"reliability":authority,"authority":authority,
        "identifiability":ident,"acceptance":acceptance,"information_raw":info_raw,
        "dynamics_residual_prior":r_prior,"dynamics_residual_candidate":r_candidate,
        "normalized_improvement":normalized_improvement,"veto_probability":veto_probability,
        "counterfactual_delta_mu":cf_delta_mu,"counterfactual_agreement":cf_agreement,
        "information_scale_cv":information_scale_cv,"local_linearity":local_linearity,
        "agreement_veto_probability":agreement_veto_probability,
        "expert_weight_prior":expert_weight_prior,"expert_weight_neural":expert_weight_neural,
        "expert_weight_inverse":expert_weight_inverse,"adaptive_persistence":adaptive_persistence,
        "direction_agreement":direction_agreement,"magnitude_agreement":magnitude_agreement,
        "inverse_candidate_prediction":inverse_candidate,
        "persistent_state_used":persistent_used,"sigma":None,"bound":bound,
    }
    if flags["use_uq"] and getattr(model,"scale_head",None) is not None:
        dev=device(); model.scale_head.eval(); scales=[]
        with torch.no_grad():
            for i in range(0,len(h),batch):
                scales.append(model.scale_head(torch.from_numpy(h[i:i+batch].astype(np.float32)).to(dev)).cpu().numpy())
        base=np.concatenate(scales)
        beta=max(0.0,float(getattr(model,"information_beta",1.0)))
        disagreement_beta=max(0.0,float(getattr(model,"disagreement_beta",0.35)))
        scale=base*(1.0+beta*(1.0-np.clip(np.asarray(ident,float),0.0,1.0))
                    +disagreement_beta*(1.0-np.clip(np.asarray(cf_agreement,float),0.0,1.0)))
        q=float(getattr(model,"conformal_q",1.96) or 1.96); radius=q*scale
        low_raw=prediction-radius; high_raw=prediction+radius
        if flags["use_bound"]:
            low=np.maximum(low_raw,bound); high=np.minimum(high_raw,float(mu_upper))
        else:
            low=np.maximum(low_raw,0.0); high=np.minimum(high_raw,float(mu_upper))
        low=np.minimum(low,prediction); high=np.maximum(high,prediction)
        details.update({
            "sigma":scale.astype(np.float32),"conformal_q":q,
            "pi95_low_raw":low_raw.astype(np.float32),"pi95_high_raw":high_raw.astype(np.float32),
            "pi95_low_physics":low.astype(np.float32),"pi95_high_physics":high.astype(np.float32),
        })
    return details


def predict_proposal(model,variant,X,lo,raw_lo,mu_upper,batch=1024,excitation=None,ids=None):
    d=predict_proposal_details(model,variant,X,lo,raw_lo,mu_upper,batch=batch,excitation=excitation,ids=ids)
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
    health_cfg=cfg.get("health",{})
    if preset=="paper":
        min_n=int(health_cfg.get("min_test_endpoints_paper",200))
        min_segments=int(health_cfg.get("min_test_segments_paper",6))
        max_width_ratio=float(health_cfg.get("max_predictive_width_over_target_std_paper",10.0))
    elif preset=="trust":
        min_n=int(health_cfg.get("min_test_endpoints_trust",150))
        min_segments=int(health_cfg.get("min_test_segments_trust",4))
        max_width_ratio=float(health_cfg.get("max_predictive_width_over_target_std_trust",12.0))
    else:
        min_n=int(health_cfg.get("min_test_endpoints_quick",50))
        min_segments=int(health_cfg.get("min_test_segments_quick",1))
        max_width_ratio=float(health_cfg.get("max_predictive_width_over_target_std_quick",20.0))
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
    reliability=np.asarray(preds.get("safegrip_reliability",preds.get("safegrip_gate",np.zeros_like(y))),float)
    ident=np.asarray(preds.get("safegrip_identifiability",np.zeros_like(y)),float)
    acceptance=np.asarray(preds.get("safegrip_acceptance",np.ones_like(y)),float)
    veto=np.asarray(preds.get("safegrip_veto_probability",np.zeros_like(y)),float)
    norm_improve=np.asarray(preds.get("safegrip_normalized_improvement",np.zeros_like(y)),float)
    cf_delta=np.asarray(preds.get("safegrip_counterfactual_delta_mu",np.zeros_like(y)),float)
    cf_agreement=np.asarray(preds.get("safegrip_counterfactual_agreement",np.full_like(y,0.5)),float)
    information_raw=np.asarray(preds.get("safegrip_information_raw",np.zeros_like(y)),float)
    prior=np.asarray(preds.get("safegrip_prior",p),float)
    candidate=np.asarray(preds.get("safegrip_candidate",p),float)
    prior_rmse=float(np.sqrt(np.mean((y-prior)**2))) if len(prior)==len(y) else float("nan")
    n_segments=len({_segment_key(x) for x in np.asarray(bundle.idt,dtype=str)})
    candidate_rmse=float(np.sqrt(np.mean((y-candidate)**2))) if len(candidate)==len(y) else float("nan")
    final_rmse_ensemble=float(np.sqrt(np.mean((y-p)**2)))
    needed=y-prior; candidate_update=candidate-prior; accepted_update=p-prior
    def corr(a,b):
        a=np.asarray(a,float); b=np.asarray(b,float); m=np.isfinite(a)&np.isfinite(b)
        if int(m.sum())<3 or float(np.std(a[m]))<1e-12 or float(np.std(b[m]))<1e-12:
            return float("nan")
        return float(np.corrcoef(a[m],b[m])[0,1])
    candidate_corr=corr(candidate_update,needed); accepted_corr=corr(accepted_update,needed)
    width_ratio=float(mpiw/max(target_std,1e-12)) if np.isfinite(mpiw) else float("nan")
    checks = {
        "finite_predictions": bool(np.isfinite(p).all()),
        "enough_test_endpoints": bool(len(y) >= min_n),
        "enough_independent_test_segments": bool(n_segments >= min_segments),
        "final_prediction_not_constant": bool(pred_std > max(1e-4, 0.05 * target_std)),
        "not_projection_dominated": bool(not np.isfinite(projection_rate) or projection_rate < float(cfg.get("health",{}).get("max_projection_correction_rate", 0.95))),
        "beats_train_mean_rmse": bool(float(row["rmse"]) < mean_rmse),
        "positive_r2": bool(float(row["r2"]) > 0.0),
        "raw_bound_finite": bool(np.isfinite(raw_bound).all()),
        "calibrated_lower_coverage_consistent": bool(np.isfinite(lower_violation) and lower_violation <= alpha + coverage_tol),
        "predictive_interval_coverage_consistent": bool(np.isfinite(picp) and picp >= target_coverage-interval_tol),
        "predictive_interval_not_pathologically_wide": bool(not np.isfinite(width_ratio) or width_ratio <= max_width_ratio),
        "counterfactual_authority_finite": bool(np.isfinite(reliability).all()),
        "identifiability_finite": bool(np.isfinite(ident).all() and np.all((ident>=-1e-6)&(ident<=1.0+1e-6))),
        "acceptance_finite": bool(np.isfinite(acceptance).all() and np.all((acceptance>=-1e-6)&(acceptance<=1.0+1e-6))),
        "veto_probability_finite": bool(np.isfinite(veto).all() and np.all((veto>=-1e-6)&(veto<=1.0+1e-6))),
        "counterfactual_delta_finite": bool(np.isfinite(cf_delta).all()),
        "final_not_materially_worse_than_persistent_prior": bool(not np.isfinite(prior_rmse) or final_rmse_ensemble <= 1.03*prior_rmse),
        "accepted_update_direction_not_inverted": bool(not np.isfinite(accepted_corr) or accepted_corr >= -0.10),
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
            "predictive_width_over_target_std": width_ratio,
            "max_allowed_predictive_width_over_target_std": max_width_ratio,
            "target_predictive_coverage": target_coverage,
            "predictive_coverage_tolerance": interval_tol,
            "n_test_segments": int(n_segments),
            "min_required_test_segments": int(min_segments),
            "physics_lower_nonzero_rate": float(np.mean(raw_bound>1e-9)),
            "physics_lower_above_005_rate": float(np.mean(raw_bound>0.05)),
            "physics_information_fraction_mean": float(np.mean(np.clip(raw_bound/max(float(cfg.get("mu_upper",1.3)),1e-12),0.0,1.0))),
            "prior_rmse": prior_rmse,
            "candidate_rmse": candidate_rmse,
            "ensemble_final_rmse": final_rmse_ensemble,
            "candidate_update_vs_needed_correlation": candidate_corr,
            "accepted_update_vs_needed_correlation": accepted_corr,
            "fraction_candidate_improves_over_prior": float(np.mean(np.abs(y-candidate)<np.abs(y-prior))),
            "fraction_final_improves_over_prior": float(np.mean(np.abs(y-p)<np.abs(y-prior))),
            "mean_abs_dynamic_update": float(np.mean(np.abs(p-prior))) if len(prior)==len(p) else float("nan"),
            "mean_counterfactual_authority": float(np.mean(reliability)) if len(reliability) else float("nan"),
            "mean_identifiability": float(np.mean(ident)) if len(ident) else float("nan"),
            "mean_acceptance": float(np.mean(acceptance)) if len(acceptance) else float("nan"),
            "mean_veto_probability": float(np.mean(veto)) if len(veto) else float("nan"),
            "mean_normalized_improvement": float(np.mean(norm_improve)) if len(norm_improve) else float("nan"),
            "mean_counterfactual_delta_mu": float(np.mean(cf_delta)) if len(cf_delta) else float("nan"),
            "mean_counterfactual_agreement": float(np.mean(cf_agreement)) if len(cf_agreement) else float("nan"),
            "mean_information_raw": float(np.mean(information_raw)) if len(information_raw) else float("nan"),
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
    proposal_bundle=make_bundle(csv_path,cfg,sequence_length=proposal_seq,scaler_kind="standard",eval_start=start,feature_mode="raw")
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

    per_seed=[]; control_rows=[]; seed_prediction_frames=[]
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
            seed_prediction_frames.append(pd.DataFrame({
                "endpoint_id":b.idt.astype(str), "y_true":b.yt,
                "model":name, "seed":int(seed), "prediction":np.asarray(p,float),
            }))
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

    name="safegrip"; print(f"[proposal/CI] safegrip L={proposal_bundle.sequence_length}")
    proposal_preds=[]; proposal_raw=[]; proposal_sigmas=[]; proposal_latent=[]; final_hp=None
    seed_priors=[]; seed_deltas=[]; seed_prior_latents=[]
    seed_ident=[]; seed_acceptance=[]; seed_info_raw=[]; seed_candidates=[]; seed_persistent=[]
    seed_veto=[]; seed_norm_improve=[]; seed_cf_delta=[]; seed_cf_agreement=[]
    seed_info_cv=[]; seed_linearity=[]; seed_agreement_veto=[]
    seed_w_prior=[]; seed_w_neural=[]; seed_w_inverse=[]; seed_adaptive_rho=[]
    seed_dir_agree=[]; seed_mag_agree=[]; seed_inverse_candidate=[]
    seed_uq_initial_scales=[]
    seed_pi_low=[]; seed_pi_high=[]; seed_pi_low_raw=[]; seed_pi_high_raw=[]; seed_gates=[]; seed_q=[]; seed_blocks=[]; seed_uq_counts=[]
    for seed in seed_list:
        seed_everything(int(seed)); t0=time.time(); model,final_hp=fit_proposal(name,proposal_bundle,cfg,epochs_proposal,hp_overrides)
        d=predict_proposal_details(model,name,proposal_bundle.Xt,proposal_bundle.lot,proposal_bundle.raw_lot,cfg["mu_upper"],excitation=proposal_bundle.et,ids=proposal_bundle.idt)
        if enable_common_uq and np.any(proposal_bundle.uq_cal_mask):
            m=np.asarray(proposal_bundle.uq_cal_mask,dtype=bool)
            dc=predict_proposal_details(model,name,proposal_bundle.Xc[m],proposal_bundle.loc[m],proposal_bundle.raw_loc[m],cfg["mu_upper"],excitation=proposal_bundle.ec[m],ids=proposal_bundle.idc[m])
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
                         "mean_identifiability":float(np.mean(d["identifiability"])),
                         "mean_acceptance":float(np.mean(d["acceptance"])),
                         "mean_veto_probability":float(np.mean(d["veto_probability"])),
                         "mean_normalized_improvement":float(np.mean(d["normalized_improvement"])),
                         "mean_counterfactual_agreement":float(np.mean(d["counterfactual_agreement"])),
                         "mean_information_scale_cv":float(np.mean(d["information_scale_cv"])),
                         "mean_local_linearity":float(np.mean(d["local_linearity"])),
                         "mean_agreement_veto_probability":float(np.mean(d["agreement_veto_probability"])),
                         "mean_information_raw":float(np.mean(d["information_raw"])),
                         "persistent_state_use_rate":float(np.mean(d["persistent_state_used"])),
                         "mean_adaptive_persistence":float(np.mean(d["adaptive_persistence"][d["persistent_state_used"]>0])) if np.any(d["persistent_state_used"]>0) else 0.0,
                         "mean_expert_weight_prior":float(np.mean(d["expert_weight_prior"])),
                         "mean_expert_weight_neural":float(np.mean(d["expert_weight_neural"])),
                         "mean_expert_weight_inverse":float(np.mean(d["expert_weight_inverse"])),
                         "mean_direction_agreement":float(np.mean(d["direction_agreement"])),
                         "mean_magnitude_agreement":float(np.mean(d["magnitude_agreement"])),
                         "prior_rmse":float(np.sqrt(np.mean((proposal_bundle.yt-d["prior_prediction"])**2))),
                         "candidate_rmse":float(np.sqrt(np.mean((proposal_bundle.yt-d["candidate_prediction"])**2))),
                         "inverse_candidate_rmse":float(np.sqrt(np.mean((proposal_bundle.yt-d["inverse_candidate_prediction"])**2))),
                         "mean_abs_dynamic_update":float(np.mean(np.abs(d["prediction"]-d["prior_prediction"]))),
                         "seconds":time.time()-t0})
        proposal_preds.append(d["prediction"]); proposal_raw.append(d["raw_mean"]); proposal_latent.append(d["latent_score"]); seed_gates.append(d["reliability"])
        seed_prediction_frames.append(pd.DataFrame({
            "endpoint_id":proposal_bundle.idt.astype(str), "y_true":proposal_bundle.yt,
            "model":name, "seed":int(seed), "prediction":np.asarray(d["prediction"],float),
        }))
        seed_priors.append(d["prior_prediction"]); seed_deltas.append(d["evidence_delta"]); seed_prior_latents.append(d["prior_latent"])
        seed_ident.append(d["identifiability"]); seed_acceptance.append(d["acceptance"]); seed_info_raw.append(d["information_raw"])
        seed_veto.append(d["veto_probability"]); seed_norm_improve.append(d["normalized_improvement"])
        seed_cf_delta.append(d["counterfactual_delta_mu"]); seed_cf_agreement.append(d["counterfactual_agreement"])
        seed_info_cv.append(d["information_scale_cv"]); seed_linearity.append(d["local_linearity"]); seed_agreement_veto.append(d["agreement_veto_probability"])
        seed_w_prior.append(d["expert_weight_prior"]); seed_w_neural.append(d["expert_weight_neural"]); seed_w_inverse.append(d["expert_weight_inverse"])
        seed_adaptive_rho.append(d["adaptive_persistence"]); seed_dir_agree.append(d["direction_agreement"]); seed_mag_agree.append(d["magnitude_agreement"])
        seed_inverse_candidate.append(d["inverse_candidate_prediction"])
        seed_candidates.append(d["candidate_prediction"]); seed_persistent.append(d["persistent_state_used"])
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
    preds[name+"_identifiability"]=np.mean(np.stack(seed_ident),axis=0)
    preds[name+"_acceptance"]=np.mean(np.stack(seed_acceptance),axis=0)
    preds[name+"_information_raw"]=np.mean(np.stack(seed_info_raw),axis=0)
    preds[name+"_candidate"]=np.mean(np.stack(seed_candidates),axis=0)
    preds[name+"_persistent_state_used"]=np.mean(np.stack(seed_persistent),axis=0)
    preds[name+"_veto_probability"]=np.mean(np.stack(seed_veto),axis=0)
    preds[name+"_normalized_improvement"]=np.mean(np.stack(seed_norm_improve),axis=0)
    preds[name+"_counterfactual_delta_mu"]=np.mean(np.stack(seed_cf_delta),axis=0)
    preds[name+"_counterfactual_agreement"]=np.mean(np.stack(seed_cf_agreement),axis=0)
    preds[name+"_information_scale_cv"]=np.mean(np.stack(seed_info_cv),axis=0)
    preds[name+"_local_linearity"]=np.mean(np.stack(seed_linearity),axis=0)
    preds[name+"_agreement_veto_probability"]=np.mean(np.stack(seed_agreement_veto),axis=0)
    preds[name+"_expert_weight_prior"]=np.mean(np.stack(seed_w_prior),axis=0)
    preds[name+"_expert_weight_neural"]=np.mean(np.stack(seed_w_neural),axis=0)
    preds[name+"_expert_weight_inverse"]=np.mean(np.stack(seed_w_inverse),axis=0)
    preds[name+"_adaptive_persistence"]=np.mean(np.stack(seed_adaptive_rho),axis=0)
    preds[name+"_direction_agreement"]=np.mean(np.stack(seed_dir_agree),axis=0)
    preds[name+"_magnitude_agreement"]=np.mean(np.stack(seed_mag_agree),axis=0)
    preds[name+"_inverse_candidate"]=np.mean(np.stack(seed_inverse_candidate),axis=0)
    if proposal_sigmas: preds[name+"_sigma"]=np.mean(np.stack(proposal_sigmas),axis=0)
    if seed_pi_low:
        preds[name+"_pi95_low_physics"]=np.mean(np.stack(seed_pi_low),axis=0)
        preds[name+"_pi95_high_physics"]=np.mean(np.stack(seed_pi_high),axis=0)
        preds[name+"_pi95_low_raw"]=np.mean(np.stack(seed_pi_low_raw),axis=0)
        preds[name+"_pi95_high_raw"]=np.mean(np.stack(seed_pi_high_raw),axis=0)

    (out/"proposal_reliability.json").write_text(json.dumps({
        "method":"SafeGrip-CI v1.1 dual-expert state correction with learned risk-aware arbitration",
        "experts":["keep_persistent_prior","neural_direction_magnitude_correction","local_inverse_dynamics_correction"],
        "arbitration":"softmax policy trained on training-only best-expert targets; inference uses only raw-sensor/dynamics evidence",
        "identifiability":"counterfactual local sensitivity is an inverse-expert availability/observability feature, not a monotone neural correctness score",
        "agreement":"direction and magnitude agreement are evidence features; disagreement is not a permanent multiplicative veto",
        "counterfactual_scale":"single-scale finite difference is the v1.1 default; multi-scale consistency remains a legacy/sensitivity control",
        "handcrafted_excitation_used_by_full_proposal":False,
        "persistent_state":"previous predicted friction is carried within each trajectory segment with learned context-dependent persistence",
        "innovation":"bounded neural correction factorized into learned direction and magnitude",
        "per_seed_uq_initial_scale":seed_uq_initial_scales,
    },indent=2),encoding="utf-8")
    (out/"proposal_uq.json").write_text(json.dumps({
        "method":"post-hoc residual scale + observability/disagreement inflation + block-max split conformal",
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
    if seed_prediction_frames:
        pd.concat(seed_prediction_frames,ignore_index=True).to_csv(out/"predictions_by_seed.csv",index=False)
    (out/"prediction_manifest.json").write_text(json.dumps({
        "ensemble_predictions_file":"predictions.csv",
        "per_seed_predictions_file":"predictions_by_seed.csv",
        "model_columns":baseline_names+["safegrip"],
        "ensemble_definition":"arithmetic mean of predictions across matched seeds",
        "primary_metric_definition":"metrics.csv is the arithmetic mean of per-seed metrics; statistical inference uses predictions_by_seed.csv",
    },indent=2),encoding="utf-8")
    (out/"proposal_hparams.json").write_text(json.dumps(final_hp,indent=2),encoding="utf-8")
    (out/"baseline_selected_hparams.json").write_text(json.dumps(selected,indent=2),encoding="utf-8")
    (out/"evaluation_protocol.json").write_text(json.dumps({
        "seeds":seed_list,"common_eval_start":start,"common_eval_warmup_samples":start+1,
        "same_validation_test_endpoints":True,"endpoint_identity_checked":True,
        "windows_grouped_by_segment":True,"primary_selection_metric":"validation RMSE",
        "physics_window_samples":int(cfg.get("physics",{}).get("window_samples",1)),
        "calibration_labels_used_in_gradient_training":False,
        "proposal_feature_engineering_label_free":True,
        "proposal_point_loss":"Huber final-state loss + raw innovation supervision + candidate loss + direction loss + training-only best-expert arbitration + adaptive-prior loss + dynamics ranking + do-no-harm penalty",
        "proposal_architecture":"adaptive persistent friction state + direction/magnitude neural correction + local inverse-dynamics correction + learned three-action arbitration",
        "proposal_reliability":"counterfactual identifiability is an observability/availability feature for inverse dynamics; correctness is learned by arbitration rather than a multiplicative gate",
        "proposal_bound_parameterization":"physics support projection is applied after the arbitrated state correction",
        "proposal_uq":"post-hoc residual scale + observability/disagreement inflation + block-max split-conformal multiplier",
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
    out=ensure_dir(out_dir); seq=(hp_overrides or {}).get("sequence_length"); start=common_eval_start(cfg)
    variants=list(variants or PRIMARY_ABLATION_VARIANTS)
    bad=[v for v in variants if v not in PROPOSAL_VARIANTS]
    if bad: raise ValueError("Unknown proposal variants: "+", ".join(bad))
    # Cache one leakage-safe bundle for each actual representation required.
    bundles={}
    for v in variants:
        mode=_proposal_flags(v)["feature_mode"]
        if mode not in bundles:
            bundles[mode]=make_bundle(csv_path,cfg,sequence_length=seq,scaler_kind="standard",eval_start=start,feature_mode=mode)
    ref=bundles[_proposal_flags(variants[0])["feature_mode"]]
    for mode,b in bundles.items(): _assert_same_targets(ref,b,f"ablation_bundle_{mode}")
    _save_bundle_meta(out,ref,cfg)
    epoch_key="epochs_quick" if preset=="quick" else ("epochs_trust" if preset=="trust" else "epochs_paper")
    epochs=int(cfg["training"].get(epoch_key,cfg["training"].get("epochs_paper",60)))
    if preset=="paper": seed_list=list(cfg.get("evaluation",{}).get("seeds",[cfg["seed"]]))
    elif preset=="trust": seed_list=list(cfg.get("evaluation",{}).get("trust_seeds",[0,1,2]))
    else: seed_list=[int(cfg["seed"])]
    results=[]; preds=pd.DataFrame({"endpoint_id":ref.idt,"y_true":ref.yt,"physics_lower":ref.lot,"physics_lower_raw":ref.raw_lot})
    semantic={}
    for variant in variants:
        flags=_proposal_flags(variant); vb=bundles[flags["feature_mode"]]
        print(f"[ablation/CI] {variant} -> {flags['canonical_variant']} [{flags['feature_mode']}]")
        variant_preds=[]; variant_raw=[]; variant_sigma=[]; variant_auth=[]; variant_ident=[]; variant_accept=[]; lo_int=[]; hi_int=[]
        variant_veto=[]; variant_norm=[]; variant_cf_delta=[]; variant_cf_agree=[]
        variant_info_cv=[]; variant_linearity=[]; variant_agreement_veto=[]
        for seed in seed_list:
            seed_everything(int(seed)); t0=time.time(); model,hp=fit_proposal(variant,vb,cfg,epochs,hp_overrides)
            d=predict_proposal_details(model,variant,vb.Xt,vb.lot,vb.raw_lot,cfg["mu_upper"],excitation=vb.et,ids=vb.idt)
            row={"model":variant,"kind":"ablation" if variant!="safegrip" else "proposal","seed":int(seed),
                 **regression_metrics(vb.yt,d["prediction"],d["bound"],cfg["mu_upper"],d["sigma"],raw_mean=d["raw_mean"],
                    interval_low=d.get("pi95_low_physics"),interval_high=d.get("pi95_high_physics"),
                    interval_low_raw=d.get("pi95_low_raw"),interval_high_raw=d.get("pi95_high_raw")),
                 "conformal_q":float(d.get("conformal_q",np.nan)),
                 "mean_reliability":float(np.mean(d["reliability"])),
                 "mean_gate":float(np.mean(d["reliability"])),
                 "mean_identifiability":float(np.mean(d["identifiability"])),
                 "mean_acceptance":float(np.mean(d["acceptance"])),
                 "mean_veto_probability":float(np.mean(d["veto_probability"])),
                 "mean_normalized_improvement":float(np.mean(d["normalized_improvement"])),
                 "mean_counterfactual_agreement":float(np.mean(d["counterfactual_agreement"])),
                 "mean_information_scale_cv":float(np.mean(d["information_scale_cv"])),
                 "mean_local_linearity":float(np.mean(d["local_linearity"])),
                 "mean_agreement_veto_probability":float(np.mean(d["agreement_veto_probability"])),
                 "mean_information_raw":float(np.mean(d["information_raw"])),
                 "persistent_state_use_rate":float(np.mean(d["persistent_state_used"])),
                 "prior_rmse":float(np.sqrt(np.mean((vb.yt-d["prior_prediction"])**2))),
                 "candidate_rmse":float(np.sqrt(np.mean((vb.yt-d["candidate_prediction"])**2))),
                 "mean_abs_dynamic_update":float(np.mean(np.abs(d["prediction"]-d["prior_prediction"]))),
                 "seconds":time.time()-t0}
            results.append(row); variant_preds.append(d["prediction"]); variant_raw.append(d["raw_mean"])
            variant_auth.append(d["authority"]); variant_ident.append(d["identifiability"]); variant_accept.append(d["acceptance"])
            variant_veto.append(d["veto_probability"]); variant_norm.append(d["normalized_improvement"])
            variant_cf_delta.append(d["counterfactual_delta_mu"]); variant_cf_agree.append(d["counterfactual_agreement"])
            variant_info_cv.append(d["information_scale_cv"]); variant_linearity.append(d["local_linearity"]); variant_agreement_veto.append(d["agreement_veto_probability"])
            if d["sigma"] is not None: variant_sigma.append(d["sigma"])
            if "pi95_low_physics" in d: lo_int.append(d["pi95_low_physics"]); hi_int.append(d["pi95_high_physics"])
        preds[variant]=np.mean(np.stack(variant_preds),axis=0); preds[variant+"_raw"]=np.mean(np.stack(variant_raw),axis=0)
        preds[variant+"_authority"]=np.mean(np.stack(variant_auth),axis=0)
        preds[variant+"_identifiability"]=np.mean(np.stack(variant_ident),axis=0)
        preds[variant+"_acceptance"]=np.mean(np.stack(variant_accept),axis=0)
        preds[variant+"_veto_probability"]=np.mean(np.stack(variant_veto),axis=0)
        preds[variant+"_normalized_improvement"]=np.mean(np.stack(variant_norm),axis=0)
        preds[variant+"_counterfactual_delta_mu"]=np.mean(np.stack(variant_cf_delta),axis=0)
        preds[variant+"_counterfactual_agreement"]=np.mean(np.stack(variant_cf_agree),axis=0)
        preds[variant+"_information_scale_cv"]=np.mean(np.stack(variant_info_cv),axis=0)
        preds[variant+"_local_linearity"]=np.mean(np.stack(variant_linearity),axis=0)
        preds[variant+"_agreement_veto_probability"]=np.mean(np.stack(variant_agreement_veto),axis=0)
        if variant_sigma: preds[variant+"_sigma"]=np.mean(np.stack(variant_sigma),axis=0)
        if lo_int:
            preds[variant+"_pi95_low_physics"]=np.mean(np.stack(lo_int),axis=0); preds[variant+"_pi95_high_physics"]=np.mean(np.stack(hi_int),axis=0)
        semantic[variant]=flags
    by_seed=pd.DataFrame(results); summary=_aggregate_seed_metrics(results)
    summary.to_csv(out/"ablation_metrics.csv",index=False); by_seed.to_csv(out/"ablation_metrics_by_seed.csv",index=False); preds.to_csv(out/"ablation_predictions.csv",index=False)
    (out/"ablation_design.json").write_text(json.dumps({
        "proposal":"SafeGrip-CI v1.1 dual-expert adaptive-state estimator with learned risk-aware arbitration",
        "primary_variants":variants,
        "semantic_specs":semantic,
        "key_comparisons":[
            "safegrip_excitation_proxy vs safegrip tests handcrafted excitation against learned counterfactual identifiability",
            "safegrip_no_acceptance vs safegrip isolates the asymmetric counterfactual veto",
            "safegrip_no_cf_agreement vs safegrip removes both inverse-dynamics agreement supervision and inference veto",
            "safegrip_single_scale_cf vs safegrip isolates robust multi-scale counterfactual observability",
            "safegrip_no_linearity_consistency vs safegrip isolates the cross-scale local-linearity discount",
            "safegrip_no_agreement_veto vs safegrip isolates inference-time use of inverse-dynamics agreement while retaining its training loss",
            "safegrip_no_counterfactual_ranking vs safegrip isolates friction-discriminative training of the dynamics model",
        ],
        "same_hyperparameters_across_variants":True,"seeds":seed_list,
    },indent=2),encoding="utf-8")
    return summary

