from __future__ import annotations
import json, time
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from joblib import dump

from .literature import LITERATURE_BASELINES, LITERATURE_ONLY, PAPER_BASELINES, QUICK_BASELINES, validate_paper_baselines
from .models import make_literature_baseline, SafeGripNet, SafeGripNoTemporal, DeterministicTCN
from .physics import project_torch, conformal_lower_correction, apply_lower_correction
from .utils import ensure_dir, seed_everything, device

META={"distance","lat","lon","match_distance_m","mu_ref","physics_lower_raw","split","source_file"}
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
    scaler: StandardScaler
    q: float
    Xtr: np.ndarray; ytr: np.ndarray; lotr: np.ndarray; raw_lotr: np.ndarray
    Xc: np.ndarray; yc: np.ndarray; loc: np.ndarray; raw_loc: np.ndarray
    Xv: np.ndarray; yv: np.ndarray; lov: np.ndarray; raw_lov: np.ndarray
    Xt: np.ndarray; yt: np.ndarray; lot: np.ndarray; raw_lot: np.ndarray


def windows_for_split(df,features,split,L,stride):
    z=df[df.split==split].reset_index(drop=True)
    X=z[features].to_numpy(np.float32); y=z.mu_ref.to_numpy(np.float32); lo=z.physics_lower_raw.to_numpy(np.float32)
    xs=[]; ys=[]; ls=[]
    for end in range(L-1,len(z),stride):
        xs.append(X[end-L+1:end+1]); ys.append(y[end]); ls.append(lo[end])
    if not xs:
        return np.empty((0,L,len(features)),np.float32),np.empty(0,np.float32),np.empty(0,np.float32)
    return np.stack(xs),np.asarray(ys,np.float32),np.asarray(ls,np.float32)


def make_bundle(csv_path, cfg, sequence_length=None):
    df=pd.read_csv(csv_path)
    features=[c for c in df.columns if c not in META and pd.api.types.is_numeric_dtype(df[c])]
    features=[c for c in features if c not in ("mu_ref","physics_lower_raw")]
    L=int(sequence_length or cfg["sequence_length"]); stride=int(cfg["stride"])
    splits={s:windows_for_split(df,features,s,L,stride) for s in ("train","calibration","validation","test")}
    Xtr,ytr,raw_lotr=splits["train"]; Xc,yc,raw_loc=splits["calibration"]
    Xv,yv,raw_lov=splits["validation"]; Xt,yt,raw_lot=splits["test"]
    if min(len(Xtr),len(Xv),len(Xt))==0:
        raise RuntimeError("One split has no windows; lower sequence_length or inspect prepared data")
    scaler=StandardScaler().fit(Xtr.reshape(-1,len(features)))
    def sc(X): return scaler.transform(X.reshape(-1,len(features))).reshape(X.shape).astype(np.float32)
    Xtr,Xc,Xv,Xt=map(sc,(Xtr,Xc,Xv,Xt))
    q=conformal_lower_correction(raw_loc,yc,cfg["alpha"]) if len(yc) else 0.0
    lotr=apply_lower_correction(raw_lotr,q).astype(np.float32)
    loc=apply_lower_correction(raw_loc,q).astype(np.float32)
    lov=apply_lower_correction(raw_lov,q).astype(np.float32)
    lot=apply_lower_correction(raw_lot,q).astype(np.float32)
    return Bundle(features,scaler,q,Xtr,ytr,lotr,raw_lotr,Xc,yc,loc,raw_loc,Xv,yv,lov,raw_lov,Xt,yt,lot,raw_lot)


def regression_metrics(y,p,lo=None,mu_upper=1.3,sigma=None):
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
        })
    if sigma is not None:
        sigma=np.maximum(np.asarray(sigma),1e-5); low=p-1.96*sigma; high=p+1.96*sigma
        out.update({
            "picp95":float(np.mean((y>=low)&(y<=high))),
            "mpiw95":float(np.mean(high-low)),
            "gaussian_nll":float(np.mean(.5*((y-p)/sigma)**2+np.log(sigma))),
        })
    return out


def _deep_loader(X,y,lo,batch):
    return DataLoader(TensorDataset(torch.from_numpy(X),torch.from_numpy(y),torch.from_numpy(lo)),batch_size=batch,shuffle=True)


def _fit_deterministic(model,Xtr,ytr,Xv,yv,cfg,epochs):
    dev=device(); model=model.to(dev); tr=cfg["training"]
    opt=torch.optim.AdamW(model.parameters(),lr=tr["lr"],weight_decay=tr["weight_decay"])
    dl=DataLoader(TensorDataset(torch.from_numpy(Xtr),torch.from_numpy(ytr)),batch_size=tr["batch_size"],shuffle=True)
    xv=torch.from_numpy(Xv).to(dev); yv_t=torch.from_numpy(yv).to(dev)
    best=None; bestloss=float("inf"); bad=0
    for _ in range(epochs):
        model.train()
        for xb,yb in dl:
            xb=xb.to(dev); yb=yb.to(dev); opt.zero_grad(); pred=model(xb)
            loss=nn.functional.mse_loss(pred,yb); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5.0); opt.step()
        model.eval()
        with torch.no_grad(): vl=nn.functional.mse_loss(model(xv),yv_t).item()
        if vl<bestloss-1e-7:
            bestloss=vl; best={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}; bad=0
        else: bad+=1
        if bad>=tr["patience"]: break
    if best: model.load_state_dict(best)
    return model


def fit_literature(name,b:Bundle,cfg,epochs,preset="paper"):
    if name=="chen2025_svdkl":
        from .svdkl import fit_svdkl
        tr=cfg["training"]; gp=cfg.get("baseline",{}).get("chen2025_svdkl",{})
        return fit_svdkl(b.Xtr,b.ytr,b.Xv,b.yv,hidden=gp.get("hidden",64),feature_dim=gp.get("feature_dim",16),
                         inducing=gp.get("inducing",128),dropout=tr["dropout"],lr=tr["lr"],
                         weight_decay=tr["weight_decay"],batch_size=tr["batch_size"],epochs=epochs,
                         patience=tr["patience"],device=device())
    model=make_literature_baseline(name,len(b.features),debug_scale=(preset=="quick"),dropout=cfg["training"]["dropout"])
    return _fit_deterministic(model,b.Xtr,b.ytr,b.Xv,b.yv,cfg,epochs)


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
    lo_train=b.raw_lotr if variant=="safegrip_no_calibration" else b.lotr
    lo_val=b.raw_lov if variant=="safegrip_no_calibration" else b.lov
    dl=_deep_loader(b.Xtr,b.ytr,lo_train,int(hp["batch_size"]))
    xv=torch.from_numpy(b.Xv).to(dev); yv=torch.from_numpy(b.yv).to(dev); lov=torch.from_numpy(lo_val).to(dev)
    mu_u=float(cfg["mu_upper"]); best=None; bestloss=float("inf"); bad=0; patience=int(cfg["training"]["patience"])
    for _ in range(epochs):
        model.train()
        for xb,yb,lb in dl:
            xb=xb.to(dev); yb=yb.to(dev); lb=lb.to(dev); ub=torch.full_like(lb,mu_u); opt.zero_grad()
            if variant=="safegrip_no_uq":
                raw=model(xb); pred=project_torch(raw,lb,ub)
                nll=nn.functional.mse_loss(pred,yb)
            else:
                raw,logsig=model(xb)
                use_projection=variant not in ("safegrip_no_projection","safegrip_data_only")
                pred=project_torch(raw,lb,ub) if use_projection else raw
                sig=torch.exp(logsig)
                nll=(.5*((yb-pred)/sig)**2+logsig).mean()
            loss=nll+float(hp["lambda_mse"])*nn.functional.mse_loss(pred,yb)
            if variant not in ("safegrip_no_physics_loss","safegrip_data_only"):
                phys=torch.relu(lb-raw).pow(2).mean()+.25*torch.relu(raw-mu_u).pow(2).mean()
                loss=loss+float(hp["lambda_physics"])*phys
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5.0); opt.step()
        model.eval()
        with torch.no_grad():
            if variant=="safegrip_no_uq": raw=model(xv); pv=project_torch(raw,lov,torch.full_like(lov,mu_u))
            else:
                raw,_=model(xv); pv=raw if variant in ("safegrip_no_projection","safegrip_data_only") else project_torch(raw,lov,torch.full_like(lov,mu_u))
            vl=nn.functional.mse_loss(pv,yv).item()
        if vl<bestloss-1e-7:
            bestloss=vl; best={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}; bad=0
        else: bad+=1
        if bad>=patience: break
    if best: model.load_state_dict(best)
    return model,hp


def predict_proposal(model,variant,X,lo,raw_lo,mu_upper,batch=1024):
    dev=device(); model.eval(); ps=[]; ss=[]
    bound=raw_lo if variant=="safegrip_no_calibration" else lo
    with torch.no_grad():
        for i in range(0,len(X),batch):
            xb=torch.from_numpy(X[i:i+batch]).to(dev); lb=torch.from_numpy(bound[i:i+batch]).to(dev); ub=torch.full_like(lb,float(mu_upper))
            if variant=="safegrip_no_uq": raw=model(xb); p=project_torch(raw,lb,ub)
            else:
                raw,logsig=model(xb); p=raw if variant in ("safegrip_no_projection","safegrip_data_only") else project_torch(raw,lb,ub)
                ss.append(torch.exp(logsig).cpu().numpy())
            ps.append(p.cpu().numpy())
    return np.concatenate(ps),(np.concatenate(ss) if ss else None),bound


def _export_literature_manifest(out:Path,names):
    rows=[]
    for n in names:
        x={"model":n,**LITERATURE_BASELINES[n]}; rows.append(x)
    pd.DataFrame(rows).to_csv(out/"baseline_manifest.csv",index=False)
    (out/"literature_only.json").write_text(json.dumps(LITERATURE_ONLY,indent=2),encoding="utf-8")


def _save_bundle_meta(out,b,cfg):
    dump(b.scaler,out/"scaler.joblib")
    (out/"features.json").write_text(json.dumps(b.features,indent=2),encoding="utf-8")
    (out/"calibration.json").write_text(json.dumps({"one_sided_q":b.q,"alpha":cfg["alpha"],"test_used_for_calibration":False},indent=2),encoding="utf-8")


def run_benchmark(csv_path,out_dir,cfg,preset="quick",models=None,hp_overrides=None):
    seed_everything(cfg["seed"]); out=ensure_dir(out_dir)
    seq=(hp_overrides or {}).get("sequence_length")
    b=make_bundle(csv_path,cfg,sequence_length=seq); _save_bundle_meta(out,b,cfg)
    baseline_names=list(models) if models else list(QUICK_BASELINES if preset=="quick" else PAPER_BASELINES)
    validate_paper_baselines(baseline_names); _export_literature_manifest(out,baseline_names)
    epochs=int(cfg["training"]["epochs_quick" if preset=="quick" else "epochs_paper"])
    results=[]; preds=pd.DataFrame({"y_true":b.yt,"physics_lower":b.lot,"physics_lower_raw":b.raw_lot})
    for name in baseline_names:
        print(f"[baseline/cited] {name} DOI={LITERATURE_BASELINES[name]['doi']}"); t0=time.time()
        model=fit_literature(name,b,cfg,epochs,preset); p,s=predict_literature(model,name,b.Xt)
        row={"model":name,"kind":"published_baseline","doi":LITERATURE_BASELINES[name]["doi"],**regression_metrics(b.yt,p,b.lot,cfg["mu_upper"],s),"seconds":time.time()-t0}
        results.append(row); preds[name]=p
        if s is not None: preds[name+"_sigma"]=s
    # Full proposal is always appended; it is not a baseline.
    name="safegrip"; print("[proposal] safegrip"); t0=time.time(); model,hp=fit_proposal(name,b,cfg,epochs,hp_overrides)
    p,s,bound=predict_proposal(model,name,b.Xt,b.lot,b.raw_lot,cfg["mu_upper"])
    results.append({"model":name,"kind":"proposal","doi":"",**regression_metrics(b.yt,p,bound,cfg["mu_upper"],s),"seconds":time.time()-t0})
    preds[name]=p; preds[name+"_sigma"]=s
    pd.DataFrame(results).to_csv(out/"metrics.csv",index=False); preds.to_csv(out/"predictions.csv",index=False)
    (out/"proposal_hparams.json").write_text(json.dumps(hp,indent=2),encoding="utf-8")
    return pd.DataFrame(results)


def run_ablation(csv_path,out_dir,cfg,preset="paper",variants=None,hp_overrides=None):
    seed_everything(cfg["seed"]); out=ensure_dir(out_dir)
    seq=(hp_overrides or {}).get("sequence_length")
    b=make_bundle(csv_path,cfg,sequence_length=seq); _save_bundle_meta(out,b,cfg)
    variants=list(variants or PROPOSAL_VARIANTS); epochs=int(cfg["training"]["epochs_quick" if preset=="quick" else "epochs_paper"])
    results=[]; preds=pd.DataFrame({"y_true":b.yt,"physics_lower":b.lot,"physics_lower_raw":b.raw_lot})
    for v in variants:
        print(f"[ablation] {v}"); t0=time.time(); model,hp=fit_proposal(v,b,cfg,epochs,hp_overrides)
        p,s,bound=predict_proposal(model,v,b.Xt,b.lot,b.raw_lot,cfg["mu_upper"])
        row={"model":v,"kind":"ablation" if v!="safegrip" else "proposal",**regression_metrics(b.yt,p,bound,cfg["mu_upper"],s),"seconds":time.time()-t0}
        results.append(row); preds[v]=p
        if s is not None: preds[v+"_sigma"]=s
    pd.DataFrame(results).to_csv(out/"ablation_metrics.csv",index=False); preds.to_csv(out/"ablation_predictions.csv",index=False)
    (out/"ablation_design.json").write_text(json.dumps({
        "safegrip_data_only":"remove hard projection and soft physics loss; retain the same TCN + UQ backbone",
        "safegrip_no_projection":"remove hard identified-set projection; keep UQ + soft physics",
        "safegrip_no_uq":"remove heteroscedastic uncertainty head; keep hard projection",
        "safegrip_no_physics_loss":"remove soft physics penalty; keep hard projection + UQ",
        "safegrip_no_calibration":"use raw mechanics lower bound; remove one-sided calibration",
        "safegrip_no_temporal":"replace TCN by last-state MLP; keep projection + UQ",
        "safegrip":"full calibrated physics-guaranteed temporal estimator",
    },indent=2),encoding="utf-8")
    return pd.DataFrame(results)
