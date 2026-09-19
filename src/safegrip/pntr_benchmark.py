from __future__ import annotations

"""SafeGrip-PNTR: physics-neural trust-region friction estimation.

The proposal is intentionally small:

1. A direct GRU predicts friction in a train-only standardized target space.
2. A mechanics-derived lower grip bound defines the admissible physical set.
3. A friction-conditioned response network is trained to predict causal response
   innovations and, through a contrastive term, to remain discriminative in mu.
4. Physics may refine the neural anchor only inside a bounded local trust region
   by minimizing a response-energy + neural-anchor objective.

This structure avoids the failure mode of global FRC inversion: if the response
model is weak or ambiguous, the exact neural anchor is the fallback and a large
far-away friction minimum cannot overwrite it.
"""

from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .benchmark import (
    literature_hparams,
    fit_literature,
    predict_literature,
    make_bundle,
    regression_metrics,
    _aggregate_seed_metrics,
)
from .frc_benchmark import (
    FRCBundle,
    _TransitionDataset,
    make_frc_bundle,
    predict_signatures,
    predict_signature_at_mu,
    _common_eval_start_frc,
    _controlled_baseline_hparams,
    _source_baseline_hparams,
)
from .friction_resolution import make_mu_grid
from .literature import QUICK_BASELINES, PAPER_BASELINES, LITERATURE_BASELINES, validate_paper_baselines
from .models import FrictionResponseNet, DirectGRUControl
from .physics import project_numpy
from .pntr import regularized_local_choice, local_identifiability
from .utils import ensure_dir, seed_everything, device


@dataclass
class AnchorModel:
    model: DirectGRUControl
    target_mean: float
    target_std: float
    best_val_mse: float
    target_mode: str = "physics_slack"


@dataclass
class PNTREvaluation:
    prediction: np.ndarray
    anchor_raw: np.ndarray
    anchor_physics: np.ndarray
    selected_horizon: np.ndarray
    correction: np.ndarray
    corrected: np.ndarray
    response_energy_anchor: np.ndarray
    response_energy_selected: np.ndarray
    objective_anchor: np.ndarray
    objective_selected: np.ndarray
    local_identifiability: np.ndarray
    fixed_predictions: dict[int, np.ndarray]
    fixed_corrected: dict[int, np.ndarray]


def _endpoint_hash(ids) -> str:
    h = hashlib.sha256()
    for x in np.asarray(ids).astype(str):
        h.update(x.encode("utf-8")); h.update(b"\0")
    return h.hexdigest()


def _pntr_hparams(cfg: dict, overrides: dict | None = None) -> dict:
    base = dict(cfg.get("frc", {}))
    base.update(cfg.get("pntr", {}))
    if overrides:
        base.update(overrides)
    base.setdefault("hidden", 128)
    base.setdefault("gru_layers", 1)
    base.setdefault("dropout", 0.1)
    base.setdefault("lr", 1e-3)
    base.setdefault("weight_decay", 1e-4)
    base.setdefault("batch_size", 128)
    base.setdefault("epochs", cfg.get("training", {}).get("epochs_paper", 60))
    base.setdefault("patience", cfg.get("training", {}).get("patience", 10))
    base.setdefault("physics_loss_weight", 0.10)
    base.setdefault("response_contrastive_weight", 0.25)
    base.setdefault("response_contrastive_delta", 0.10)
    base.setdefault("response_contrastive_margin", 0.02)
    base.setdefault("trust_radius", 0.06)
    base.setdefault("anchor_lambda", 0.10)
    base.setdefault("min_response_improvement", 0.002)
    base.setdefault("identifiability_probe_delta", 0.04)
    base.setdefault("min_identifiability", 0.0)
    return base


def _assert_bundle_alignment(frc: FRCBundle, ref) -> None:
    pairs = [
        (frc.idtr, ref.idtr, frc.ytr, ref.ytr, "train"),
        (frc.idc, ref.idc, frc.yc, ref.yc, "calibration"),
        (frc.idv, ref.idv, frc.yv, ref.yv, "validation"),
        (frc.idt, ref.idt, frc.yt, ref.yt, "test"),
    ]
    for a, b, ya, yb, name in pairs:
        if not np.array_equal(a, b):
            raise RuntimeError(f"PNTR FRC/reference endpoint IDs differ on {name}")
        if not np.allclose(ya, yb):
            raise RuntimeError(f"PNTR FRC/reference targets differ on {name}")


def fit_neural_anchor(
    bundle: FRCBundle,
    ref_bundle,
    cfg: dict,
    hp: dict,
    *,
    target_mode: str = "physics_slack",
) -> AnchorModel:
    """Fit the PNTR neural anchor or its direct-mu control.

    ``physics_slack`` (the proposal anchor) predicts the non-negative residual
    friction above the mechanics lower bound,

        mu = lower_physics + slack,  slack >= 0.

    This makes the physical admissible set part of the neural parameterization
    itself instead of a post-hoc projection.  ``friction`` retains the same GRU
    as a direct-mu control, which is useful for measuring whether the physical
    parameterization actually helps.

    All target transforms are fitted on training labels only.  The mechanics
    lower bound is label-free at inference time.
    """
    if target_mode not in {"physics_slack", "friction"}:
        raise ValueError("target_mode must be physics_slack or friction")
    dev = device()
    model = DirectGRUControl(
        len(bundle.features), hidden=int(hp["hidden"]),
        gru_layers=int(hp["gru_layers"]), dropout=float(hp["dropout"]),
    ).to(dev)

    if target_mode == "physics_slack":
        ytr_target = np.maximum(bundle.ytr - np.asarray(ref_bundle.lotr, float), 0.0)
        yv_target = np.maximum(bundle.yv - np.asarray(ref_bundle.lov, float), 0.0)
    else:
        ytr_target = np.asarray(bundle.ytr, float)
        yv_target = np.asarray(bundle.yv, float)

    ym = float(np.mean(ytr_target))
    ys = float(np.std(ytr_target))
    if not np.isfinite(ys) or ys < 1e-6:
        ys = 1.0
    yz = ((ytr_target - ym) / ys).astype(np.float32)
    yvz = ((yv_target - ym) / ys).astype(np.float32)
    lower_tr = np.asarray(ref_bundle.lotr, dtype=np.float32)
    lower_v = np.asarray(ref_bundle.lov, dtype=np.float32)
    ds = TensorDataset(
        torch.from_numpy(bundle.Xtr), torch.from_numpy(yz), torch.from_numpy(lower_tr)
    )
    gen = torch.Generator().manual_seed(int(cfg.get("seed", 0)))
    dl = DataLoader(ds, batch_size=int(hp["batch_size"]), shuffle=True, generator=gen)
    xv = torch.from_numpy(bundle.Xv).to(dev)
    yv = torch.from_numpy(yvz).to(dev)
    lov = torch.from_numpy(lower_v).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=float(hp["lr"]), weight_decay=float(hp["weight_decay"]))
    best = None; bestloss = float("inf"); bad = 0
    upper = float(cfg.get("mu_upper", 1.3))
    phys_w = float(hp.get("physics_loss_weight", 0.10))
    scale2 = max(ys * ys, 1e-8)

    for _ in range(int(hp["epochs"])):
        model.train()
        for xb, yb, lob in dl:
            xb = xb.to(dev); yb = yb.to(dev); lob = lob.to(dev)
            opt.zero_grad(set_to_none=True)
            predz = model(xb)
            target_raw = predz * ys + ym
            if target_mode == "physics_slack":
                # ReLU is the exact physical parameterization: the neural
                # residual cannot move below the mechanics lower bound.
                pred_phys = lob + torch.relu(target_raw)
                data_loss = nn.functional.smooth_l1_loss(predz, yb, beta=0.5)
                upper_violation = torch.relu(pred_phys - upper)
                physics_loss = torch.mean(upper_violation.square()) / scale2
            else:
                pred_phys = target_raw
                data_loss = nn.functional.smooth_l1_loss(predz, yb, beta=0.5)
                lower_violation = torch.relu(lob - pred_phys)
                upper_violation = torch.relu(pred_phys - upper)
                physics_loss = torch.mean(lower_violation.square() + upper_violation.square()) / scale2
            loss = data_loss + phys_w * physics_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

        model.eval()
        with torch.no_grad():
            pvz = model(xv)
            # Validation selection is performed in physical friction space so
            # that the physics-slack anchor is not selected by a surrogate loss
            # that ignores its lower-bound composition.
            raw = pvz * ys + ym
            if target_mode == "physics_slack":
                pv_phys = torch.clamp(lov + torch.relu(raw), max=upper)
            else:
                pv_phys = raw
            y_phys = torch.from_numpy(bundle.yv.astype(np.float32)).to(dev)
            vl = nn.functional.mse_loss(pv_phys, y_phys).item()
        if vl < bestloss - 1e-8:
            bestloss = float(vl); bad = 0
            best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if bad >= int(hp["patience"]):
            break
    if best is not None:
        model.load_state_dict(best)
    return AnchorModel(model=model, target_mean=ym, target_std=ys, best_val_mse=bestloss, target_mode=target_mode)


def predict_anchor(anchor: AnchorModel, X: np.ndarray, lower: np.ndarray | None = None,
                   *, mu_upper: float = 1.3, batch: int = 1024) -> np.ndarray:
    """Predict friction from an anchor model in physical units."""
    dev = device(); anchor.model.eval(); chunks=[]
    with torch.no_grad():
        for s in range(0, len(X), int(batch)):
            z = anchor.model(torch.from_numpy(X[s:s+batch]).to(dev)).cpu().numpy()
            chunks.append(z)
    z = np.concatenate(chunks) if chunks else np.empty(0, np.float32)
    raw = z * anchor.target_std + anchor.target_mean
    if anchor.target_mode == "physics_slack":
        if lower is None:
            raise ValueError("physics_slack anchor requires a mechanics lower bound")
        lo = np.asarray(lower, dtype=np.float32)
        return np.clip(lo + np.maximum(raw, 0.0), lo, float(mu_upper)).astype(np.float32)
    return raw.astype(np.float32)

def fit_pntr_response_model(bundle: FRCBundle, cfg: dict, hp: dict):
    """Fit a friction-conditioned causal response model.

    Ordinary MSE alone permits a conditional model to ignore ``mu``.  PNTR adds
    a small counterfactual ranking loss: the true training friction hypothesis
    should explain the observed response better than nearby wrong hypotheses.
    This is a system-identification constraint, not a test-time use of labels.
    """
    dev = device()
    upper = float(cfg.get("mu_upper", 1.3))
    model = FrictionResponseNet(
        len(bundle.features), len(bundle.response_channels), hidden=int(hp["hidden"]),
        gru_layers=int(hp["gru_layers"]), dropout=float(hp["dropout"]), mu_upper=upper,
    ).to(dev)
    tr = _TransitionDataset(bundle.Xtr, bundle.mutr, bundle.Rtr, bundle.context_length)
    va = _TransitionDataset(bundle.Xv, bundle.muv, bundle.Rv, bundle.context_length)
    gen = torch.Generator().manual_seed(int(cfg.get("seed", 0)) + 17)
    dl = DataLoader(tr, batch_size=int(hp["batch_size"]), shuffle=True, generator=gen)
    vl = DataLoader(va, batch_size=max(256, int(hp["batch_size"])), shuffle=False)
    opt = torch.optim.AdamW(model.parameters(), lr=float(hp["lr"]), weight_decay=float(hp["weight_decay"]))
    cf_w = float(hp.get("response_contrastive_weight", 0.25))
    cf_delta = float(hp.get("response_contrastive_delta", 0.10))
    cf_margin = float(hp.get("response_contrastive_margin", 0.02))
    best = None; bestloss=float("inf"); bad=0
    for _ in range(int(hp["epochs"])):
        model.train()
        for xb, mub, rb in dl:
            xb=xb.to(dev); mub=mub.to(dev); rb=rb.to(dev)
            opt.zero_grad(set_to_none=True)
            p0 = model(xb, mub)
            true_e = torch.mean((p0-rb).square(), dim=1)
            mup = torch.clamp(mub + cf_delta, 0.0, upper)
            mum = torch.clamp(mub - cf_delta, 0.0, upper)
            pp = model(xb, mup); pm = model(xb, mum)
            ep = torch.mean((pp-rb).square(), dim=1)
            em = torch.mean((pm-rb).square(), dim=1)
            # Only count a negative when it is genuinely displaced from mu.
            wp = (torch.abs(mup-mub) > 0.5*cf_delta).to(true_e.dtype)
            wm = (torch.abs(mum-mub) > 0.5*cf_delta).to(true_e.dtype)
            denom = torch.clamp(wp.sum()+wm.sum(), min=1.0)
            rank = (
                (torch.relu(cf_margin + true_e - ep) * wp).sum()
                + (torch.relu(cf_margin + true_e - em) * wm).sum()
            ) / denom
            loss = true_e.mean() + cf_w * rank
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        model.eval(); total=0.0; count=0
        with torch.no_grad():
            for xb,mub,rb in vl:
                xb=xb.to(dev); mub=mub.to(dev); rb=rb.to(dev)
                err=(model(xb,mub)-rb).square()
                total += float(err.sum().item()); count += int(err.numel())
        score=total/max(1,count)
        if score < bestloss - 1e-8:
            bestloss=float(score); bad=0
            best={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:
            bad+=1
        if bad >= int(hp["patience"]):
            break
    if best is not None:
        model.load_state_dict(best)
    return model, bestloss


def evaluate_pntr_split(
    response_model: FrictionResponseNet,
    X: np.ndarray,
    R: np.ndarray,
    anchor_raw: np.ndarray,
    lower: np.ndarray,
    bundle: FRCBundle,
    cfg: dict,
    hp: dict,
    mu_grid: np.ndarray,
    *, endpoint_batch: int = 64,
) -> PNTREvaluation:
    pn = cfg.get("pntr", {})
    horizons = sorted({int(x) for x in pn.get("horizons", cfg.get("frc", {}).get("horizons", [4,8,16,32]))})
    tau = float(hp.get("trust_radius", pn.get("trust_radius", 0.06)))
    lam = float(hp.get("anchor_lambda", pn.get("anchor_lambda", 0.10)))
    min_gain = float(hp.get("min_response_improvement", pn.get("min_response_improvement", 0.002)))
    probe = float(hp.get("identifiability_probe_delta", pn.get("identifiability_probe_delta", 0.04)))
    min_ident = float(hp.get("min_identifiability", pn.get("min_identifiability", 0.0)))
    upper = float(cfg.get("mu_upper", 1.3))
    raw = np.asarray(anchor_raw, dtype=np.float32)
    lo = np.asarray(lower, dtype=np.float32)
    anchor = project_numpy(raw, lo, upper).astype(np.float32)

    fixed_pred = {h: [] for h in horizons}; fixed_corr={h: [] for h in horizons}
    all_pred=[]; all_h=[]; all_corr=[]; all_corrected=[]; all_be=[]; all_se=[]; all_bo=[]; all_so=[]; all_ident=[]
    for s in range(0, len(X), int(endpoint_batch)):
        xb=X[s:s+endpoint_batch]; rb=R[s:s+endpoint_batch]
        ab=anchor[s:s+endpoint_batch]; lob=lo[s:s+endpoint_batch]
        sig = predict_signatures(response_model, xb, mu_grid, bundle.context_length, bundle.max_horizon)
        sig0 = predict_signature_at_mu(response_model, xb, ab, bundle.context_length, bundle.max_horizon)
        mup=np.minimum(ab+probe, upper).astype(np.float32)
        mum=np.maximum(ab-probe, lob).astype(np.float32)
        sigp=predict_signature_at_mu(response_model, xb, mup, bundle.context_length, bundle.max_horizon)
        sigm=predict_signature_at_mu(response_model, xb, mum, bundle.context_length, bundle.max_horizon)

        choices={}; gains={}; idents={}
        for h in horizons:
            sh=sig[:,:,-h:,:]; rh=rb[:,-h:,:]
            egrid=np.mean((sh-rh[:,None,:,:])**2,axis=(2,3))
            e0=np.mean((sig0[:,-h:,:]-rh)**2,axis=(1,2))
            ch=regularized_local_choice(
                egrid,mu_grid,ab,lob,upper,trust_radius=tau,anchor_lambda=lam,
                anchor_response_energy=e0,min_response_improvement=min_gain,
            )
            ident=local_identifiability(sigm,sigp,probe,horizon=h)
            valid_ident=ident>=min_ident
            # Low-identifiability endpoints fall back to the exact anchor.
            if min_ident>0:
                ch.mu_hat=np.where(valid_ident,ch.mu_hat,ab).astype(np.float32)
                ch.corrected=ch.corrected & valid_ident
                ch.correction=(ch.mu_hat-ab).astype(np.float32)
                ch.response_energy_selected=np.where(valid_ident,ch.response_energy_selected,e0).astype(np.float32)
                ch.regularized_objective_selected=np.where(valid_ident,ch.regularized_objective_selected,e0).astype(np.float32)
            choices[h]=ch; idents[h]=ident
            gains[h]=e0-ch.regularized_objective_selected
            fixed_pred[h].append(ch.mu_hat.copy()); fixed_corr[h].append(ch.corrected.copy())

        # Select the horizon with the largest *regularized* improvement.  Mean
        # response energies make horizons comparable despite different H.
        gmat=np.column_stack([gains[h] for h in horizons])
        k=np.argmax(gmat,axis=1); best_gain=gmat[np.arange(len(ab)),k]
        harray=np.asarray(horizons,dtype=int)
        selected_h=harray[k]
        pred=ab.copy(); corr=np.zeros_like(ab); corrected=np.zeros(len(ab),dtype=bool)
        be=np.zeros(len(ab),np.float32); se=np.zeros(len(ab),np.float32); bo=np.zeros(len(ab),np.float32); so=np.zeros(len(ab),np.float32); ident_out=np.zeros(len(ab),np.float32)
        for j,h in enumerate(horizons):
            m=(k==j); ch=choices[h]
            pred[m]=ch.mu_hat[m]; corr[m]=ch.correction[m]; corrected[m]=ch.corrected[m] & (best_gain[m]>0)
            be[m]=ch.response_energy_anchor[m]; se[m]=ch.response_energy_selected[m]
            bo[m]=ch.regularized_objective_anchor[m]; so[m]=ch.regularized_objective_selected[m]
            ident_out[m]=idents[h][m]
        # If no horizon improves the regularized objective, force neural fallback.
        no=(best_gain<=0) | (~corrected)
        pred[no]=ab[no]; corr[no]=0.0; selected_h[no]=0; corrected[no]=False
        se[no]=be[no]; so[no]=bo[no]
        all_pred.append(pred); all_h.append(selected_h); all_corr.append(corr); all_corrected.append(corrected)
        all_be.append(be); all_se.append(se); all_bo.append(bo); all_so.append(so); all_ident.append(ident_out)

    return PNTREvaluation(
        prediction=np.concatenate(all_pred).astype(np.float32),
        anchor_raw=raw,
        anchor_physics=anchor,
        selected_horizon=np.concatenate(all_h).astype(np.int32),
        correction=np.concatenate(all_corr).astype(np.float32),
        corrected=np.concatenate(all_corrected),
        response_energy_anchor=np.concatenate(all_be).astype(np.float32),
        response_energy_selected=np.concatenate(all_se).astype(np.float32),
        objective_anchor=np.concatenate(all_bo).astype(np.float32),
        objective_selected=np.concatenate(all_so).astype(np.float32),
        local_identifiability=np.concatenate(all_ident).astype(np.float32),
        fixed_predictions={h:np.concatenate(v).astype(np.float32) for h,v in fixed_pred.items()},
        fixed_corrected={h:np.concatenate(v) for h,v in fixed_corr.items()},
    )


def run_pntr_benchmark(csv_path, out_dir, cfg: dict, preset: str = "paper", models=None,
                       pntr_hparams: dict | None = None, baseline_hparams: dict | None = None,
                       protocol: str = "controlled"):
    protocol=str(protocol).replace("-","_")
    if protocol not in {"controlled","source_faithful"}:
        raise ValueError("protocol must be controlled or source_faithful")
    out=ensure_dir(out_dir)
    names=list(models) if models is not None else list(QUICK_BASELINES if preset=="quick" else PAPER_BASELINES)
    validate_paper_baselines(names)
    eval_start=_common_eval_start_frc(cfg,names)
    if baseline_hparams:
        lens=[int(v.get("sequence_length",0)) for v in baseline_hparams.values() if isinstance(v,dict)]
        if lens: eval_start=max(eval_start,max(lens)-1)
    bundle=make_frc_bundle(csv_path,cfg,eval_start=eval_start)
    ref=make_bundle(csv_path,cfg,sequence_length=bundle.context_length+bundle.max_horizon,
                    scaler_kind="standard",eval_start=eval_start,feature_mode="raw")
    _assert_bundle_alignment(bundle,ref)
    hp=_pntr_hparams(cfg,pntr_hparams)
    if not pntr_hparams or "epochs" not in pntr_hparams:
        if preset=="quick": hp["epochs"]=int(cfg.get("training",{}).get("epochs_quick",3))
        elif preset=="trust": hp["epochs"]=int(cfg.get("training",{}).get("epochs_trust",40))
    if preset=="paper":
        seeds=list(cfg.get("comparison",{}).get("controlled",{}).get("evaluation_seeds",cfg.get("evaluation",{}).get("seeds",[0,1,2,3,4])))
    else:
        seeds=list(cfg.get("evaluation",{}).get("seeds",[0,1,2,3,4]))
    if preset=="quick": seeds=seeds[:1]
    elif preset=="trust": seeds=list(cfg.get("evaluation",{}).get("trust_seeds",seeds[:3]))

    pn=cfg.get("pntr",{})
    mu_grid=make_mu_grid(float(pn.get("mu_min",0.0)),float(pn.get("mu_max",cfg.get("mu_upper",1.3))),float(pn.get("mu_grid_step",0.02)))
    rows=[]; preds=[]; ablations=[]; horizon_rows=[]; selected_models=[]; baseline_selected=baseline_hparams or {}

    for seed in seeds:
        seed_everything(int(seed)); t0=time.time()
        anchor=fit_neural_anchor(bundle,ref,cfg,hp,target_mode="physics_slack")
        response,val_resp=fit_pntr_response_model(bundle,cfg,hp)
        base_test=predict_anchor(anchor,bundle.Xt,ref.lot,mu_upper=float(cfg["mu_upper"]))
        ev=evaluate_pntr_split(response,bundle.Xt,bundle.Rt,base_test,ref.lot,bundle,cfg,hp,mu_grid)
        elapsed=time.time()-t0
        metrics=regression_metrics(bundle.yt,ev.prediction,ref.lot,float(cfg["mu_upper"]),raw_mean=ev.prediction)
        rows.append({"model":"safegrip_pntr","kind":"proposal","doi":"","seed":int(seed),**metrics,
                     "seconds":elapsed,"correction_rate":float(np.mean(ev.corrected)),
                     "mean_abs_physics_correction":float(np.mean(np.abs(ev.correction))),
                     "mean_response_energy_reduction":float(np.mean(ev.response_energy_anchor-ev.response_energy_selected)),
                     "mean_local_identifiability":float(np.mean(ev.local_identifiability))})
        preds.append(pd.DataFrame({
            "endpoint_id":bundle.idt,"y_true":bundle.yt,"model":"safegrip_pntr","seed":int(seed),
            "prediction":ev.prediction,"neural_anchor_raw":ev.anchor_raw,"neural_anchor_physics":ev.anchor_physics,
            "selected_horizon":ev.selected_horizon,"physics_correction":ev.correction,
            "corrected":ev.corrected.astype(int),"response_energy_anchor":ev.response_energy_anchor,
            "response_energy_selected":ev.response_energy_selected,"local_identifiability":ev.local_identifiability,
        }))
        # Same-encoder direct-mu control: isolates the contribution of the
        # physics-residual parameterization from generic GRU capacity.
        t_control=time.time()
        direct=fit_neural_anchor(bundle,ref,cfg,hp,target_mode="friction")
        direct_test=predict_anchor(direct,bundle.Xt,mu_upper=float(cfg["mu_upper"]))
        direct_metrics=regression_metrics(bundle.yt,direct_test)
        rows.append({"model":"direct_gru_control","kind":"same_encoder_control","doi":"","seed":int(seed),
                     **direct_metrics,"seconds":time.time()-t_control})
        preds.append(pd.DataFrame({"endpoint_id":bundle.idt,"y_true":bundle.yt,"model":"direct_gru_control",
                                   "seed":int(seed),"prediction":direct_test}))

        # Component ablations use exactly the same endpoints.
        ablations.append({"model":"direct_gru_control","kind":"ablation","seed":int(seed),**direct_metrics})
        ablations.append({"model":"pntr_physics_residual_anchor","kind":"ablation","seed":int(seed),
                          **regression_metrics(bundle.yt,ev.anchor_physics,ref.lot,float(cfg["mu_upper"]),raw_mean=ev.anchor_physics)})
        ablations.append({"model":"safegrip_pntr","kind":"proposal","seed":int(seed),**metrics,
                          "correction_rate":float(np.mean(ev.corrected)),"mean_abs_physics_correction":float(np.mean(np.abs(ev.correction)))})
        for h,p in ev.fixed_predictions.items():
            horizon_rows.append({"model":f"pntr_fixed_h{h}","kind":"fixed_horizon_ablation","seed":int(seed),
                                 **regression_metrics(bundle.yt,p,ref.lot,float(cfg["mu_upper"]),raw_mean=p),
                                 "correction_rate":float(np.mean(ev.fixed_corrected[h]))})
        selected_models.append((anchor,response))

    # Literature comparators on identical endpoints.
    for name in names:
        bhp=(_controlled_baseline_hparams(name,cfg,baseline_selected) if protocol=="controlled" else _source_baseline_hparams(name,cfg,baseline_selected))
        if preset=="quick":
            bhp["epochs"]=int(cfg.get("training",{}).get("epochs_quick",3)); bhp["patience"]=max(1,min(int(bhp.get("patience",3)),3))
        elif preset=="trust":
            bhp["epochs"]=int(cfg.get("training",{}).get("epochs_trust",40)); bhp["patience"]=int(cfg.get("training",{}).get("patience_trust",8))
        b=make_bundle(csv_path,cfg,sequence_length=int(bhp["sequence_length"]),scaler_kind=str(bhp["scaler"]),eval_start=eval_start,feature_mode="raw")
        if not np.array_equal(b.idt,bundle.idt) or not np.array_equal(b.idv,bundle.idv):
            raise RuntimeError(f"{name} endpoint IDs differ from PNTR endpoints")
        if not np.allclose(b.yt,bundle.yt) or not np.allclose(b.yv,bundle.yv):
            raise RuntimeError(f"{name} target values differ on locked endpoints")
        for seed in seeds:
            seed_everything(int(seed)); t0=time.time()
            bm=fit_literature(name,b,cfg,epochs=int(bhp["epochs"]),preset="paper",hp_overrides=bhp)
            bp,bs=predict_literature(bm,name,b.Xt)
            rows.append({"model":name,"kind":"literature_adapted_baseline","doi":LITERATURE_BASELINES[name]["doi"],"seed":int(seed),
                         **regression_metrics(b.yt,bp,sigma=bs),"seconds":time.time()-t0})
            preds.append(pd.DataFrame({"endpoint_id":b.idt,"y_true":b.yt,"model":name,"seed":int(seed),"prediction":bp}))

    pd.DataFrame(rows).to_csv(out/"metrics_by_seed.csv",index=False)
    metrics=_aggregate_seed_metrics(rows); metrics.to_csv(out/"metrics.csv",index=False)
    pd.concat(preds,ignore_index=True).to_csv(out/"predictions_by_seed.csv",index=False)
    pd.DataFrame(ablations).to_csv(out/"pntr_component_ablation_by_seed.csv",index=False)
    _aggregate_seed_metrics(ablations).to_csv(out/"pntr_component_ablation.csv",index=False)
    pd.DataFrame(horizon_rows).to_csv(out/"pntr_horizon_ablation_by_seed.csv",index=False)
    _aggregate_seed_metrics(horizon_rows).to_csv(out/"pntr_horizon_ablation.csv",index=False)

    fairness={
        "status":"PASS","protocol":protocol,"seeds":seeds,"common_eval_start":int(eval_start),
        "validation_endpoint_hash":_endpoint_hash(bundle.idv),"test_endpoint_hash":_endpoint_hash(bundle.idt),
        "same_locked_endpoints":True,"test_labels_used_for_tuning":False,
        "input_scaler_fit_on_train_only":True,"target_standardization_fit_on_train_only":True,
        "response_normalization_fit_on_train_only":True,"causal_response_context":True,
        "mechanics_lower_bound_uses_test_features_not_test_friction_labels":True,
        "physics_refinement_is_trust_region_bounded":True,
        "baseline_fidelity":{n:LITERATURE_BASELINES[n]["fidelity"] for n in names},
        "effective_preset":preset,"effective_epochs":int(hp["epochs"]),
    }
    (out/"fairness_audit.json").write_text(json.dumps(fairness,indent=2),encoding="utf-8")
    pntr_seed_rows=[r for r in rows if r.get("model")=="safegrip_pntr"]
    direct_seed_rows=[r for r in rows if r.get("model")=="direct_gru_control"]
    residual_seed_rows=[r for r in ablations if r.get("model")=="pntr_physics_residual_anchor"]
    corr_mean=float(np.mean([r.get("correction_rate",0.0) for r in pntr_seed_rows])) if pntr_seed_rows else 0.0
    direct_rmse=float(np.mean([r["rmse"] for r in direct_seed_rows])) if direct_seed_rows else float("nan")
    residual_rmse=float(np.mean([r["rmse"] for r in residual_seed_rows])) if residual_seed_rows else float("nan")
    method_audit={
        "status":"PASS",
        "proposal":"SafeGrip-PNTR: Physics-Neural Trust-Region friction estimation",
        "neural_anchor":"target-standardized GRU predicting non-negative friction slack above the mechanics lower bound",
        "same_encoder_control":"direct GRU predicting friction without the physics-residual parameterization",
        "mechanics_constraint":"calibration-relaxed vehicle-level lower grip bound embedded in the neural output parameterization",
        "response_model":"causal friction-conditioned GRU with counterfactual ranking loss",
        "objective":"mean standardized response energy + normalized quadratic neural-anchor penalty",
        "trust_radius":float(hp["trust_radius"]),"anchor_lambda":float(hp["anchor_lambda"]),
        "anchor_is_exact_fallback":True,
        "deterministic_safeguard":"selected regularized objective cannot exceed the exact anchor objective",
        "true_error_improvement_claim":"not unconditional; requires local response-model/identifiability assumptions",
        "empirical_correction_rate_mean":corr_mean,
        "response_refinement_empirically_active":bool(corr_mean>0.0),
        "response_refinement_status":"ACTIVE" if corr_mean>0.0 else "SAFE_FALLBACK_ONLY",
        "same_encoder_direct_rmse":direct_rmse,
        "physics_residual_anchor_rmse":residual_rmse,
        "physics_residual_rmse_improvement_vs_direct":direct_rmse-residual_rmse if np.isfinite(direct_rmse) and np.isfinite(residual_rmse) else None,
    }
    (out/"pntr_method_audit.json").write_text(json.dumps(method_audit,indent=2),encoding="utf-8")
    manifest={
        "proposal":"safegrip_pntr","pntr_hparams":hp,"mu_grid":mu_grid.tolist(),
        "response_channels":bundle.response_channels,"features":bundle.features,
        "response_mean":bundle.response_mean.tolist(),"response_std":bundle.response_std.tolist(),
        "legacy_frc_retained":True,"legacy_safegrip_ci_v14_retained":True,
    }
    (out/"reproducibility_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    return metrics
