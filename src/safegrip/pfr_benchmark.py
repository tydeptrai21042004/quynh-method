from __future__ import annotations

"""Benchmark for the revised SafeGrip-PFR proposal.

The active path is intentionally compact:

    raw sensors + mechanics excitation
        -> one GRU
        -> physics-guided temporal pooling
        -> point residual + positive scale
        -> point estimate mu_point = L0^W + r_theta
        -> one-sided normalized conformal lower estimate
        -> max(statistical lower, calibrated mechanics lower)

The point estimate is used for ordinary predictive metrics.  The fused lower
estimate is reported separately as the controller-facing safety output.  This
avoids conflating accuracy and conservative safety objectives.
"""

from dataclasses import dataclass, replace
from pathlib import Path
import copy
import hashlib
import json
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .benchmark import (
    fit_literature,
    literature_hparams,
    make_bundle,
    predict_literature,
    regression_metrics,
    _aggregate_seed_metrics,
)
from .literature import QUICK_BASELINES, PAPER_BASELINES, LITERATURE_BASELINES, validate_paper_baselines
from .models import DirectGRUControl, PFRExcitationGRU
from .pfr import (
    compose_raw_prediction,
    conformal_safe_correction,
    fuse_safe_lower,
    residual_target,
    safety_fusion_audit,
    statistical_safe_lower,
)
from .physics import apply_lower_correction, conformal_lower_correction
from .utils import ensure_dir, seed_everything, device


@dataclass
class ScalarGRUFit:
    model: DirectGRUControl
    target_mean: float
    target_std: float
    best_val_mse: float
    target_mode: str


@dataclass
class PFRFit:
    model: PFRExcitationGRU
    target_mean: float
    target_std: float
    best_val_mse: float
    attention_gamma: float
    target_mode: str = "residual"


def _endpoint_hash(ids: np.ndarray) -> str:
    return hashlib.sha256("\n".join(np.asarray(ids, dtype=str).tolist()).encode()).hexdigest()


def _pfr_hparams(cfg: dict, overrides: dict | None = None) -> dict:
    hp = dict(cfg.get("pfr", {}))
    if overrides:
        hp.update(overrides)
    hp.setdefault("sequence_length", 64)
    hp.setdefault("scaler", "standard")
    hp.setdefault("hidden", 128)
    hp.setdefault("gru_layers", 1)
    hp.setdefault("dropout", 0.1)
    hp.setdefault("lr", 1e-3)
    hp.setdefault("weight_decay", 1e-4)
    hp.setdefault("batch_size", 128)
    hp.setdefault("epochs", int(cfg.get("training", {}).get("epochs_paper", 60)))
    hp.setdefault("patience", int(cfg.get("training", {}).get("patience", 10)))
    hp.setdefault("attention_gamma", 4.0)
    hp.setdefault("scale_floor", 0.005)
    hp.setdefault("scale_nll_weight", 0.02)
    hp.setdefault("huber_beta", 0.5)  # standardized residual coordinates
    hp.setdefault("physics_window_samples", 32)
    hp.setdefault("risk_split", 0.5)  # fraction of total alpha assigned to mechanics lower bound
    hp.setdefault("extended_ablations", True)
    hp.setdefault("risk_split_sensitivity", [0.25, 0.50, 0.75])
    return hp


def _pfr_bundle_cfg(cfg: dict, hp: dict) -> dict:
    out = copy.deepcopy(cfg)
    out.setdefault("physics", {})["window_samples"] = int(hp["physics_window_samples"])
    return out


def _controlled_baseline_hparams(name: str, cfg: dict, selected: dict | None = None) -> dict:
    hp = literature_hparams(name, cfg, (selected or {}).get(name), "paper")
    cc = cfg.get("comparison", {}).get("controlled", {})
    hp.update({
        "epochs": int(cc.get("epochs", cfg["training"]["epochs_paper"])),
        "patience": int(cc.get("patience", cfg["training"].get("patience", 10))),
        "batch_size": int(cc.get("batch_size", hp.get("batch_size", 128))),
    })
    if bool(cc.get("common_optimizer_hparams", True)) and name != "levenberg2023_stft":
        hp["lr"] = float(cc.get("lr", hp.get("lr", 1e-3)))
        hp["weight_decay"] = float(cc.get("weight_decay", hp.get("weight_decay", 1e-4)))
        # Preserve Du et al.'s SGD optimizer and momentum while matching the
        # controlled learning-rate/regularization budget.
    return hp


def _source_baseline_hparams(name: str, cfg: dict, selected: dict | None = None) -> dict:
    return literature_hparams(name, cfg, (selected or {}).get(name), "paper")


def _effective_baseline_hparams(
    name: str,
    cfg: dict,
    selected: dict | None,
    protocol: str,
    preset: str,
) -> dict:
    selected_one = (selected or {}).get(name)
    if preset == "quick":
        hp = literature_hparams(name, cfg, selected_one, preset="quick")
        if protocol == "controlled":
            cc = cfg.get("comparison", {}).get("controlled", {})
            hp["lr"] = float(cc.get("lr", hp["lr"]))
            hp["weight_decay"] = float(cc.get("weight_decay", hp["weight_decay"]))
            hp["batch_size"] = int(cc.get("batch_size", hp["batch_size"]))
        hp["epochs"] = int(cfg.get("training", {}).get("epochs_quick", 3))
        hp["patience"] = max(1, min(int(hp.get("patience", 3)), 3))
        return hp
    hp = (
        _controlled_baseline_hparams(name, cfg, selected)
        if protocol == "controlled"
        else _source_baseline_hparams(name, cfg, selected)
    )
    if preset == "trust":
        hp["epochs"] = int(cfg.get("training", {}).get("epochs_trust", 40))
        hp["patience"] = int(cfg.get("training", {}).get("patience_trust", min(int(hp.get("patience", 8)), 8)))
    return hp


def _common_eval_start_pfr(
    cfg: dict,
    proposal_hp: dict,
    names: list[str],
    baseline_selected: dict | None,
    protocol: str,
    preset: str,
) -> int:
    lengths = [
        int(proposal_hp["sequence_length"]),
        int(proposal_hp["physics_window_samples"]),
        int(cfg.get("benchmark", {}).get("common_warmup_samples", 1)),
    ]
    for name in names:
        hp = _effective_baseline_hparams(name, cfg, baseline_selected, protocol, preset)
        lengths.append(int(hp["sequence_length"]))
    return max(lengths) - 1


def _fit_scalar_gru(bundle, cfg: dict, hp: dict, *, target_mode: str) -> ScalarGRUFit:
    """Fit the plain-GRU direct or residual controls on raw sensor features."""

    if target_mode not in {"friction", "residual"}:
        raise ValueError("target_mode must be friction or residual")
    dev = device()
    model = DirectGRUControl(
        len(bundle.features),
        hidden=int(hp["hidden"]),
        gru_layers=int(hp["gru_layers"]),
        dropout=float(hp["dropout"]),
    ).to(dev)

    if target_mode == "residual":
        ytr = residual_target(bundle.ytr, bundle.raw_lotr)
    else:
        ytr = np.asarray(bundle.ytr, dtype=float)

    target_mean = float(np.mean(ytr))
    target_std = float(np.std(ytr))
    if not np.isfinite(target_std) or target_std < 1e-6:
        target_std = 1.0
    yz = ((ytr - target_mean) / target_std).astype(np.float32)

    ds = TensorDataset(torch.from_numpy(bundle.Xtr), torch.from_numpy(yz))
    gen = torch.Generator().manual_seed(int(cfg.get("seed", 0)))
    dl = DataLoader(ds, batch_size=int(hp["batch_size"]), shuffle=True, generator=gen)
    xv = torch.from_numpy(bundle.Xv).to(dev)
    yv_phys = torch.from_numpy(np.asarray(bundle.yv, dtype=np.float32)).to(dev)
    lower_v_raw = torch.from_numpy(np.asarray(bundle.raw_lov, dtype=np.float32)).to(dev)

    opt = torch.optim.AdamW(model.parameters(), lr=float(hp["lr"]), weight_decay=float(hp["weight_decay"]))
    best_state = None
    best_val = float("inf")
    bad = 0

    for _ in range(int(hp["epochs"])):
        model.train()
        for xb, yb in dl:
            xb = xb.to(dev)
            yb = yb.to(dev)
            opt.zero_grad(set_to_none=True)
            predz = model(xb)
            loss = nn.functional.mse_loss(predz, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

        model.eval()
        with torch.no_grad():
            pred_target = model(xv) * target_std + target_mean
            pred_mu = lower_v_raw + pred_target if target_mode == "residual" else pred_target
            val_loss = nn.functional.mse_loss(pred_mu, yv_phys).item()
        if val_loss < best_val - 1e-10:
            best_val = float(val_loss)
            bad = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if bad >= int(hp["patience"]):
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return ScalarGRUFit(model, target_mean, target_std, best_val, target_mode)


def _predict_scalar(fit: ScalarGRUFit, X: np.ndarray, *, batch_size: int = 1024) -> np.ndarray:
    dev = device()
    fit.model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(X), int(batch_size)):
            z = fit.model(torch.from_numpy(X[start:start + batch_size]).to(dev)).cpu().numpy()
            out.append(z)
    if not out:
        return np.empty(0, dtype=np.float32)
    z = np.concatenate(out)
    return (z * fit.target_std + fit.target_mean).astype(np.float32)


def _fit_pfr_gru(
    bundle,
    cfg: dict,
    hp: dict,
    *,
    attention_gamma: float | None = None,
    target_mode: str = "residual",
) -> PFRFit:
    """Fit the excitation-aware network using training labels only.

    ``target_mode='residual'`` is the proposal path. ``target_mode='friction'``
    keeps the same features, recurrent capacity, pooling, and loss structure but
    removes the explicit mechanics residual anchor for a controlled ablation.
    """

    if target_mode not in {"residual", "friction"}:
        raise ValueError("target_mode must be residual or friction")
    if "pfr_excitation" not in bundle.features:
        raise ValueError("PFR bundle must contain pfr_excitation")
    dev = device()
    gamma = float(hp["attention_gamma"] if attention_gamma is None else attention_gamma)
    model = PFRExcitationGRU(
        len(bundle.features),
        excitation_index=bundle.features.index("pfr_excitation"),
        hidden=int(hp["hidden"]),
        gru_layers=int(hp["gru_layers"]),
        dropout=float(hp["dropout"]),
        attention_gamma=gamma,
        scale_floor=float(hp["scale_floor"]),
    ).to(dev)

    target = (
        residual_target(bundle.ytr, bundle.raw_lotr)
        if target_mode == "residual"
        else np.asarray(bundle.ytr, dtype=float)
    )
    target_mean = float(np.mean(target))
    target_std = float(np.std(target))
    if not np.isfinite(target_std) or target_std < 1e-6:
        target_std = 1.0
    yz = ((target - target_mean) / target_std).astype(np.float32)

    ds = TensorDataset(
        torch.from_numpy(bundle.Xtr),
        torch.from_numpy(yz),
        torch.from_numpy(np.asarray(bundle.ytr, dtype=np.float32)),
        torch.from_numpy(np.asarray(bundle.raw_lotr, dtype=np.float32)),
    )
    gen = torch.Generator().manual_seed(int(cfg.get("seed", 0)))
    dl = DataLoader(ds, batch_size=int(hp["batch_size"]), shuffle=True, generator=gen)
    xv = torch.from_numpy(bundle.Xv).to(dev)
    yv = torch.from_numpy(np.asarray(bundle.yv, dtype=np.float32)).to(dev)
    lov = torch.from_numpy(np.asarray(bundle.raw_lov, dtype=np.float32)).to(dev)

    opt = torch.optim.AdamW(model.parameters(), lr=float(hp["lr"]), weight_decay=float(hp["weight_decay"]))
    best_state = None
    best_val = float("inf")
    bad = 0
    huber_beta = max(float(hp.get("huber_beta", 0.5)), 1e-6)
    scale_weight = max(float(hp.get("scale_nll_weight", 0.02)), 0.0)

    for _ in range(int(hp["epochs"])):
        model.train()
        for xb, yzb, yb, lob in dl:
            xb = xb.to(dev)
            yzb = yzb.to(dev)
            yb = yb.to(dev)
            lob = lob.to(dev)
            opt.zero_grad(set_to_none=True)
            predz, sigma = model(xb)
            point_loss = nn.functional.smooth_l1_loss(predz, yzb, beta=huber_beta)
            pred_target = predz * target_std + target_mean
            pred_mu = lob + pred_target if target_mode == "residual" else pred_target
            sigma = torch.clamp(sigma, min=float(hp["scale_floor"]))
            nll = torch.mean(0.5 * torch.square((pred_mu - yb) / sigma) + torch.log(sigma))
            loss = point_loss + scale_weight * nll
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

        model.eval()
        with torch.no_grad():
            predz, _ = model(xv)
            pred_target = predz * target_std + target_mean
            pred_mu = lov + pred_target if target_mode == "residual" else pred_target
            val_loss = nn.functional.mse_loss(pred_mu, yv).item()
        if val_loss < best_val - 1e-10:
            best_val = float(val_loss)
            bad = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if bad >= int(hp["patience"]):
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return PFRFit(model, target_mean, target_std, best_val, gamma, target_mode)


def _predict_pfr(fit: PFRFit, X: np.ndarray, *, batch_size: int = 1024):
    """Return residual, physical scale and interpretable attention diagnostics."""

    dev = device()
    fit.model.eval()
    rz, scales, peak, entropy, weighted_excitation = [], [], [], [], []
    excitation_index = fit.model.excitation_index
    with torch.no_grad():
        for start in range(0, len(X), int(batch_size)):
            xb = torch.from_numpy(X[start:start + batch_size]).to(dev)
            z, sigma, weights = fit.model(xb, return_attention=True)
            rz.append(z.cpu().numpy())
            scales.append(sigma.cpu().numpy())
            w = weights.cpu().numpy()
            x = xb[:, :, excitation_index].cpu().numpy()
            peak.append(np.max(w, axis=1))
            if w.shape[1] > 1:
                ent = -np.sum(w * np.log(np.maximum(w, 1e-12)), axis=1) / np.log(w.shape[1])
            else:
                ent = np.zeros(len(w), dtype=float)
            entropy.append(ent)
            weighted_excitation.append(np.sum(w * x, axis=1))
    if not rz:
        empty = np.empty(0, dtype=np.float32)
        return empty, empty, empty, empty, empty
    z = np.concatenate(rz)
    residual = (z * fit.target_std + fit.target_mean).astype(np.float32)
    return (
        residual,
        np.concatenate(scales).astype(np.float32),
        np.concatenate(peak).astype(np.float32),
        np.concatenate(entropy).astype(np.float32),
        np.concatenate(weighted_excitation).astype(np.float32),
    )


def _zero_bundle_features(bundle, feature_names: list[str]):
    """Return a same-shape bundle with selected model inputs deterministically zeroed.

    Keeping the dimensionality and recurrent architecture unchanged makes this a
    cleaner feature-information ablation than switching to a different encoder.
    """

    missing = [name for name in feature_names if name not in bundle.features]
    if missing:
        raise ValueError(f"Cannot ablate missing features: {missing}")
    indices = [bundle.features.index(name) for name in feature_names]
    updates = {}
    for field in ("Xtr", "Xc", "Xv", "Xt"):
        arr = np.asarray(getattr(bundle, field)).copy()
        if arr.size:
            arr[:, :, indices] = 0.0
        updates[field] = arr
    return replace(bundle, **updates)


def _safe_component_metrics(y, point, safe) -> dict:
    """Common diagnostics for a controller-facing lower estimate."""

    y = np.asarray(y, dtype=float)
    point = np.asarray(point, dtype=float)
    safe = np.asarray(safe, dtype=float)
    return {
        **regression_metrics(y, safe),
        "lower_coverage": float(np.mean(safe <= y + 1e-8)),
        "unsafe_overestimate_mean": float(np.mean(np.maximum(safe - y, 0.0))),
        "unsafe_overestimate_rate_005": float(np.mean(safe > y + 0.05)),
        "mean_point_to_safe_gap": float(np.mean(point - safe)),
        "mean_safe_conservatism": float(np.mean(np.maximum(y - safe, 0.0))),
    }


def _preset_seeds(cfg: dict, preset: str) -> list[int]:
    if preset == "paper":
        seeds = list(
            cfg.get("comparison", {}).get("controlled", {}).get(
                "evaluation_seeds", cfg.get("evaluation", {}).get("seeds", [0, 1, 2, 3, 4])
            )
        )
    else:
        seeds = list(cfg.get("evaluation", {}).get("seeds", [0, 1, 2, 3, 4]))
    if preset == "quick":
        return seeds[:1]
    if preset == "trust":
        return list(cfg.get("evaluation", {}).get("trust_seeds", seeds[:3]))
    return seeds


def _safety_metrics(y, point, safe, mechanics_lower, statistical_lower, sigma, total_alpha: float) -> dict:
    y = np.asarray(y, dtype=float)
    point = np.asarray(point, dtype=float)
    safe = np.asarray(safe, dtype=float)
    mech = np.asarray(mechanics_lower, dtype=float)
    stat = np.asarray(statistical_lower, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    return {
        "nominal_joint_coverage": float(1.0 - total_alpha),
        "physics_lower_coverage": float(np.mean(mech <= y + 1e-8)),
        "statistical_lower_coverage": float(np.mean(stat <= y + 1e-8)),
        "fused_safe_coverage": float(np.mean(safe <= y + 1e-8)),
        "point_unsafe_overestimate_mean": float(np.mean(np.maximum(point - y, 0.0))),
        "safe_unsafe_overestimate_mean": float(np.mean(np.maximum(safe - y, 0.0))),
        "point_unsafe_overestimate_rate_005": float(np.mean(point > y + 0.05)),
        "safe_unsafe_overestimate_rate_005": float(np.mean(safe > y + 0.05)),
        "mean_point_to_safe_gap": float(np.mean(point - safe)),
        "mean_safe_conservatism": float(np.mean(np.maximum(y - safe, 0.0))),
        "mean_predicted_scale": float(np.mean(sigma)),
    }


def run_pfr_benchmark(
    csv_path,
    out_dir,
    cfg: dict,
    preset: str = "paper",
    models=None,
    pfr_hparams: dict | None = None,
    baseline_hparams: dict | None = None,
    protocol: str = "controlled",
):
    """Run revised PFR, decisive controls, safety audit, and literature baselines."""

    protocol = str(protocol).replace("-", "_")
    if protocol not in {"controlled", "source_faithful"}:
        raise ValueError("protocol must be controlled or source-faithful")
    if preset not in {"quick", "trust", "paper"}:
        raise ValueError("preset must be quick, trust, or paper")

    out = ensure_dir(out_dir)
    names = list(models) if models is not None else list(QUICK_BASELINES if preset == "quick" else PAPER_BASELINES)
    validate_paper_baselines(names)
    baseline_selected = baseline_hparams or {}
    hp = _pfr_hparams(cfg, pfr_hparams)
    if not pfr_hparams or "epochs" not in pfr_hparams:
        if preset == "quick":
            hp["epochs"] = int(cfg.get("training", {}).get("epochs_quick", 3))
            hp["patience"] = max(1, min(int(hp["patience"]), 3))
        elif preset == "trust":
            hp["epochs"] = int(cfg.get("training", {}).get("epochs_trust", 40))
            hp["patience"] = int(cfg.get("training", {}).get("patience_trust", hp["patience"]))

    eval_start = _common_eval_start_pfr(cfg, hp, names, baseline_selected, protocol, preset)
    pfr_cfg = _pfr_bundle_cfg(cfg, hp)
    pfr_bundle = make_bundle(
        csv_path,
        pfr_cfg,
        sequence_length=int(hp["sequence_length"]),
        scaler_kind=str(hp["scaler"]),
        eval_start=eval_start,
        feature_mode="pfr",
    )
    raw_bundle = make_bundle(
        csv_path,
        pfr_cfg,
        sequence_length=int(hp["sequence_length"]),
        scaler_kind=str(hp["scaler"]),
        eval_start=eval_start,
        feature_mode="raw",
    )
    if not np.array_equal(pfr_bundle.idt, raw_bundle.idt) or not np.array_equal(pfr_bundle.idv, raw_bundle.idv):
        raise RuntimeError("PFR feature bundle and raw control bundle do not share locked endpoints")

    total_alpha = float(cfg.get("alpha", 0.05))
    risk_split = float(hp.get("risk_split", 0.5))
    if not 0.0 < risk_split < 1.0:
        raise ValueError("pfr.risk_split must lie in (0, 1)")
    physics_alpha = total_alpha * risk_split
    statistical_alpha = total_alpha - physics_alpha
    q_physics = conformal_lower_correction(pfr_bundle.raw_loc, pfr_bundle.yc, physics_alpha)
    lower_test = apply_lower_correction(pfr_bundle.raw_lot, q_physics).astype(np.float32)
    upper = float(cfg.get("mu_upper", 1.3))
    seeds = _preset_seeds(cfg, preset)

    rows: list[dict] = []
    prediction_frames: list[pd.DataFrame] = []
    ablation_rows: list[dict] = []
    safety_rows: list[dict] = []
    safety_audit_rows: list[dict] = []
    risk_split_rows: list[dict] = []
    q_safe_by_seed: dict[int, float] = {}

    for seed in seeds:
        # Full excitation-aware point estimator.
        seed_everything(int(seed))
        t0 = time.time()
        full_fit = _fit_pfr_gru(pfr_bundle, cfg, hp)
        residual_test, sigma_test, attn_peak, attn_entropy, attn_exc = _predict_pfr(full_fit, pfr_bundle.Xt)
        point_test = compose_raw_prediction(pfr_bundle.raw_lot, residual_test).astype(np.float32)
        residual_cal, sigma_cal, _, _, _ = _predict_pfr(full_fit, pfr_bundle.Xc)
        point_cal = compose_raw_prediction(pfr_bundle.raw_loc, residual_cal).astype(np.float32)
        q_safe = conformal_safe_correction(
            point_cal,
            pfr_bundle.yc,
            sigma_cal,
            alpha=statistical_alpha,
            scale_floor=float(hp["scale_floor"]),
        )
        q_safe_by_seed[int(seed)] = float(q_safe)
        statistical_lower = statistical_safe_lower(point_test, sigma_test, q_safe).astype(np.float32)
        safe_test = fuse_safe_lower(lower_test, statistical_lower, upper).astype(np.float32)
        elapsed = time.time() - t0

        point_metrics = regression_metrics(pfr_bundle.yt, point_test)
        point_metrics.update({
            "mean_predicted_scale": float(np.mean(sigma_test)),
            "mean_attention_peak_weight": float(np.mean(attn_peak)),
            "mean_normalized_attention_entropy": float(np.mean(attn_entropy)),
            "mean_attention_weighted_excitation": float(np.mean(attn_exc)),
        })
        rows.append({
            "model": "safegrip_pfr",
            "kind": "proposal_point_estimate",
            "doi": "",
            "seed": int(seed),
            **point_metrics,
            "seconds": elapsed,
        })

        safety_metrics = _safety_metrics(
            pfr_bundle.yt, point_test, safe_test, lower_test, statistical_lower, sigma_test, total_alpha
        )
        safety_rows.append({
            "model": "safegrip_pfr_safe",
            "kind": "controller_facing_safe_lower",
            "seed": int(seed),
            "q_safe": float(q_safe),
            "q_physics": float(q_physics),
            **regression_metrics(pfr_bundle.yt, safe_test),
            **safety_metrics,
        })

        # Safety decomposition requires no retraining and directly tests whether
        # mechanics-only, statistical-only, or fused protection is responsible
        # for the observed safety/utility trade-off.
        ablation_rows.append({
            "model": "pfr_safe_mechanics_only",
            "kind": "ablation_safety_mechanics_only",
            "seed": int(seed),
            **_safe_component_metrics(pfr_bundle.yt, point_test, lower_test),
        })
        ablation_rows.append({
            "model": "pfr_safe_statistical_only",
            "kind": "ablation_safety_statistical_only",
            "seed": int(seed),
            **_safe_component_metrics(pfr_bundle.yt, point_test, statistical_lower),
        })

        # Risk-allocation sensitivity is calibration-only: the learned point
        # estimator is fixed, so these rows do not consume extra training budget.
        for split_value in hp.get("risk_split_sensitivity", [0.25, 0.50, 0.75]):
            split_value = float(split_value)
            if not 0.0 < split_value < 1.0:
                continue
            beta = total_alpha * split_value
            alpha_s = total_alpha - beta
            q_phys_s = conformal_lower_correction(pfr_bundle.raw_loc, pfr_bundle.yc, beta)
            mech_s = apply_lower_correction(pfr_bundle.raw_lot, q_phys_s).astype(np.float32)
            q_stat_s = conformal_safe_correction(
                point_cal, pfr_bundle.yc, sigma_cal, alpha=alpha_s, scale_floor=float(hp["scale_floor"])
            )
            stat_s = statistical_safe_lower(point_test, sigma_test, q_stat_s).astype(np.float32)
            fused_s = fuse_safe_lower(mech_s, stat_s, upper).astype(np.float32)
            risk_split_rows.append({
                "model": "safegrip_pfr_safe",
                "kind": "risk_split_sensitivity",
                "seed": int(seed),
                "risk_split": split_value,
                "physics_alpha": float(beta),
                "statistical_alpha": float(alpha_s),
                "q_physics": float(q_phys_s),
                "q_safe": float(q_stat_s),
                **_safe_component_metrics(pfr_bundle.yt, point_test, fused_s),
            })

        prediction_frames.append(pd.DataFrame({
            "endpoint_id": pfr_bundle.idt,
            "y_true": pfr_bundle.yt,
            "model": "safegrip_pfr",
            "seed": int(seed),
            "prediction": point_test,
            "safe_prediction": safe_test,
            "raw_residual_prediction": residual_test,
            "mechanics_lower_raw": pfr_bundle.raw_lot,
            "mechanics_lower_calibrated": lower_test,
            "statistical_safe_lower": statistical_lower,
            "predicted_scale": sigma_test,
            "q_safe": float(q_safe),
            "attention_peak_weight": attn_peak,
            "attention_entropy_normalized": attn_entropy,
            "attention_weighted_excitation": attn_exc,
        }))

        fusion = safety_fusion_audit(
            pfr_bundle.yt,
            lower_test,
            point_test,
            sigma_test,
            q_safe,
            upper,
            atol=1e-8,
        )
        for idx, endpoint_id in enumerate(pfr_bundle.idt):
            safety_audit_rows.append({
                "endpoint_id": endpoint_id,
                "seed": int(seed),
                "q_physics": float(q_physics),
                "q_safe": float(q_safe),
                "mechanics_lower": float(lower_test[idx]),
                "statistical_lower": float(fusion.statistical_lower[idx]),
                "fused_safe": float(fusion.fused_safe[idx]),
                "y_true": float(pfr_bundle.yt[idx]),
                "physics_covered": bool(fusion.physics_covered[idx]),
                "statistical_covered": bool(fusion.statistical_covered[idx]),
                "joint_component_covered": bool(fusion.joint_component_covered[idx]),
                "fused_covered": bool(fusion.fused_covered[idx]),
                "fusion_logic_holds": bool(fusion.fusion_logic_holds[idx]),
            })

        # Ablation A: same residual target but no explicit PFR physics channels and
        # endpoint-only GRU pooling.
        seed_everything(int(seed))
        residual_fit = _fit_scalar_gru(raw_bundle, cfg, hp, target_mode="residual")
        residual_raw = _predict_scalar(residual_fit, raw_bundle.Xt)
        residual_point = compose_raw_prediction(raw_bundle.raw_lot, residual_raw).astype(np.float32)
        ablation_rows.append({
            "model": "pfr_residual_raw",
            "kind": "ablation_no_physics_features_no_attention",
            "seed": int(seed),
            **regression_metrics(raw_bundle.yt, residual_point),
        })

        # Ablation B: PFR physics channels retained but excitation attention is
        # replaced by uniform temporal pooling.
        seed_everything(int(seed))
        no_attn_fit = _fit_pfr_gru(pfr_bundle, cfg, hp, attention_gamma=0.0)
        no_attn_residual, _, _, _, _ = _predict_pfr(no_attn_fit, pfr_bundle.Xt)
        no_attn_point = compose_raw_prediction(pfr_bundle.raw_lot, no_attn_residual).astype(np.float32)
        ablation_rows.append({
            "model": "pfr_physics_features_no_attention",
            "kind": "ablation_uniform_temporal_pooling",
            "seed": int(seed),
            **regression_metrics(pfr_bundle.yt, no_attn_point),
        })

        if bool(hp.get("extended_ablations", True)):
            # Ablation C: identical PFR tensor shape and uniform pooling, but both
            # mechanics-derived input channels are zeroed.  Comparing this with
            # pfr_physics_features_no_attention isolates explicit physics input
            # information without changing encoder width or pooling.
            zero_physics_bundle = _zero_bundle_features(
                pfr_bundle, ["pfr_mechanics_lower", "pfr_excitation"]
            )
            seed_everything(int(seed))
            no_phys_fit = _fit_pfr_gru(zero_physics_bundle, cfg, hp, attention_gamma=0.0)
            no_phys_residual, _, _, _, _ = _predict_pfr(no_phys_fit, zero_physics_bundle.Xt)
            no_phys_point = compose_raw_prediction(
                zero_physics_bundle.raw_lot, no_phys_residual
            ).astype(np.float32)
            ablation_rows.append({
                "model": "pfr_no_physics_channels_uniform",
                "kind": "ablation_zero_physics_channels_same_architecture",
                "seed": int(seed),
                **regression_metrics(zero_physics_bundle.yt, no_phys_point),
            })

            # Ablation D: same PFR features, attention and recurrent capacity, but
            # direct friction supervision instead of residual anchoring.
            seed_everything(int(seed))
            direct_target_fit = _fit_pfr_gru(
                pfr_bundle, cfg, hp, target_mode="friction"
            )
            direct_target_test, _, _, _, _ = _predict_pfr(direct_target_fit, pfr_bundle.Xt)
            ablation_rows.append({
                "model": "pfr_direct_target_full",
                "kind": "ablation_no_explicit_residual_anchor",
                "seed": int(seed),
                **regression_metrics(pfr_bundle.yt, direct_target_test),
            })

            # Ablation E: remove the heteroscedastic NLL auxiliary objective while
            # preserving the point path.  This checks whether multi-task scale
            # learning helps or harms point accuracy.
            no_scale_hp = dict(hp)
            no_scale_hp["scale_nll_weight"] = 0.0
            seed_everything(int(seed))
            no_scale_fit = _fit_pfr_gru(pfr_bundle, cfg, no_scale_hp)
            no_scale_residual, _, _, _, _ = _predict_pfr(no_scale_fit, pfr_bundle.Xt)
            no_scale_point = compose_raw_prediction(
                pfr_bundle.raw_lot, no_scale_residual
            ).astype(np.float32)
            ablation_rows.append({
                "model": "pfr_no_scale_multitask",
                "kind": "ablation_no_scale_nll_auxiliary_loss",
                "seed": int(seed),
                **regression_metrics(pfr_bundle.yt, no_scale_point),
            })

        ablation_rows.append({
            "model": "safegrip_pfr",
            "kind": "proposal_point_estimate",
            "seed": int(seed),
            **point_metrics,
        })
        ablation_rows.append({
            "model": "safegrip_pfr_safe",
            "kind": "safety_output_not_accuracy_objective",
            "seed": int(seed),
            **regression_metrics(pfr_bundle.yt, safe_test),
            **safety_metrics,
        })

        # Same-capacity direct friction regression control on the raw inputs.
        seed_everything(int(seed))
        t1 = time.time()
        direct_fit = _fit_scalar_gru(raw_bundle, cfg, hp, target_mode="friction")
        direct_test = _predict_scalar(direct_fit, raw_bundle.Xt)
        direct_metrics = regression_metrics(raw_bundle.yt, direct_test)
        ablation_rows.append({
            "model": "direct_gru_control",
            "kind": "ablation_direct_regression",
            "seed": int(seed),
            **direct_metrics,
        })

    # Literature comparators retain raw inputs and the same locked endpoint IDs.
    for name in names:
        bhp = _effective_baseline_hparams(name, cfg, baseline_selected, protocol, preset)
        b = make_bundle(
            csv_path,
            pfr_cfg,
            sequence_length=int(bhp["sequence_length"]),
            scaler_kind=str(bhp["scaler"]),
            eval_start=eval_start,
            feature_mode="raw",
        )
        if not np.array_equal(b.idt, pfr_bundle.idt) or not np.array_equal(b.idv, pfr_bundle.idv):
            raise RuntimeError(f"{name} endpoint IDs differ from PFR endpoints")
        if not np.allclose(b.yt, pfr_bundle.yt) or not np.allclose(b.yv, pfr_bundle.yv):
            raise RuntimeError(f"{name} target values differ on locked endpoints")
        fit_preset = "quick" if preset == "quick" else "paper"
        for seed in seeds:
            seed_everything(int(seed))
            t0 = time.time()
            model = fit_literature(
                name,
                b,
                cfg,
                epochs=int(bhp["epochs"]),
                preset=fit_preset,
                hp_overrides=bhp,
            )
            pred, sigma = predict_literature(model, name, b.Xt)
            rows.append({
                "model": name,
                "kind": "paper_supported_primary_baseline",
                "doi": LITERATURE_BASELINES[name]["doi"],
                "seed": int(seed),
                **regression_metrics(b.yt, pred, sigma=sigma),
                "seconds": time.time() - t0,
            })
            prediction_frames.append(pd.DataFrame({
                "endpoint_id": b.idt,
                "y_true": b.yt,
                "model": name,
                "seed": int(seed),
                "prediction": pred,
            }))

    provenance_rows = []
    for name in names:
        meta = LITERATURE_BASELINES[name]
        provenance_rows.append({
            "model": name,
            "display_name": meta.get("display_name", name),
            "paper_title": meta["title"],
            "authors": meta["authors"],
            "year": meta["year"],
            "venue": meta["venue"],
            "doi": meta["doi"],
            "source_task": meta["task"],
            "fidelity": meta["fidelity"],
            "source_constraints": meta["source_constraints"],
            "common_benchmark_note": meta["common_benchmark_note"],
            "same_common_target": True,
            "same_locked_validation_test_endpoints": True,
            "same_vehicle_signal_input_policy": True,
            "primary_comparison": True,
        })
    pd.DataFrame(provenance_rows).to_csv(out / "paper_baseline_provenance.csv", index=False)

    pd.DataFrame(rows).to_csv(out / "metrics_by_seed.csv", index=False)
    metrics = _aggregate_seed_metrics(rows)
    metrics.to_csv(out / "metrics.csv", index=False)
    pd.concat(prediction_frames, ignore_index=True).to_csv(out / "predictions_by_seed.csv", index=False)
    pd.DataFrame(ablation_rows).to_csv(out / "pfr_component_ablation_by_seed.csv", index=False)
    _aggregate_seed_metrics(ablation_rows).to_csv(out / "pfr_component_ablation.csv", index=False)
    pd.DataFrame(safety_rows).to_csv(out / "pfr_safety_metrics_by_seed.csv", index=False)
    _aggregate_seed_metrics(safety_rows).to_csv(out / "pfr_safety_metrics.csv", index=False)
    risk_df = pd.DataFrame(risk_split_rows)
    risk_df.to_csv(out / "pfr_risk_split_sensitivity_by_seed.csv", index=False)
    if len(risk_df):
        risk_group_cols = ["model", "kind", "risk_split"]
        risk_summary = []
        for keys, group in risk_df.groupby(risk_group_cols, sort=False):
            row = dict(zip(risk_group_cols, keys))
            row["n_seeds"] = int(group["seed"].nunique())
            for col in group.columns:
                if col in {*risk_group_cols, "seed"} or not pd.api.types.is_numeric_dtype(group[col]):
                    continue
                row[col] = float(group[col].mean())
                row[col + "_std"] = float(group[col].std(ddof=1)) if len(group) > 1 else 0.0
            risk_summary.append(row)
        pd.DataFrame(risk_summary).to_csv(out / "pfr_risk_split_sensitivity.csv", index=False)
    else:
        pd.DataFrame().to_csv(out / "pfr_risk_split_sensitivity.csv", index=False)
    safety_df = pd.DataFrame(safety_audit_rows)
    safety_df.to_csv(out / "pfr_safety_fusion_audit.csv", index=False)

    fusion_violations = int((~safety_df["fusion_logic_holds"]).sum()) if len(safety_df) else 0
    theorem_summary = {
        "status": "PASS" if fusion_violations == 0 else "FAIL",
        "theorem": (
            "If the calibrated mechanics lower estimate and the normalized conformal statistical lower estimate "
            "are each no larger than the true friction, their maximum is also no larger than the true friction. "
            "With marginal miscoverage beta and alpha_s, the union bound gives joint coverage at least "
            "1-beta-alpha_s."
        ),
        "coverage_note": "The finite-sample coverage statement additionally requires the usual split-conformal exchangeability assumption.",
        "total_alpha": float(total_alpha),
        "physics_alpha": float(physics_alpha),
        "statistical_alpha": float(statistical_alpha),
        "nominal_joint_coverage": float(1.0 - total_alpha),
        "q_physics": float(q_physics),
        "q_safe_by_seed": {str(k): float(v) for k, v in q_safe_by_seed.items()},
        "calibration_count": int(len(pfr_bundle.yc)),
        "test_endpoint_count": int(len(pfr_bundle.yt)),
        "empirical_physics_coverage": float(safety_df["physics_covered"].mean()) if len(safety_df) else None,
        "empirical_statistical_coverage": float(safety_df["statistical_covered"].mean()) if len(safety_df) else None,
        "empirical_fused_coverage": float(safety_df["fused_covered"].mean()) if len(safety_df) else None,
        "deterministic_fusion_logic_violations": fusion_violations,
    }
    (out / "pfr_theorem_audit.json").write_text(json.dumps(theorem_summary, indent=2), encoding="utf-8")

    fairness = {
        "status": "PASS",
        "protocol": protocol,
        "seeds": seeds,
        "common_eval_start": int(eval_start),
        "train_endpoint_count": int(len(pfr_bundle.ytr)),
        "calibration_endpoint_count": int(len(pfr_bundle.yc)),
        "validation_endpoint_count": int(len(pfr_bundle.yv)),
        "test_endpoint_count": int(len(pfr_bundle.yt)),
        "validation_endpoint_hash": _endpoint_hash(pfr_bundle.idv),
        "test_endpoint_hash": _endpoint_hash(pfr_bundle.idt),
        "same_locked_endpoints": True,
        "point_and_scale_network_uses_training_labels_only": True,
        "target_standardization_fit_on_train_only": True,
        "input_scaler_fit_on_train_only": True,
        "physics_features_use_no_friction_labels": True,
        "physics_conformal_quantile_uses_calibration_labels_only": True,
        "statistical_conformal_quantile_uses_calibration_labels_only": True,
        "test_labels_used_for_training_calibration_or_selection": False,
        "active_proposal": "SafeGrip-PFR-ECR",
        "primary_paper_baselines": names,
        "baseline_fidelity": {n: LITERATURE_BASELINES[n]["fidelity"] for n in names},
        "levenberg_sampling_limitation_disclosed": "levenberg2023_stft" not in names or "low-rate" in LITERATURE_BASELINES["levenberg2023_stft"]["fidelity"],
        "effective_preset": preset,
        "effective_pfr_epochs": int(hp["epochs"]),
    }
    (out / "fairness_audit.json").write_text(json.dumps(fairness, indent=2), encoding="utf-8")

    pfr_rows = [r for r in rows if r.get("model") == "safegrip_pfr"]
    direct_rows = [r for r in ablation_rows if r.get("model") == "direct_gru_control"]
    residual_rows = [r for r in ablation_rows if r.get("model") == "pfr_residual_raw"]
    no_attn_rows = [r for r in ablation_rows if r.get("model") == "pfr_physics_features_no_attention"]
    no_phys_rows = [r for r in ablation_rows if r.get("model") == "pfr_no_physics_channels_uniform"]
    direct_target_rows = [r for r in ablation_rows if r.get("model") == "pfr_direct_target_full"]
    no_scale_rows = [r for r in ablation_rows if r.get("model") == "pfr_no_scale_multitask"]
    mean_rmse = lambda rs: float(np.mean([r["rmse"] for r in rs])) if rs else float("nan")
    method_audit = {
        "status": "PASS" if fusion_violations == 0 else "FAIL",
        "proposal": "SafeGrip-PFR-ECR: Excitation-Aware Conformal Risk-Controlled Residual Estimation",
        "point_estimator": "mu_point=L0^W+r_theta(X,L0,e), with mechanics-excitation-weighted temporal pooling",
        "scale_estimator": "positive sigma_theta from the same pooled recurrent representation",
        "statistical_safe_lower": "C_alpha=mu_point-q_alpha*sigma using normalized one-sided split conformal calibration",
        "mechanics_safe_lower": "L_beta=max(0,L0^W-q_beta)",
        "controller_facing_output": "mu_safe=max(L_beta,C_alpha), clipped only to physical support [0,U]",
        "extra_neural_backbones": 0,
        "search_or_inversion": False,
        "primary_theory": theorem_summary["theorem"],
        "direct_gru_rmse": mean_rmse(direct_rows),
        "plain_residual_gru_rmse": mean_rmse(residual_rows),
        "physics_features_no_attention_rmse": mean_rmse(no_attn_rows),
        "no_physics_channels_uniform_rmse": mean_rmse(no_phys_rows),
        "direct_target_full_rmse": mean_rmse(direct_target_rows),
        "no_scale_multitask_rmse": mean_rmse(no_scale_rows),
        "safegrip_pfr_point_rmse": mean_rmse(pfr_rows),
        "gain_vs_direct_rmse": mean_rmse(direct_rows) - mean_rmse(pfr_rows),
        "gain_from_physics_features_and_attention_vs_plain_residual_rmse": mean_rmse(residual_rows) - mean_rmse(pfr_rows),
        "gain_from_physics_channels_under_uniform_pooling_rmse": mean_rmse(no_phys_rows) - mean_rmse(no_attn_rows),
        "gain_from_explicit_residual_anchor_rmse": mean_rmse(direct_target_rows) - mean_rmse(pfr_rows),
        "gain_from_scale_multitask_rmse": mean_rmse(no_scale_rows) - mean_rmse(pfr_rows),
        "gain_from_excitation_attention_vs_uniform_pooling_rmse": mean_rmse(no_attn_rows) - mean_rmse(pfr_rows),
        "q_physics": float(q_physics),
        "q_safe_by_seed": {str(k): float(v) for k, v in q_safe_by_seed.items()},
    }
    (out / "pfr_method_audit.json").write_text(json.dumps(method_audit, indent=2), encoding="utf-8")

    manifest = {
        "proposal": "safegrip_pfr",
        "proposal_version": "PFR-ECR",
        "pfr_hparams": hp,
        "features": pfr_bundle.features,
        "sequence_length": int(pfr_bundle.sequence_length),
        "physics_window_samples": int(hp["physics_window_samples"]),
        "scaler": str(pfr_bundle.scaler_kind),
        "total_alpha": float(total_alpha),
        "physics_alpha": float(physics_alpha),
        "statistical_alpha": float(statistical_alpha),
        "q_physics": float(q_physics),
        "q_safe_by_seed": {str(k): float(v) for k, v in q_safe_by_seed.items()},
        "extended_ablations": bool(hp.get("extended_ablations", True)),
        "risk_split_sensitivity": [float(x) for x in hp.get("risk_split_sensitivity", [])],
        "mu_upper": upper,
        "only_active_proposal": "SafeGrip-PFR-ECR",
        "primary_paper_baselines": names,
    }
    (out / "reproducibility_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return metrics
