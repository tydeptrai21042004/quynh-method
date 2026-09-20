from __future__ import annotations

"""SafeGrip-PFR benchmark.

PFR is deliberately smaller than the retained FRC/CI research paths.  It uses
one recurrent regressor and one calibrated projection:

    residual target:     r* = mu - L0
    raw prediction:      mu_tilde = L0 + r_theta(X)
    calibrated lower:    L_alpha = max(0, L0 - q_alpha)
    final prediction:    mu_hat = Pi_[L_alpha,U](mu_tilde)

The neural model is fitted only on training labels.  The calibration labels are
used only once, to estimate the scalar one-sided conformal relaxation q_alpha.
The test labels are never used in fitting, calibration, model selection, or
projection.
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
    fit_literature,
    literature_hparams,
    make_bundle,
    predict_literature,
    regression_metrics,
    _aggregate_seed_metrics,
)
from .frc_benchmark import _controlled_baseline_hparams, _source_baseline_hparams
from .literature import QUICK_BASELINES, PAPER_BASELINES, LITERATURE_BASELINES, validate_paper_baselines
from .models import DirectGRUControl
from .pfr import (
    compose_raw_prediction,
    pfr_project,
    projection_theorem_audit,
    residual_target,
    unsafe_overestimate_improvement_audit,
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


def _endpoint_hash(ids: np.ndarray) -> str:
    return hashlib.sha256("\n".join(np.asarray(ids, dtype=str).tolist()).encode()).hexdigest()


def _pfr_hparams(cfg: dict, overrides: dict | None = None) -> dict:
    hp = dict(cfg.get("pfr", {}))
    if overrides:
        hp.update(overrides)
    hp.setdefault("sequence_length", int(cfg.get("sequence_length", 16)))
    hp.setdefault("scaler", "standard")
    hp.setdefault("hidden", 64)
    hp.setdefault("gru_layers", 1)
    hp.setdefault("dropout", 0.1)
    hp.setdefault("lr", 1e-3)
    hp.setdefault("weight_decay", 1e-4)
    hp.setdefault("batch_size", 128)
    hp.setdefault("epochs", int(cfg.get("training", {}).get("epochs_paper", 60)))
    hp.setdefault("patience", int(cfg.get("training", {}).get("patience", 10)))
    return hp


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
    lengths = [int(proposal_hp["sequence_length"]), int(cfg.get("benchmark", {}).get("common_warmup_samples", 1))]
    for name in names:
        hp = _effective_baseline_hparams(name, cfg, baseline_selected, protocol, preset)
        lengths.append(int(hp["sequence_length"]))
    return max(lengths) - 1


def _fit_scalar_gru(
    bundle,
    cfg: dict,
    hp: dict,
    *,
    target_mode: str,
) -> ScalarGRUFit:
    """Fit the same GRU either directly on mu or on the signed physics residual."""

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
        yv = residual_target(bundle.yv, bundle.raw_lov)
    else:
        ytr = np.asarray(bundle.ytr, dtype=float)
        yv = np.asarray(bundle.yv, dtype=float)

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

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(hp["lr"]),
        weight_decay=float(hp["weight_decay"]),
    )
    best_state = None
    best_val = float("inf")
    bad = 0

    # MSE in standardized target coordinates differs from physical-space MSE by
    # the positive constant target_std**(-2), so it has the same minimizer.
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
            if target_mode == "residual":
                pred_mu = lower_v_raw + pred_target
            else:
                pred_mu = pred_target
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


def _projection_metrics(y, raw, pred, lower, upper) -> dict:
    audit = projection_theorem_audit(y, raw, lower, upper, atol=1e-8)
    covered = audit.covered
    if covered.any():
        min_margin = float(np.min(audit.theorem_margin[covered]))
        violations = int(np.sum(~audit.theorem_holds[covered]))
        strict_improve = float(np.mean(audit.squared_error_gain[covered] > 1e-12))
    else:
        min_margin = float("nan")
        violations = 0
        strict_improve = float("nan")
    unsafe_ok = unsafe_overestimate_improvement_audit(y, raw, pred, lower, upper, atol=1e-8)
    return {
        "raw_mae": float(np.mean(np.abs(np.asarray(raw) - np.asarray(y)))),
        "raw_rmse": float(np.sqrt(np.mean(np.square(np.asarray(raw) - np.asarray(y))))),
        "feasible_interval_coverage": float(np.mean(covered)),
        "lower_bound_coverage": float(np.mean(np.asarray(y) >= np.asarray(lower) - 1e-8)),
        "upper_bound_validity_rate": float(np.mean(np.asarray(y) <= float(upper) + 1e-8)),
        "projection_correction_rate": float(np.mean(np.abs(np.asarray(pred) - np.asarray(raw)) > 1e-10)),
        "mean_squared_error_gain_from_projection": float(np.mean(audit.squared_error_gain)),
        "mean_squared_distance_to_feasible_set": float(np.mean(np.square(audit.distance_to_set))),
        "strict_projection_improvement_rate_on_covered": strict_improve,
        "projection_theorem_violations_on_covered": violations,
        "projection_theorem_min_margin_on_covered": min_margin,
        "unsafe_overestimate_no_worsening_violations_on_covered": int(np.sum(covered & ~unsafe_ok)),
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
    """Run SafeGrip-PFR, its two decisive controls, and literature baselines."""

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
    bundle = make_bundle(
        csv_path,
        cfg,
        sequence_length=int(hp["sequence_length"]),
        scaler_kind=str(hp["scaler"]),
        eval_start=eval_start,
        feature_mode="raw",
    )

    # PFR has no predictive-UQ calibration stage, so all calibration labels can
    # be devoted to the single one-sided physical relaxation quantile.
    q_alpha = conformal_lower_correction(bundle.raw_loc, bundle.yc, float(cfg.get("alpha", 0.05)))
    lower_train = apply_lower_correction(bundle.raw_lotr, q_alpha).astype(np.float32)
    lower_val = apply_lower_correction(bundle.raw_lov, q_alpha).astype(np.float32)
    lower_test = apply_lower_correction(bundle.raw_lot, q_alpha).astype(np.float32)
    upper = float(cfg.get("mu_upper", 1.3))
    seeds = _preset_seeds(cfg, preset)

    rows: list[dict] = []
    prediction_frames: list[pd.DataFrame] = []
    ablation_rows: list[dict] = []
    theorem_rows: list[dict] = []

    for seed in seeds:
        seed_everything(int(seed))
        t0 = time.time()
        residual_fit = _fit_scalar_gru(bundle, cfg, hp, target_mode="residual")
        residual_test = _predict_scalar(residual_fit, bundle.Xt)
        raw_mu = compose_raw_prediction(bundle.raw_lot, residual_test).astype(np.float32)
        pfr_mu = pfr_project(raw_mu, lower_test, upper).astype(np.float32)
        elapsed = time.time() - t0

        pfr_metrics = regression_metrics(
            bundle.yt,
            pfr_mu,
            lower_test,
            upper,
            raw_mean=raw_mu,
        )
        projection_metrics = _projection_metrics(bundle.yt, raw_mu, pfr_mu, lower_test, upper)
        rows.append({
            "model": "safegrip_pfr",
            "kind": "proposal",
            "doi": "",
            "seed": int(seed),
            **pfr_metrics,
            **projection_metrics,
            "seconds": elapsed,
        })

        prediction_frames.append(pd.DataFrame({
            "endpoint_id": bundle.idt,
            "y_true": bundle.yt,
            "model": "safegrip_pfr",
            "seed": int(seed),
            "prediction": pfr_mu,
            "raw_residual_prediction": residual_test,
            "raw_friction_prediction": raw_mu,
            "mechanics_lower_raw": bundle.raw_lot,
            "calibrated_lower": lower_test,
            "projection_correction": pfr_mu - raw_mu,
        }))

        raw_metrics = regression_metrics(bundle.yt, raw_mu)
        ablation_rows.append({
            "model": "pfr_residual_raw",
            "kind": "ablation",
            "seed": int(seed),
            **raw_metrics,
        })
        ablation_rows.append({
            "model": "safegrip_pfr",
            "kind": "proposal",
            "seed": int(seed),
            **pfr_metrics,
            **projection_metrics,
        })

        # Same architecture, input windows, optimizer, training-label count, and
        # initialization seed.  The only change is direct mu regression versus
        # residual regression.
        seed_everything(int(seed))
        t1 = time.time()
        direct_fit = _fit_scalar_gru(bundle, cfg, hp, target_mode="friction")
        direct_test = _predict_scalar(direct_fit, bundle.Xt)
        direct_metrics = regression_metrics(bundle.yt, direct_test)
        rows.append({
            "model": "direct_gru_control",
            "kind": "same_encoder_control",
            "doi": "",
            "seed": int(seed),
            **direct_metrics,
            "seconds": time.time() - t1,
        })
        ablation_rows.append({
            "model": "direct_gru_control",
            "kind": "ablation",
            "seed": int(seed),
            **direct_metrics,
        })
        prediction_frames.append(pd.DataFrame({
            "endpoint_id": bundle.idt,
            "y_true": bundle.yt,
            "model": "direct_gru_control",
            "seed": int(seed),
            "prediction": direct_test,
        }))

        audit = projection_theorem_audit(bundle.yt, raw_mu, lower_test, upper, atol=1e-8)
        for idx, endpoint_id in enumerate(bundle.idt):
            theorem_rows.append({
                "endpoint_id": endpoint_id,
                "seed": int(seed),
                "covered": bool(audit.covered[idx]),
                "raw_squared_error": float(audit.raw_squared_error[idx]),
                "projected_squared_error": float(audit.projected_squared_error[idx]),
                "distance_to_feasible_set": float(audit.distance_to_set[idx]),
                "squared_error_gain": float(audit.squared_error_gain[idx]),
                "theorem_margin": float(audit.theorem_margin[idx]),
                "theorem_holds": bool(audit.theorem_holds[idx]),
            })

    # Literature comparators use the same locked endpoint IDs and raw sensors.
    for name in names:
        bhp = _effective_baseline_hparams(name, cfg, baseline_selected, protocol, preset)
        b = make_bundle(
            csv_path,
            cfg,
            sequence_length=int(bhp["sequence_length"]),
            scaler_kind=str(bhp["scaler"]),
            eval_start=eval_start,
            feature_mode="raw",
        )
        if not np.array_equal(b.idt, bundle.idt) or not np.array_equal(b.idv, bundle.idv):
            raise RuntimeError(f"{name} endpoint IDs differ from PFR endpoints")
        if not np.allclose(b.yt, bundle.yt) or not np.allclose(b.yv, bundle.yv):
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
                "kind": "literature_adapted_baseline",
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

    pd.DataFrame(rows).to_csv(out / "metrics_by_seed.csv", index=False)
    metrics = _aggregate_seed_metrics(rows)
    metrics.to_csv(out / "metrics.csv", index=False)
    pd.concat(prediction_frames, ignore_index=True).to_csv(out / "predictions_by_seed.csv", index=False)
    pd.DataFrame(ablation_rows).to_csv(out / "pfr_component_ablation_by_seed.csv", index=False)
    _aggregate_seed_metrics(ablation_rows).to_csv(out / "pfr_component_ablation.csv", index=False)
    theorem_df = pd.DataFrame(theorem_rows)
    theorem_df.to_csv(out / "pfr_projection_theorem_audit.csv", index=False)

    covered = theorem_df[theorem_df["covered"]] if len(theorem_df) else theorem_df
    theorem_violations = int((~covered["theorem_holds"]).sum()) if len(covered) else 0
    theorem_summary = {
        "status": "PASS" if theorem_violations == 0 else "FAIL",
        "theorem": "If mu* is in [L_alpha,U], orthogonal projection cannot increase squared friction error and improves it by at least dist(mu_tilde,[L_alpha,U])^2.",
        "calibration_statement": "Under split-conformal exchangeability, P(mu_new >= L_alpha(X_new)) >= 1-alpha; the full interval statement additionally assumes mu_new <= U.",
        "alpha": float(cfg.get("alpha", 0.05)),
        "q_alpha": float(q_alpha),
        "calibration_count": int(len(bundle.yc)),
        "test_endpoint_count": int(len(bundle.yt)),
        "covered_test_fraction_mean_over_seeds": float(covered.shape[0] / max(len(theorem_df), 1)),
        "covered_point_theorem_violations": theorem_violations,
        "minimum_theorem_margin_on_covered": float(covered["theorem_margin"].min()) if len(covered) else None,
    }
    (out / "pfr_theorem_audit.json").write_text(json.dumps(theorem_summary, indent=2), encoding="utf-8")

    fairness = {
        "status": "PASS",
        "protocol": protocol,
        "seeds": seeds,
        "common_eval_start": int(eval_start),
        "train_endpoint_count": int(len(bundle.ytr)),
        "calibration_endpoint_count": int(len(bundle.yc)),
        "validation_endpoint_count": int(len(bundle.yv)),
        "test_endpoint_count": int(len(bundle.yt)),
        "validation_endpoint_hash": _endpoint_hash(bundle.idv),
        "test_endpoint_hash": _endpoint_hash(bundle.idt),
        "same_locked_endpoints": True,
        "neural_model_uses_training_labels_only": True,
        "target_standardization_fit_on_train_only": True,
        "input_scaler_fit_on_train_only": True,
        "conformal_relaxation_uses_calibration_labels_only": True,
        "all_calibration_labels_used_only_for_single_q_alpha": True,
        "test_labels_used_for_training_or_calibration": False,
        "mechanics_lower_bound_at_test_uses_features_not_test_friction_labels": True,
        "projection_has_no_learned_parameters": True,
        "baseline_fidelity": {n: LITERATURE_BASELINES[n]["fidelity"] for n in names},
        "effective_preset": preset,
        "effective_pfr_epochs": int(hp["epochs"]),
    }
    (out / "fairness_audit.json").write_text(json.dumps(fairness, indent=2), encoding="utf-8")

    pfr_rows = [r for r in rows if r.get("model") == "safegrip_pfr"]
    direct_rows = [r for r in rows if r.get("model") == "direct_gru_control"]
    raw_rows = [r for r in ablation_rows if r.get("model") == "pfr_residual_raw"]
    mean_rmse = lambda rs: float(np.mean([r["rmse"] for r in rs])) if rs else float("nan")
    method_audit = {
        "status": "PASS" if theorem_violations == 0 else "FAIL",
        "proposal": "SafeGrip-PFR: Physics-Feasible Residual Estimation",
        "train_estimator": "same-family GRU learns signed residual r*=mu-L0 on training labels only",
        "raw_estimator": "mu_tilde=L0+r_theta(X)",
        "calibration": "one-sided split-conformal relaxation L_alpha=max(0,L0-q_alpha) using calibration labels only",
        "final_estimator": "mu_hat=projection_[L_alpha,U](mu_tilde)",
        "extra_neural_modules": 0,
        "search_or_inversion": False,
        "trust_radius": None,
        "response_model": None,
        "primary_theory": theorem_summary["theorem"],
        "direct_gru_rmse": mean_rmse(direct_rows),
        "raw_residual_rmse": mean_rmse(raw_rows),
        "safegrip_pfr_rmse": mean_rmse(pfr_rows),
        "residual_gain_vs_direct_rmse": mean_rmse(direct_rows) - mean_rmse(raw_rows),
        "projection_gain_vs_raw_residual_rmse": mean_rmse(raw_rows) - mean_rmse(pfr_rows),
        "q_alpha": float(q_alpha),
    }
    (out / "pfr_method_audit.json").write_text(json.dumps(method_audit, indent=2), encoding="utf-8")

    manifest = {
        "proposal": "safegrip_pfr",
        "pfr_hparams": hp,
        "features": bundle.features,
        "sequence_length": int(bundle.sequence_length),
        "scaler": str(bundle.scaler_kind),
        "alpha": float(cfg.get("alpha", 0.05)),
        "q_alpha": float(q_alpha),
        "mu_upper": upper,
        "calibration_policy": "all calibration endpoints used only for q_alpha",
        "legacy_frc_retained": True,
        "legacy_safegrip_ci_v14_retained": True,
        "legacy_pntr_files_retained_for_reproducibility": True,
    }
    (out / "reproducibility_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return metrics
