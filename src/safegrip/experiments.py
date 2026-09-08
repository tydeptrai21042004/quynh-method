from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import json
import tempfile
from copy import deepcopy

import numpy as np
import pandas as pd

from .benchmark import (
    make_bundle,
    common_eval_start,
    windows_for_split,
    regression_metrics,
    fit_proposal,
    predict_proposal_details,
    run_benchmark,
)
from .physics import (
    vehicle_level_lower_bound,
    conformal_lower_correction,
    apply_lower_correction,
    project_numpy,
    robust_force_utilization_lower,
)
from .utils import ensure_dir, seed_everything


def _proposal_sequence_length(results_dir: str | Path, cfg: dict) -> int:
    p = Path(results_dir) / "proposal_hparams.json"
    if p.exists():
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            return int(data.get("sequence_length", cfg.get("sequence_length", 64)))
        except Exception:
            pass
    return int(cfg.get("sequence_length", 64))


def _bundle_for_results(csv_path, results_dir, cfg):
    protocol_path=Path(results_dir)/"evaluation_protocol.json"
    eval_start=common_eval_start(cfg)
    if protocol_path.exists():
        try:
            eval_start=int(json.loads(protocol_path.read_text(encoding="utf-8")).get("common_eval_start",eval_start))
        except Exception:
            pass
    return make_bundle(
        csv_path,
        cfg,
        sequence_length=_proposal_sequence_length(results_dir, cfg),
        scaler_kind="standard",
        eval_start=eval_start,
        proposal_features=True,
    )


def _inverse_endpoint_features(bundle, split: str) -> pd.DataFrame:
    mapping = {
        "train": bundle.Xtr,
        "calibration": bundle.Xc,
        "validation": bundle.Xv,
        "test": bundle.Xt,
    }
    X = mapping[split]
    if len(X) == 0:
        return pd.DataFrame(columns=bundle.features)
    last = X[:, -1, :]
    raw = bundle.scaler.inverse_transform(last)
    return pd.DataFrame(raw, columns=bundle.features)


def _prediction_model_columns(pred: pd.DataFrame) -> list[str]:
    reserved = {
        "endpoint_id", "y_true", "physics_lower", "physics_lower_raw",
    }
    suffixes = (
        "_sigma", "_raw", "_latent", "_gate", "_excitation",
        "_pi95_low_physics", "_pi95_high_physics", "_pi95_low_raw", "_pi95_high_raw",
    )
    return [c for c in pred.columns if c not in reserved and not c.endswith(suffixes) and "_latent_seed_" not in c]


def run_excitation_analysis(csv_path, results_dir, out_dir, cfg, bins: int = 4) -> pd.DataFrame:
    """Stratify test performance by dynamic excitation.

    This directly tests the partial-identification hypothesis: weak excitation
    should generally leave a wider admissible set than stronger excitation.
    """
    out = ensure_dir(out_dir)
    pred_path = Path(results_dir) / "predictions.csv"
    if not pred_path.exists():
        raise FileNotFoundError(f"Run benchmark first; missing {pred_path}")
    pred = pd.read_csv(pred_path)
    b = _bundle_for_results(csv_path, results_dir, cfg)
    if "endpoint_id" in pred and not np.array_equal(pred.endpoint_id.astype(str).to_numpy(), b.idt.astype(str)):
        raise RuntimeError("Benchmark predictions do not correspond to the current prepared test endpoints")
    excitation = np.asarray(b.et, float)
    frame = pd.DataFrame({
        "endpoint_id": b.idt,
        "y_true": b.yt,
        "physics_lower": b.lot,
        "physics_lower_raw": b.raw_lot,
        "excitation_score": excitation,
    })
    quant = min(max(2, int(bins)), max(2, len(np.unique(excitation))))
    try:
        frame["excitation_bin"] = pd.qcut(
            frame.excitation_score,
            q=quant,
            labels=[f"Q{i+1}" for i in range(quant)],
            duplicates="drop",
        )
    except ValueError:
        frame["excitation_bin"] = "all"
    for c in pred.columns:
        if c not in frame and c != "endpoint_id":
            frame[c] = pred[c].to_numpy()

    rows = []
    for model in _prediction_model_columns(pred):
        sigma_col = model + "_sigma"
        raw_col = model + "_raw"
        for group, g in frame.groupby("excitation_bin", observed=True, sort=True):
            sigma = g[sigma_col].to_numpy() if sigma_col in g else None
            raw_mean = g[raw_col].to_numpy() if raw_col in g else g[model].to_numpy()
            low_col=model+"_pi95_low_physics"; high_col=model+"_pi95_high_physics"
            low_raw_col=model+"_pi95_low_raw"; high_raw_col=model+"_pi95_high_raw"
            metrics = regression_metrics(
                g.y_true.to_numpy(),
                g[model].to_numpy(),
                g.physics_lower.to_numpy(),
                cfg["mu_upper"],
                sigma,
                raw_mean=raw_mean,
                project_uncertainty=False,
                interval_low=g[low_col].to_numpy() if low_col in g else None,
                interval_high=g[high_col].to_numpy() if high_col in g else None,
                interval_low_raw=g[low_raw_col].to_numpy() if low_raw_col in g else None,
                interval_high_raw=g[high_raw_col].to_numpy() if high_raw_col in g else None,
            )
            rows.append({
                "model": model,
                "excitation_bin": str(group),
                "n": int(len(g)),
                "excitation_mean": float(g.excitation_score.mean()),
                "excitation_min": float(g.excitation_score.min()),
                "excitation_max": float(g.excitation_score.max()),
                **metrics,
            })
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "excitation_metrics.csv", index=False)
    frame.to_csv(out / "excitation_samples.csv", index=False)
    (out / "excitation_protocol.json").write_text(json.dumps({
        "score": "SafeGrip-v2 label-free composite excitation score (acceleration, relative wheel spread, torque and jerk)",
        "score_range": [0.0, 1.0],
        "physics_window_samples": int(cfg.get("physics", {}).get("window_samples", 1)),
        "bins": int(bins),
        "test_only": True,
        "hypothesis": "identified-set width should decrease as informative excitation increases",
    }, indent=2), encoding="utf-8")
    return summary


def _recompute_bound(features: pd.DataFrame, cfg: dict, *, mass_scale=1.0, accel_error_scale=1.0):
    v = cfg["vehicle"]
    if not {"ax", "ay", "speed"}.issubset(features.columns):
        raise RuntimeError("Physics sensitivity requires ax, ay and speed features")
    return vehicle_level_lower_bound(
        features.ax,
        features.ay,
        features.speed,
        mass=float(v["mass_kg"]) * float(mass_scale),
        g=v["gravity"],
        crr=v["crr"],
        rho_air=v["rho_air"],
        cdA=v["cdA_m2"],
        accel_error=float(v["accel_error_ms2"]) * float(accel_error_scale),
        external_force_margin=v["external_force_margin_n"],
        vertical_force_margin=v["vertical_force_margin_n"],
    )


def run_robustness_analysis(csv_path, results_dir, out_dir, cfg) -> pd.DataFrame:
    """Post-hoc sensitivity of the physics/calibration/projection layer.

    The neural estimator is frozen.  Each scenario recomputes the *pointwise*
    mechanics bound on the full prepared table and then rebuilds the same fixed
    trailing physics window and one-sided calibration correction used by the
    main benchmark.  This avoids the earlier inconsistency where robustness was
    evaluated only at the endpoint acceleration while the main estimator used a
    window maximum.
    """
    out = ensure_dir(out_dir)
    pred = pd.read_csv(Path(results_dir) / "predictions.csv")
    if "safegrip_latent" not in pred:
        raise RuntimeError("Run the SafeGrip-v2 benchmark first; predictions.csv must contain safegrip_latent")
    base = _bundle_for_results(csv_path, results_dir, cfg)
    latent_cols=[c for c in pred.columns if c.startswith("safegrip_latent_seed_")]
    latents=[pred[c].to_numpy(float) for c in latent_cols] or [pred.safegrip_latent.to_numpy(float)]
    sigma = pred.safegrip_sigma.to_numpy(float) if "safegrip_sigma" in pred else None
    source_df = pd.read_csv(csv_path)
    if not {"ax", "ay", "speed"}.issubset(source_df.columns):
        raise RuntimeError("Physics sensitivity requires ax, ay and speed in the prepared dataset")

    scenarios = [{"scenario": "nominal", "mass_scale": 1.0, "accel_error_scale": 1.0, "external_force_scale": 1.0, "vertical_margin_scale": 1.0, "mu_upper": float(cfg["mu_upper"])}]
    for scale in (0.90, 0.95, 1.05, 1.10):
        scenarios.append({"scenario": f"mass_{scale:.2f}x", "mass_scale": scale, "accel_error_scale": 1.0, "external_force_scale": 1.0, "vertical_margin_scale": 1.0, "mu_upper": float(cfg["mu_upper"])})
    for scale in (0.5, 1.5, 2.0):
        scenarios.append({"scenario": f"accel_error_{scale:.1f}x", "mass_scale": 1.0, "accel_error_scale": scale, "external_force_scale": 1.0, "vertical_margin_scale": 1.0, "mu_upper": float(cfg["mu_upper"])})
    for mu_u in (1.10, 1.30, 1.50):
        scenarios.append({"scenario": f"mu_upper_{mu_u:.2f}", "mass_scale": 1.0, "accel_error_scale": 1.0, "external_force_scale": 1.0, "vertical_margin_scale": 1.0, "mu_upper": mu_u})
    for scale in (0.5, 1.5, 2.0):
        scenarios.append({"scenario": f"external_force_margin_{scale:.1f}x", "mass_scale": 1.0, "accel_error_scale": 1.0, "external_force_scale": scale, "vertical_margin_scale": 1.0, "mu_upper": float(cfg["mu_upper"])})
    for scale in (0.5, 1.5, 2.0):
        scenarios.append({"scenario": f"vertical_margin_{scale:.1f}x", "mass_scale": 1.0, "accel_error_scale": 1.0, "external_force_scale": 1.0, "vertical_margin_scale": scale, "mu_upper": float(cfg["mu_upper"])})

    rows = []
    sample_rows = []
    v = cfg["vehicle"]
    seq = _proposal_sequence_length(results_dir, cfg)
    start_eval = base.eval_start
    with tempfile.TemporaryDirectory(prefix="safegrip_robustness_") as tmp:
        tmp = Path(tmp)
        for sc in scenarios:
            z = source_df.copy()
            mechanics = vehicle_level_lower_bound(
                z.ax, z.ay, z.speed,
                mass=float(v["mass_kg"]) * float(sc["mass_scale"]),
                g=v["gravity"], crr=v["crr"], rho_air=v["rho_air"], cdA=v["cdA_m2"],
                accel_error=float(v["accel_error_ms2"]) * float(sc["accel_error_scale"]),
                external_force_margin=float(v["external_force_margin_n"]) * float(sc["external_force_scale"]),
                vertical_force_margin=float(v["vertical_force_margin_n"]) * float(sc["vertical_margin_scale"]),
            )
            mu_u = float(sc["mu_upper"])
            z["physics_lower_raw"] = np.minimum(np.maximum(np.asarray(mechanics, float), 0.0), mu_u)
            sc_csv = tmp / f"{sc['scenario']}.csv"
            z.to_csv(sc_csv, index=False)
            sc_cfg = deepcopy(cfg); sc_cfg["mu_upper"] = mu_u
            sb = make_bundle(sc_csv, sc_cfg, sequence_length=seq, scaler_kind="standard", eval_start=start_eval, proposal_features=True)
            if not np.array_equal(sb.idt.astype(str), base.idt.astype(str)):
                raise RuntimeError(f"Robustness scenario {sc['scenario']} changed test endpoint identity")
            lo = sb.lot
            scenario_preds=[]
            for latent in latents:
                sigmoid = 1.0 / (1.0 + np.exp(-latent))
                scenario_preds.append(lo + np.maximum(mu_u - lo, 0.0) * sigmoid)
            p=np.mean(np.stack(scenario_preds),axis=0)
            metrics = regression_metrics(sb.yt, p, lo, mu_u, sigma, raw_mean=p)
            rows.append({**sc, "calibration_q": float(sb.q), "n_seed_latents":int(len(latents)), **metrics})
            sample_rows.append(pd.DataFrame({
                "scenario": sc["scenario"], "endpoint_id": sb.idt, "y_true": sb.yt,
                "physics_lower": lo, "prediction": p,
            }))
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "physics_robustness_metrics.csv", index=False)
    pd.concat(sample_rows, ignore_index=True).to_csv(out / "physics_robustness_samples.csv", index=False)
    (out / "physics_robustness_protocol.json").write_text(json.dumps({
        "network_retrained": False,
        "purpose": "isolate sensitivity of the physics lower endpoint while keeping the learned latent friction position fixed",
        "bound_parameterization": "lower + (mu_upper-lower)*sigmoid(latent)",
        "uses_same_fixed_physics_window_as_main_benchmark": True,
        "mass_scales": [0.90, 0.95, 1.0, 1.05, 1.10],
        "accel_error_scales": [0.5, 1.0, 1.5, 2.0],
        "mu_upper_values": [1.10, 1.30, 1.50],
        "external_force_margin_scales": [0.5, 1.0, 1.5, 2.0],
        "vertical_margin_scales": [0.5, 1.0, 1.5, 2.0],
    }, indent=2), encoding="utf-8")
    return summary


def _subset_training_bundle(bundle, fraction: float, seed: int):
    n = len(bundle.Xtr)
    keep = max(1, int(round(float(fraction) * n)))
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(n, size=keep, replace=False)) if keep < n else np.arange(n)
    return replace(
        bundle,
        Xtr=bundle.Xtr[idx],
        ytr=bundle.ytr[idx],
        lotr=bundle.lotr[idx],
        raw_lotr=bundle.raw_lotr[idx],
        idtr=bundle.idtr[idx],
        etr=bundle.etr[idx],
    )


def run_scarcity_analysis(csv_path, out_dir, cfg, *, preset="paper", hp_overrides=None, fractions=None) -> pd.DataFrame:
    """Compare SafeGrip with its data-only backbone as training data decrease."""
    out = ensure_dir(out_dir)
    fractions = list(fractions or [0.10, 0.25, 0.50, 0.75, 1.00])
    seq = int((hp_overrides or {}).get("sequence_length", cfg["sequence_length"]))
    b = make_bundle(csv_path, cfg, sequence_length=seq, scaler_kind="standard", eval_start=common_eval_start(cfg), proposal_features=True)
    seeds = list(cfg.get("evaluation", {}).get("seeds", [cfg["seed"]])) if preset == "paper" else [int(cfg["seed"])]
    epochs = int(cfg["training"]["epochs_quick" if preset == "quick" else "epochs_paper"])
    rows = []
    for fraction in fractions:
        for seed in seeds:
            sb = _subset_training_bundle(b, fraction, int(seed) + 773)
            for variant in ("safegrip_data_only", "safegrip"):
                seed_everything(int(seed))
                model, _ = fit_proposal(variant, sb, cfg, epochs, hp_overrides)
                d = predict_proposal_details(model, variant, sb.Xt, sb.lot, sb.raw_lot, cfg["mu_upper"], excitation=sb.et)
                rows.append({
                    "training_fraction": float(fraction),
                    "training_windows": int(len(sb.Xtr)),
                    "model": variant,
                    "seed": int(seed),
                    **regression_metrics(
                        sb.yt, d["prediction"], d["bound"], cfg["mu_upper"], d["sigma"],
                        raw_mean=d["raw_mean"],
                        interval_low=d.get("pi95_low_physics"), interval_high=d.get("pi95_high_physics"),
                        interval_low_raw=d.get("pi95_low_raw"), interval_high_raw=d.get("pi95_high_raw"),
                    ),
                })
    by_seed = pd.DataFrame(rows)
    numeric = [c for c in by_seed.columns if pd.api.types.is_numeric_dtype(by_seed[c]) and c not in ("seed", "training_fraction", "training_windows")]
    summary_rows = []
    for (fraction, model), g in by_seed.groupby(["training_fraction", "model"], sort=True):
        row = {"training_fraction": float(fraction), "model": model, "n_seeds": int(g.seed.nunique()), "training_windows": int(round(g.training_windows.mean()))}
        for c in numeric:
            row[c] = float(g[c].mean())
            row[c + "_std"] = float(g[c].std(ddof=1)) if len(g) > 1 else 0.0
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    by_seed.to_csv(out / "scarcity_metrics_by_seed.csv", index=False)
    summary.to_csv(out / "scarcity_metrics.csv", index=False)
    return summary


def run_cross_route_analysis(csv_path, out_dir, cfg, *, preset="quick", hp_overrides=None) -> pd.DataFrame:
    """Leave one explicit route out for the final test while keeping calibration/validation on other routes."""
    out = ensure_dir(out_dir)
    df = pd.read_csv(csv_path)
    if "route_id" not in df:
        raise RuntimeError("Prepared data have no route_id column")
    routes = [r for r in sorted(df.route_id.dropna().astype(str).unique()) if r.lower() != "unknown"]
    if len(routes) < 2:
        raise RuntimeError("Cross-route evaluation requires at least two explicitly identified routes")
    rows = []
    for route in routes:
        z = df.copy()
        held = z.route_id.astype(str) == route
        # Held route is test only. Existing test rows on training routes are not
        # recycled into training; they are purged to preserve the original
        # calibration/validation chronology on those routes.
        z.loc[~held & (z.split == "test"), "split"] = "purged"
        z.loc[held, "split"] = "test"
        if not {"train", "calibration", "validation", "test"}.issubset(set(z.split.astype(str))):
            continue
        route_dir = ensure_dir(out / f"heldout_{route}")
        temp_csv = route_dir / "cross_route.csv"
        z.to_csv(temp_csv, index=False)
        metrics = run_benchmark(temp_csv, route_dir, cfg, preset=preset, models=[], hp_overrides=hp_overrides)
        safe = metrics[metrics.model == "safegrip"].copy()
        if len(safe):
            row = safe.iloc[0].to_dict()
            row["heldout_route"] = route
            rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "cross_route_metrics.csv", index=False)
    return summary


def run_force_validation(prepared_csv, out_dir, cfg, *, eps_t=250.0, eps_z=250.0) -> pd.DataFrame:
    """Validate the mechanics lower-bound calculation on prepared wheel-force data."""
    out = ensure_dir(out_dir)
    df = pd.read_csv(prepared_csv)
    if {"fx", "fy", "fz"}.issubset(df.columns):
        fx = df.fx.to_numpy(float); fy = df.fy.to_numpy(float); fz = df.fz.to_numpy(float)
    else:
        fx_cols = [c for c in df if c.startswith("fx_")]
        fy_cols = [c for c in df if c.startswith("fy_")]
        fz_cols = [c for c in df if c.startswith("fz_")]
        if not fz_cols or (not fx_cols and not fy_cols):
            raise RuntimeError("Prepared force data do not contain resolvable fx/fy/fz columns")
        fx = df[fx_cols].fillna(0.0).sum(axis=1).to_numpy(float) if fx_cols else np.zeros(len(df))
        fy = df[fy_cols].fillna(0.0).sum(axis=1).to_numpy(float) if fy_cols else np.zeros(len(df))
        fz = df[fz_cols].fillna(0.0).sum(axis=1).to_numpy(float)
    mask = np.isfinite(fx) & np.isfinite(fy) & np.isfinite(fz) & (np.abs(fz) > 1e-6)
    fx, fy, fz = fx[mask], fy[mask], fz[mask]
    nominal = np.hypot(fx, fy) / np.abs(fz)
    robust = robust_force_utilization_lower(fx, fy, fz, eps_t=eps_t, eps_z=eps_z)
    samples = pd.DataFrame({"fx": fx, "fy": fy, "fz": fz, "utilization_nominal": nominal, "utilization_robust_lower": robust})
    samples.to_csv(out / "force_validation_samples.csv", index=False)
    summary = pd.DataFrame([{
        "n": int(len(samples)),
        "eps_t_n": float(eps_t),
        "eps_z_n": float(eps_z),
        "robust_not_above_nominal_rate": float(np.mean(robust <= nominal + 1e-12)),
        "nominal_utilization_mean": float(np.mean(nominal)) if len(nominal) else np.nan,
        "robust_lower_mean": float(np.mean(robust)) if len(robust) else np.nan,
        "mean_relaxation": float(np.mean(nominal - robust)) if len(robust) else np.nan,
    }])
    summary.to_csv(out / "force_validation_metrics.csv", index=False)
    return summary
