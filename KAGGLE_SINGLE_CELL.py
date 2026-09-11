# ============================================================
# SafeGrip-CI v1.0.0 — FULL CORRECTED KAGGLE SINGLE-CELL DRIVER
# Repository:
#   https://github.com/tydeptrai21042004/quynh-method
#
# DEFAULT:
#   MODE = "trust"
#
# TRUST MODE:
#   - clone the corrected v1.0.0 repository and record exact git commit
#   - install paper/dev dependencies and run the full pytest suite
#   - verify the release audit (core repo files/configs/tests/docs)
#   - download + prepare the real LiRA friction subset
#   - print preprocessing/alignment/physics diagnostics
#   - run the 3-seed SafeGrip-CI trust benchmark
#   - run Todorovic CNN + Lampe GRU literature comparators
#   - run fairness controls (feature parity, label budget, common conformal,
#     projection parity)
#   - export per-seed predictions
#   - run hierarchical seed/trajectory bootstrap statistics
#   - run the 15 primary SafeGrip-CI v1.0 ablations
#   - verify ablation semantic distinctness and prediction distinctness
#   - create proposal-specific diagnostics for candidate/update authority
#   - export a result ZIP even if the scientific gate says REVIEW
#
# PAPER MODE:
#   Set MODE = "paper" only after the TRUST run is scientifically acceptable.
#   - validation-only SafeGrip-CI tuning (test locked)
#   - equal-budget literature-baseline tuning
#   - exact validation-endpoint hash parity check
#   - full 5-seed paper benchmark
#   - fairness controls and per-seed prediction export
#   - hierarchical seed/trajectory bootstrap statistics
#   - 5-seed controlled 18-variant CI ablation (15 mechanism + 3 training-objective controls)
#   - excitation-proxy stratification (reviewer control)
#   - physics robustness analysis
#   - optional retuned-ablation / cross-route / scarcity analyses
#   - strong paper-readiness gate
#   - PAPER_READY ZIP only when every required gate passes
#
# KAGGLE:
#   Internet = ON
#   Accelerator = T4 GPU or better recommended
#
# IMPORTANT SCIENTIFIC NOTES:
#   1) The full SafeGrip-CI proposal uses RAW sensor features for its point
#      estimator. Handcrafted excitation is NOT a proposal input.
#   2) `safegrip_excitation_proxy` is a controlled legacy comparison only.
#   3) v1.0 separates raw candidate-innovation learning from update authority.
#   4) v1.0 uses MULTI-SCALE counterfactual observability, cross-scale local-
#      linearity consistency, residual veto, and an identifiability-aware
#      inverse-dynamics disagreement veto.
#   5) Statistical inference uses matched seeds and trajectory segments, not
#      highly overlapping endpoints as independent bootstrap units.
#   6) Hyperparameter sensitivity is validation-only and never touches test labels.
# ============================================================

from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable

# ------------------------------------------------------------
# 0. USER SETTINGS
# ------------------------------------------------------------
# Run TRUST first. When it passes, change to PAPER and rerun the cell.
MODE = "trust"  # "trust" or "paper"
EXPECTED_VERSION = "1.0.0"

# Source selection. GitHub is the normal path. If the corrected v1.0 repository
# has not been pushed yet, upload quynh-method-safegrip-v1.0-full.zip as a
# Kaggle Dataset and set SOURCE_MODE = "kaggle_zip".
SOURCE_MODE = "github"  # "github" or "kaggle_zip"
GITHUB_REPO = "https://github.com/tydeptrai21042004/quynh-method.git"
GITHUB_BRANCH = "main"
KAGGLE_ZIP_PATH = ""  # optional exact path; blank => auto-discover v1.0 full ZIP under /kaggle/input

# Paper tuning budget. Keep proposal and literature comparators equal for the
# strongest tuning-budget fairness claim. Reduce both together only if needed.
PAPER_TRIALS = 60
BASELINE_TRIALS = 60
TRUST_BOOTSTRAP_REPLICATES = 2000
PAPER_BOOTSTRAP_REPLICATES = 5000

# v1.0 validation-only one-factor hyperparameter stability study.
RUN_SENSITIVITY = True
SENSITIVITY_PARAMETERS = [
    "state_persistence",
    "counterfactual_delta",
    "counterfactual_scale_span",
    "identifiability_lambda",
    "linearity_penalty",
    "acceptance_strength",
    "agreement_strength",
    "inverse_dynamics_max_step",
    "innovation_loss_weight",
    "dynamics_pretrain_epochs",
]
SENSITIVITY_EPOCHS = None  # None => use tuning.epochs from config

# Expensive supplementary analyses.
RUN_RETUNED_ABLATION = False
RETUNED_ABLATION_TRIALS = 15
RUN_CROSS_ROUTE = False
RUN_SCARCITY = False
RUN_EXTENDED_SCRIPT = False
DOWNLOAD_AUX = False

# 15 central mechanism/component ablations for SafeGrip-CI v1.0.
PRIMARY_ABLATIONS = [
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
    "safegrip_no_innovation_supervision",
    "safegrip_no_bound",
    "safegrip_no_uq",
    "safegrip",
]
PRIMARY_ABLATIONS_CSV = ",".join(PRIMARY_ABLATIONS)

# Three extra optimization/training-objective controls. They are reported as
# supplementary ablations, not as separate novelty claims.
SUPPLEMENTARY_ABLATIONS = [
    "safegrip_no_state_update_loss",
    "safegrip_no_direction_loss",
    "safegrip_no_dynamics_pretrain",
]
PAPER_ABLATIONS = PRIMARY_ABLATIONS[:-1] + SUPPLEMENTARY_ABLATIONS + ["safegrip"]
PAPER_ABLATIONS_CSV = ",".join(PAPER_ABLATIONS)

# Every ablation except no-UQ should alter the point-estimation/training path.
POINT_DISTINCT_ABLATIONS = [v for v in PAPER_ABLATIONS if v != "safegrip_no_uq"]

PAPER_MODELS = {
    "todorovic2022_cnn",
    "lampe2023_lstm",
    "lampe2023_gru",
    "schaefke2023_transformer",
    "chen2025_svdkl",
    "safegrip",
}

# This notebook driver can be newer than wrapper scripts tracked in the release.
RELEASE_AUDIT_DRIVER_EXCLUSIONS = {
    "KAGGLE_SINGLE_CELL.py",
    "KAGGLE_TRUST_SINGLE_CELL.py",
}

# ------------------------------------------------------------
# 1. PATHS
# ------------------------------------------------------------
WORK = Path("/kaggle/working")
REPO = WORK / "quynh-method"

TRUST_ZIP = WORK / "safegrip_ci_lira_trust_results_v100.zip"
PAPER_ZIP = WORK / "safegrip_ci_paper_release_v100.zip"
PAPER_REVIEW_ZIP = WORK / "safegrip_ci_paper_review_results_v100.zip"


# ------------------------------------------------------------
# 2. SMALL UTILITIES
# ------------------------------------------------------------
def run(cmd: str, cwd: Path | None = None, env: dict | None = None, check: bool = True):
    """Run a shell command with visible logging."""
    print("\n" + "=" * 112)
    print("RUN:", cmd)
    print("=" * 112)
    return subprocess.run(
        cmd,
        shell=True,
        check=check,
        cwd=str(cwd) if cwd else None,
        executable="/bin/bash",
        env=env,
    )


def capture(cmd: str, cwd: Path | None = None) -> str:
    return subprocess.check_output(
        cmd,
        shell=True,
        cwd=str(cwd) if cwd else None,
        executable="/bin/bash",
        text=True,
    ).strip()


def print_file(path: Path, max_chars: int = 26000) -> None:
    print("\n" + "=" * 104)
    print("AUDIT:", path)
    print("=" * 104)
    if not path.exists():
        print("[MISSING]", path)
        return
    if path.suffix.lower() == ".csv":
        import pandas as pd

        try:
            print(pd.read_csv(path).to_string(index=False)[:max_chars])
        except Exception:
            print(path.read_text(encoding="utf-8", errors="replace")[:max_chars])
    else:
        print(path.read_text(encoding="utf-8", errors="replace")[:max_chars])


def require_files(paths: Iterable[Path], label: str) -> None:
    missing = [str(Path(p)) for p in paths if not Path(p).exists()]
    if missing:
        raise RuntimeError(f"{label}: missing required output(s):\n" + "\n".join(missing))


def load_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _finite_corr(a, b) -> float:
    import numpy as np

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    if int(m.sum()) < 3:
        return float("nan")
    aa = a[m]
    bb = b[m]
    if float(np.std(aa)) < 1e-12 or float(np.std(bb)) < 1e-12:
        return float("nan")
    return float(np.corrcoef(aa, bb)[0, 1])


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _copy_tree_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, dst)


# ------------------------------------------------------------
# 3. RELEASE / FAIRNESS / STATISTICS / ABLATION VERIFIERS
# ------------------------------------------------------------
def verify_release_audit(repo: Path) -> dict:
    """Verify audited repository bytes, excluding only notebook-driver wrappers."""
    audit_path = repo / "RELEASE_AUDIT.json"
    if not audit_path.exists():
        raise RuntimeError("RELEASE_AUDIT.json is missing from the corrected v1.0.0 repository.")

    audit = load_json(audit_path)
    release_name = str(audit.get("release", ""))
    if EXPECTED_VERSION not in release_name:
        raise RuntimeError(
            f"Release audit does not describe SafeGrip {EXPECTED_VERSION}. Found: {release_name!r}"
        )

    missing: list[str] = []
    bad_hash: list[dict] = []
    skipped: list[str] = []
    checked = 0

    for row in audit.get("files", []):
        rel = str(row.get("path", ""))
        expected = str(row.get("sha256", ""))
        if not rel:
            continue
        if rel in RELEASE_AUDIT_DRIVER_EXCLUSIONS:
            skipped.append(rel)
            continue
        p = repo / rel
        if not p.exists():
            missing.append(rel)
            continue
        actual = _sha256(p)
        checked += 1
        if expected and actual != expected:
            bad_hash.append({"path": rel, "expected": expected, "actual": actual})

    report = {
        "status": "PASS" if not missing and not bad_hash else "FAIL",
        "release": release_name,
        "checked_files": checked,
        "skipped_driver_files": skipped,
        "missing": missing,
        "bad_hash": bad_hash,
    }
    print("\nRELEASE HASH AUDIT:\n", json.dumps(report, indent=2))
    if report["status"] != "PASS":
        raise RuntimeError(
            "Static release audit failed. Do not run scientific experiments from altered core bytes."
        )
    return report


def verify_tuning_endpoint_parity(
    proposal_dir: Path,
    baseline_dir: Path,
    out_path: Path,
) -> dict:
    """Require proposal and every tuned baseline to use identical validation endpoints."""
    proposal_manifest = proposal_dir / "tuning_endpoint_manifest.json"
    require_files([proposal_manifest], "proposal tuning endpoint parity")
    p = load_json(proposal_manifest)

    baseline_manifests = sorted(baseline_dir.glob("*/tuning_endpoint_manifest.json"))
    if not baseline_manifests:
        raise RuntimeError("No baseline tuning endpoint manifests were produced.")

    rows = []
    all_ok = True
    for manifest in baseline_manifests:
        b = load_json(manifest)
        same_eval = int(b.get("eval_start", -1)) == int(p.get("eval_start", -2))
        same_n = int(b.get("n", -1)) == int(p.get("n", -2))
        same_hash = str(b.get("sha256")) == str(p.get("sha256"))
        ok = same_eval and same_n and same_hash
        rows.append(
            {
                "baseline": manifest.parent.name,
                "same_eval_start": same_eval,
                "same_n": same_n,
                "same_sha256": same_hash,
                "pass": bool(ok),
            }
        )
        all_ok &= ok

    report = {
        "status": "PASS" if all_ok else "FAIL",
        "proposal_eval_start": p.get("eval_start"),
        "proposal_n": p.get("n"),
        "proposal_sha256": p.get("sha256"),
        "baselines": rows,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\nTUNING ENDPOINT PARITY:\n", json.dumps(report, indent=2))

    if not all_ok:
        raise RuntimeError(
            "Proposal/baseline tuning endpoint parity FAILED. "
            "Do not continue to the final paper comparison."
        )
    return report


def verify_fairness_outputs(result_dir: Path, expect_feature_parity: bool = True) -> dict:
    """Check mandatory benchmark fairness outputs."""
    required = [
        result_dir / "fairness_audit.json",
        result_dir / "evaluation_protocol.json",
        result_dir / "projection_control_metrics.csv",
        result_dir / "label_budget_parity_metrics.csv",
        result_dir / "common_conformal_uq_metrics.csv",
    ]
    if expect_feature_parity:
        required.append(result_dir / "feature_parity_metrics.csv")
    require_files(required, "v1.0 fairness outputs")

    audit = load_json(result_dir / "fairness_audit.json")
    print("\nFAIRNESS AUDIT:\n", json.dumps(audit, indent=2))
    if audit.get("status") != "PASS":
        raise RuntimeError("Scientific fairness audit did not pass.")
    return audit


def verify_statistics_outputs(result_dir: Path) -> dict:
    """Require v1.0 per-seed predictions and hierarchical statistical inference."""
    required = [
        result_dir / "predictions_by_seed.csv",
        result_dir / "prediction_manifest.json",
        result_dir / "statistics" / "paired_bootstrap_rmse.csv",
        result_dir / "statistics" / "statistical_protocol.json",
    ]
    require_files(required, "v1.0 statistical outputs")

    protocol = load_json(result_dir / "statistics" / "statistical_protocol.json")
    method = str(protocol.get("method", "")).lower()
    source = str(protocol.get("source", ""))
    estimand = str(protocol.get("estimand", ""))

    checks = {
        "uses_per_seed_predictions": source == "predictions_by_seed.csv",
        "hierarchical_or_segment_blocked": (
            "hierarchical" in method or ("segment" in method and "bootstrap" in method)
        ),
        "matched_seed_estimand": "per-seed" in estimand.lower(),
    }
    report = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "protocol": protocol,
    }
    print("\nSTATISTICAL PROTOCOL AUDIT:\n", json.dumps(report, indent=2))
    if report["status"] != "PASS":
        raise RuntimeError(
            "Statistical protocol does not match the v1.0 matched-seed/trajectory inference contract."
        )
    return report


def verify_primary_ablation(
    ablation_dir: Path,
    expected_seed_count: int | None = None,
    expected_variants: list[str] | None = None,
) -> tuple[list[str], dict]:
    """Require requested v1.0 ablations and validate semantic/runtime distinctness."""
    import numpy as np
    import pandas as pd

    metrics_path = ablation_dir / "ablation_metrics.csv"
    seed_path = ablation_dir / "ablation_metrics_by_seed.csv"
    design_path = ablation_dir / "ablation_design.json"
    pred_path = ablation_dir / "ablation_predictions.csv"
    require_files([metrics_path, seed_path, design_path, pred_path], "SafeGrip-CI v1.0 ablation")

    expected_variants = list(expected_variants or PRIMARY_ABLATIONS)
    point_distinct = [v for v in POINT_DISTINCT_ABLATIONS if v in expected_variants]

    df = pd.read_csv(metrics_path)
    models = set(df["model"].astype(str))
    missing = [v for v in expected_variants if v not in models]
    if missing:
        raise RuntimeError("Primary v1.0 ablation is incomplete. Missing: " + ", ".join(missing))

    design = load_json(design_path)
    specs = design.get("semantic_specs", {})
    missing_specs = [v for v in expected_variants if v not in specs]
    if missing_specs:
        raise RuntimeError("Ablation semantic specs are missing: " + ", ".join(missing_specs))

    core_specs: dict[str, str] = {}
    duplicate_specs: list[list[str]] = []
    for v in point_distinct:
        spec = dict(specs[v])
        spec.pop("canonical_variant", None)
        # Compatibility/derived fields should not make two scientifically
        # identical core variants look different.
        for key in [
            "raw_features_only",
            "use_gate",
            "use_excitation_regularizer",
            "use_excitation_uq_inflation",
        ]:
            spec.pop(key, None)
        signature = json.dumps(spec, sort_keys=True)
        if signature in core_specs:
            duplicate_specs.append([core_specs[signature], v])
        else:
            core_specs[signature] = v

    # Runtime guard. Two point-path variants should not be bitwise identical
    # across all exported ensemble endpoints. We use exact equality only to
    # catch true aliases/no-op implementations, not numerical similarity.
    pred = pd.read_csv(pred_path)
    duplicate_predictions: list[list[str]] = []
    for i, a in enumerate(point_distinct):
        for b in point_distinct[i + 1 :]:
            if a not in pred.columns or b not in pred.columns:
                continue
            xa = pred[a].to_numpy(float)
            xb = pred[b].to_numpy(float)
            if len(xa) == len(xb) and len(xa) > 0 and np.array_equal(xa, xb):
                duplicate_predictions.append([a, b])

    seed_ok = True
    seed_counts: dict[str, int] = {}
    if expected_seed_count is not None:
        by_seed = pd.read_csv(seed_path)
        if "seed" not in by_seed.columns:
            seed_ok = False
        else:
            seed_counts = {
                str(k): int(v)
                for k, v in by_seed.groupby("model")["seed"].nunique().to_dict().items()
            }
            seed_ok = all(
                int(seed_counts.get(v, 0)) >= int(expected_seed_count)
                for v in expected_variants
            )

    report = {
        "status": (
            "PASS"
            if not duplicate_specs and not duplicate_predictions and seed_ok
            else "FAIL"
        ),
        "all_requested_variants_present": True,
        "requested_variants": expected_variants,
        "all_semantic_specs_present": True,
        "duplicate_core_semantic_specs": duplicate_specs,
        "bitwise_identical_core_prediction_pairs": duplicate_predictions,
        "seed_counts": seed_counts,
        "expected_seed_count": expected_seed_count,
        "key_scientific_comparisons": [
            "safegrip_excitation_proxy vs safegrip",
            "safegrip_no_acceptance vs safegrip",
            "safegrip_no_cf_agreement vs safegrip",
            "safegrip_single_scale_cf vs safegrip",
            "safegrip_no_linearity_consistency vs safegrip",
            "safegrip_no_agreement_veto vs safegrip",
            "safegrip_no_counterfactual_ranking vs safegrip",
        ],
    }
    out_path = ablation_dir / "ablation_semantic_audit_v100.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\nABLATION SEMANTIC AUDIT:\n", json.dumps(report, indent=2))

    if report["status"] != "PASS":
        raise RuntimeError(
            "SafeGrip-CI v1.0 ablation semantic audit FAILED. "
            "Do not interpret the ablation table until duplicate/no-op paths are fixed."
        )
    return sorted(models), report


# ------------------------------------------------------------
# 4. PROPOSAL-SPECIFIC DIAGNOSTICS
# ------------------------------------------------------------
def write_ci_diagnostics(result_dir: Path, ablation_dir: Path | None = None) -> dict:
    """Expose the v0.8 failure modes and the v1.0 authority redesign directly."""
    import numpy as np
    import pandas as pd

    pred_path = result_dir / "predictions.csv"
    metrics_path = result_dir / "metrics.csv"
    require_files([pred_path, metrics_path], "SafeGrip-CI v1.0 diagnostics")

    p = pd.read_csv(pred_path)
    y = p["y_true"].to_numpy(float)
    final = p["safegrip"].to_numpy(float)
    prior = p["safegrip_prior"].to_numpy(float) if "safegrip_prior" in p else final.copy()
    candidate = (
        p["safegrip_candidate"].to_numpy(float)
        if "safegrip_candidate" in p
        else final.copy()
    )

    def col(name: str, default: float = float("nan")):
        key = "safegrip_" + name
        return p[key].to_numpy(float) if key in p else np.full_like(y, default, dtype=float)

    authority = col("reliability")
    ident = col("identifiability")
    trust = col("acceptance")
    veto = col("veto_probability", 0.0)
    norm_improve = col("normalized_improvement", 0.0)
    cf_delta = col("counterfactual_delta_mu", 0.0)
    cf_agreement = col("counterfactual_agreement", 0.5)
    info_raw = col("information_raw", 0.0)
    info_scale_cv = col("information_scale_cv", 0.0)
    local_linearity = col("local_linearity", 1.0)
    agreement_veto = col("agreement_veto_probability", 0.0)
    persistent = col("persistent_state_used", 0.0)

    needed = y - prior
    accepted_update = final - prior
    candidate_update = candidate - prior

    target_std = float(np.std(y))
    pred_std = float(np.std(final))
    prior_rmse = float(np.sqrt(np.mean((y - prior) ** 2)))
    candidate_rmse = float(np.sqrt(np.mean((y - candidate) ** 2)))
    final_rmse = float(np.sqrt(np.mean((y - final) ** 2)))

    report = {
        "n_endpoints": int(len(y)),
        "target_mean": float(np.mean(y)),
        "target_std": target_std,
        "prediction_mean": float(np.mean(final)),
        "prediction_std": pred_std,
        "prediction_std_over_target_std": float(pred_std / max(target_std, 1e-12)),
        "target_prediction_correlation": _finite_corr(y, final),
        "prior_rmse": prior_rmse,
        "candidate_rmse": candidate_rmse,
        "final_rmse": final_rmse,
        "final_minus_prior_rmse": final_rmse - prior_rmse,
        "candidate_minus_prior_rmse": candidate_rmse - prior_rmse,
        "accepted_update_vs_needed_correlation": _finite_corr(accepted_update, needed),
        "candidate_update_vs_needed_correlation": _finite_corr(candidate_update, needed),
        "fraction_final_improves_over_prior": float(
            np.mean(np.abs(y - final) < np.abs(y - prior))
        ),
        "fraction_candidate_improves_over_prior": float(
            np.mean(np.abs(y - candidate) < np.abs(y - prior))
        ),
        "mean_abs_accepted_update": float(np.mean(np.abs(accepted_update))),
        "mean_abs_candidate_update": float(np.mean(np.abs(candidate_update))),
        "authority_mean": float(np.nanmean(authority)),
        "authority_std": float(np.nanstd(authority)),
        "identifiability_mean": float(np.nanmean(ident)),
        "identifiability_std": float(np.nanstd(ident)),
        "trust_multiplier_mean": float(np.nanmean(trust)),
        "trust_multiplier_std": float(np.nanstd(trust)),
        "veto_probability_mean": float(np.nanmean(veto)),
        "veto_probability_std": float(np.nanstd(veto)),
        "normalized_improvement_mean": float(np.nanmean(norm_improve)),
        "normalized_improvement_std": float(np.nanstd(norm_improve)),
        "counterfactual_delta_mu_mean": float(np.nanmean(cf_delta)),
        "counterfactual_delta_mu_std": float(np.nanstd(cf_delta)),
        "counterfactual_agreement_mean": float(np.nanmean(cf_agreement)),
        "counterfactual_agreement_std": float(np.nanstd(cf_agreement)),
        "information_raw_mean": float(np.nanmean(info_raw)),
        "information_scale_cv_mean": float(np.nanmean(info_scale_cv)),
        "information_scale_cv_std": float(np.nanstd(info_scale_cv)),
        "local_linearity_mean": float(np.nanmean(local_linearity)),
        "local_linearity_std": float(np.nanstd(local_linearity)),
        "agreement_veto_probability_mean": float(np.nanmean(agreement_veto)),
        "agreement_veto_probability_std": float(np.nanstd(agreement_veto)),
        "persistent_state_use_rate": float(np.nanmean(persistent)),
        "interpretation": {
            "candidate_update_vs_needed_correlation": (
                "tests whether the learned raw candidate innovation points toward the required correction"
            ),
            "accepted_update_vs_needed_correlation": (
                "tests whether counterfactual authority preserves useful update direction; materially negative is a failure mode"
            ),
            "trust_multiplier": (
                "v1.0 asymmetric trust should stay near 1 under neutral evidence and decrease mainly for harmful candidates"
            ),
            "fraction_final_improves_over_prior": (
                "fraction of endpoints where the authorized update reduces absolute error relative to the persistent prior"
            ),
        },
    }

    if ablation_dir is not None and (ablation_dir / "ablation_metrics.csv").exists():
        a = pd.read_csv(ablation_dir / "ablation_metrics.csv")
        table = a.set_index(a["model"].astype(str))
        if "safegrip" in table.index:
            full = table.loc["safegrip"]
            comparisons = {}
            for name in [
                "safegrip_excitation_proxy",
                "safegrip_no_acceptance",
                "safegrip_no_cf_agreement",
                "safegrip_single_scale_cf",
                "safegrip_no_linearity_consistency",
                "safegrip_no_agreement_veto",
                "safegrip_no_counterfactual_ranking",
                "safegrip_neural_innovation",
                "safegrip_no_identifiability",
            ]:
                if name in table.index:
                    other = table.loc[name]
                    comparisons[name] = {
                        "full_safegrip_rmse": float(full["rmse"]),
                        "variant_rmse": float(other["rmse"]),
                        "rmse_full_minus_variant": float(full["rmse"] - other["rmse"]),
                        "full_safegrip_r2": float(full["r2"]),
                        "variant_r2": float(other["r2"]),
                        "r2_full_minus_variant": float(full["r2"] - other["r2"]),
                        "full_better_rmse": bool(float(full["rmse"]) < float(other["rmse"])),
                    }
            report["primary_ablation_comparisons"] = comparisons

    out = result_dir / "ci_diagnostics_v100.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\nSAFEGRIP-CI v1.0 DIAGNOSTICS:\n", json.dumps(report, indent=2))
    return report


# ------------------------------------------------------------
# 5. RESULT PACKAGING
# ------------------------------------------------------------
def stage_common_metadata(
    stage: Path,
    source_kind: str,
    safegrip_module,
    git_commit: str,
    release_hash_audit: dict,
) -> None:
    """Copy reproducibility/scientific metadata into the exported result archive."""
    stage.mkdir(parents=True, exist_ok=True)

    for rel in [
        "RELEASE_AUDIT.json",
        "SAFEGRIP_CI_METHOD.md",
        "RELATED_WORK_V100.md",
        "PROPOSAL_IMPROVEMENT_REPORT.md",
        "RELATED_WORK_V090.md",
        "SAFEGRIP_V3_METHOD.md",
        "SAFEGRIP_V2_METHOD.md",
        "RESEARCH_PROTOCOL.md",
        "LITERATURE_BASELINES.md",
        "ABLATION_AND_TUNING.md",
        "FAIRNESS_AND_ABLATION_V2.md",
        "IMPLEMENTED_IMPROVEMENTS.md",
        "FINAL_RESEARCH_RELEASE.md",
        "SCIENTIFIC_VALIDITY_FIX.md",
        "VALIDATION.md",
    ]:
        src = REPO / rel
        if src.exists():
            shutil.copy2(src, stage / rel)

    import torch

    env_info = {
        "safegrip_version": safegrip_module.__version__,
        "git_commit": git_commit,
        "git_branch": GITHUB_BRANCH,
        "github_repo": GITHUB_REPO,
        "python": sys.version,
        "torch": torch.__version__,
        "cuda": bool(torch.cuda.is_available()),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "source_kind": source_kind,
        "mode": MODE,
        "paper_trials": PAPER_TRIALS,
        "baseline_trials": BASELINE_TRIALS,
        "trust_bootstrap_replicates": TRUST_BOOTSTRAP_REPLICATES,
        "paper_bootstrap_replicates": PAPER_BOOTSTRAP_REPLICATES,
        "primary_ablations": PRIMARY_ABLATIONS,
        "supplementary_ablations": SUPPLEMENTARY_ABLATIONS,
        "paper_ablations": PAPER_ABLATIONS,
        "run_sensitivity": RUN_SENSITIVITY,
        "sensitivity_parameters": SENSITIVITY_PARAMETERS,
        "run_retuned_ablation": RUN_RETUNED_ABLATION,
        "run_cross_route": RUN_CROSS_ROUTE,
        "run_scarcity": RUN_SCARCITY,
        "release_hash_audit": release_hash_audit,
        "proposal": (
            "SafeGrip-CI v1.0: persistent state + supervised raw innovation + "
            "multi-scale counterfactual observability + cross-scale linearity discount + "
            "residual veto + identifiability-aware inverse-dynamics agreement veto"
        ),
        "statistics": "paired hierarchical bootstrap over matched seeds and trajectory segments",
    }
    (stage / "run_environment.json").write_text(
        json.dumps(env_info, indent=2), encoding="utf-8"
    )


def make_review_or_release_zip(
    target_zip: Path,
    folders: Iterable[Path],
    diagnostics_dir: Path | None,
    config_path: Path | None,
    source_kind: str,
    safegrip_module,
    git_commit: str,
    release_hash_audit: dict,
    extra_files: Iterable[Path] = (),
) -> Path:
    stage = WORK / ("_stage_" + target_zip.stem)
    shutil.rmtree(stage, ignore_errors=True)
    stage.mkdir(parents=True)

    for folder in folders:
        src = REPO / folder
        if src.exists():
            dst = stage / folder
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(src, dst)

    if diagnostics_dir is not None and diagnostics_dir.exists():
        dst = stage / "diagnostics"
        dst.mkdir(parents=True, exist_ok=True)
        for p in diagnostics_dir.iterdir():
            if p.is_file() and p.name.startswith("lira_"):
                shutil.copy2(p, dst / p.name)

    if config_path is not None and config_path.exists():
        shutil.copy2(config_path, stage / config_path.name)

    for extra in extra_files:
        p = Path(extra)
        if p.exists() and p.is_file():
            shutil.copy2(p, stage / p.name)

    stage_common_metadata(
        stage,
        source_kind,
        safegrip_module,
        git_commit,
        release_hash_audit,
    )

    if target_zip.exists():
        target_zip.unlink()
    shutil.make_archive(str(target_zip.with_suffix("")), "zip", root_dir=stage)
    shutil.rmtree(stage)
    return target_zip


# ------------------------------------------------------------
# 6. CLEAN + LOAD EXACT UPDATED v1.0 REPOSITORY
# ------------------------------------------------------------
if MODE not in {"trust", "paper"}:
    raise ValueError('MODE must be "trust" or "paper".')
if SOURCE_MODE not in {"github", "kaggle_zip"}:
    raise ValueError('SOURCE_MODE must be "github" or "kaggle_zip".')

shutil.rmtree(REPO, ignore_errors=True)
for p in [TRUST_ZIP, PAPER_ZIP, PAPER_REVIEW_ZIP]:
    if p.exists():
        p.unlink()

if SOURCE_MODE == "github":
    run(
        f"git clone --depth 1 --branch {GITHUB_BRANCH} {GITHUB_REPO} {REPO}",
        WORK,
    )
    SOURCE_KIND = f"github:{GITHUB_BRANCH}"
    GIT_COMMIT = capture("git rev-parse HEAD", REPO)
else:
    if KAGGLE_ZIP_PATH:
        archive = Path(KAGGLE_ZIP_PATH)
        candidates = [archive]
    else:
        candidates = sorted(Path("/kaggle/input").rglob("quynh-method-safegrip-v1.0-full.zip"))
        if not candidates:
            candidates = sorted(Path("/kaggle/input").rglob("*safegrip*v1.0*.zip"))
    if not candidates or not candidates[0].exists():
        raise FileNotFoundError(
            "SOURCE_MODE='kaggle_zip' but no SafeGrip-CI v1.0 ZIP was found under /kaggle/input. "
            "Upload quynh-method-safegrip-v1.0-full.zip as a Kaggle Dataset or set KAGGLE_ZIP_PATH."
        )
    archive = candidates[0]
    extract_dir = WORK / "_safegrip_v100_extract"
    shutil.rmtree(extract_dir, ignore_errors=True)
    extract_dir.mkdir(parents=True, exist_ok=True)
    shutil.unpack_archive(str(archive), str(extract_dir))
    roots = [p.parent for p in extract_dir.rglob("pyproject.toml") if (p.parent / "src/safegrip").exists()]
    if len(roots) != 1:
        raise RuntimeError(f"Expected exactly one SafeGrip repository in {archive}; found {len(roots)}")
    shutil.copytree(roots[0], REPO)
    SOURCE_KIND = f"kaggle_zip:{archive.name}"
    GIT_COMMIT = "archive-" + _sha256(archive)[:16]

print("\nRepository:", REPO)
print("Source:", SOURCE_KIND)
print("Source revision:", GIT_COMMIT)

# ------------------------------------------------------------
# 7. FORCE IMPORT FROM JUST-CLONED REPOSITORY
# ------------------------------------------------------------
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

existing_pythonpath = os.environ.get("PYTHONPATH", "")
os.environ["PYTHONPATH"] = (
    str(SRC)
    if not existing_pythonpath
    else str(SRC) + os.pathsep + existing_pythonpath
)
os.chdir(REPO)

# ------------------------------------------------------------
# 8. INSTALL + VERIFY VERSION / RELEASE / TESTS
# ------------------------------------------------------------
run(f"{sys.executable} -m pip install -q --upgrade pip setuptools wheel", REPO)
# Install both extras so the same cell can switch trust -> paper safely.
run(f'{sys.executable} -m pip install -q -e ".[paper,dev]"', REPO)

importlib.invalidate_caches()
import safegrip
import numpy as np
import pandas as pd
import torch
import yaml

print("\nSafeGrip imported from:", safegrip.__file__)
print("SafeGrip version:", safegrip.__version__)
print("Git commit:", GIT_COMMIT)
print("Python:", sys.version.split()[0])
print("PyTorch:", torch.__version__)
print("CUDA:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

if safegrip.__version__ != EXPECTED_VERSION:
    raise RuntimeError(
        "\nWRONG REPOSITORY VERSION.\n\n"
        f"Found SafeGrip {safegrip.__version__}, but this workflow requires "
        f"SafeGrip {EXPECTED_VERSION}.\n"
        f"Repository commit: {GIT_COMMIT}\n\n"
        "Push/merge the corrected SafeGrip-CI v1.0.0 files to the configured GitHub branch, "
        "or set SOURCE_MODE='kaggle_zip' and upload the corrected full v1.0 ZIP."
    )

print_file(REPO / "RELEASE_AUDIT.json", max_chars=18000)
release_hash_audit = verify_release_audit(REPO)

run(f"{sys.executable} -m pytest -q", REPO)

# Fail early if Kaggle is using a stale CLI even when a local import was shadowed.
cli_help = capture(f"{sys.executable} -m safegrip.cli --help", REPO)
for required_cmd in ["benchmark", "ablation", "tune", "tune-baselines", "sensitivity", "statistics", "experiment"]:
    if required_cmd not in cli_help:
        raise RuntimeError(f"Required SafeGrip-CI v1.0 CLI command is missing: {required_cmd}")

# ============================================================
# 9. TRUST MODE
# ============================================================
if MODE == "trust":
    # Explicitly enable every required fairness control even if an older/user-
    # edited trust YAML omitted one. Do not alter proposal hyperparameters here.
    base_cfg_path = REPO / "configs/kaggle_trust.yaml"
    trust_cfg = yaml.safe_load(base_cfg_path.read_text(encoding="utf-8")) or {}
    trust_eval = trust_cfg.setdefault("evaluation", {})
    trust_eval["projection_parity_controls"] = True
    trust_eval["feature_parity_controls"] = True
    trust_eval["feature_parity_models"] = ["lampe2023_gru"]
    trust_eval["label_budget_parity_controls"] = True
    trust_eval["common_conformal_controls"] = True

    runtime_cfg = REPO / "configs/kaggle_trust_runtime_v100.yaml"
    runtime_cfg.write_text(
        yaml.safe_dump(trust_cfg, sort_keys=False),
        encoding="utf-8",
    )

    CONFIG = str(runtime_cfg.relative_to(REPO))
    SG = f"{sys.executable} -m safegrip.cli --config {CONFIG}"

    print("\n" + "#" * 112)
    print("RUNNING SAFEGRIP-CI v1.0.0 TRUST EXPERIMENT")
    print("#" * 112)

    run(f"{SG} download --datasets lira", REPO)
    run(f"{SG} prepare --dataset lira", REPO)

    proc = REPO / "data" / "processed" / "lira"
    diagnostic_files = [
        "lira_signal_audit.csv",
        "lira_physics_audit.json",
        "lira_friction_schema_report.csv",
        "lira_stream_assembly_report.json",
        "lira_alignment_report.csv",
        "lira_preprocessing_report.json",
    ]
    for name in diagnostic_files:
        print_file(proc / name)

    # Proposal + two literature comparators.
    run(
        f"{SG} benchmark --dataset lira --preset trust "
        "--models todorovic2022_cnn,lampe2023_gru",
        REPO,
    )

    trust_dir = REPO / "results" / "lira_trust"

    # v1.0 statistical inference: matched seeds + trajectory/segment blocks.
    run(
        f"{SG} statistics --results results/lira_trust "
        f"--bootstrap {TRUST_BOOTSTRAP_REPLICATES}",
        REPO,
    )

    # Controlled SafeGrip-CI concept ablation.
    run(
        f"{SG} ablation --dataset lira --preset trust "
        f"--variants {PRIMARY_ABLATIONS_CSV}",
        REPO,
    )

    ablation_dir = REPO / "results" / "lira_ablation_trust"

    require_files(
        [
            trust_dir / "result_health.json",
            trust_dir / "metrics.csv",
            trust_dir / "metrics_by_seed.csv",
            trust_dir / "predictions.csv",
            trust_dir / "predictions_by_seed.csv",
            trust_dir / "prediction_manifest.json",
            trust_dir / "proposal_reliability.json",
            trust_dir / "proposal_uq.json",
            trust_dir / "fairness_audit.json",
            trust_dir / "statistics" / "paired_bootstrap_rmse.csv",
            trust_dir / "statistics" / "statistical_protocol.json",
        ],
        "trust benchmark",
    )

    health = load_json(trust_dir / "result_health.json")
    fairness = verify_fairness_outputs(trust_dir, expect_feature_parity=True)
    statistics_audit = verify_statistics_outputs(trust_dir)
    ablation_models, ablation_semantic = verify_primary_ablation(
        ablation_dir,
        expected_seed_count=3,
        expected_variants=PRIMARY_ABLATIONS,
    )
    ci_diag = write_ci_diagnostics(trust_dir, ablation_dir)

    print("\n" + "=" * 112)
    print("SCIENTIFIC HEALTH")
    print("=" * 112)
    print(json.dumps(health, indent=2))

    for title, p in [
        ("MAIN TRUST METRICS", trust_dir / "metrics.csv"),
        ("PER-SEED TRUST METRICS", trust_dir / "metrics_by_seed.csv"),
        ("TRAIN-ONLY SANITY BASELINES", trust_dir / "sanity_baselines.csv"),
        ("FEATURE PARITY", trust_dir / "feature_parity_metrics.csv"),
        ("EQUAL LABEL BUDGET", trust_dir / "label_budget_parity_metrics.csv"),
        ("COMMON CONFORMAL UQ", trust_dir / "common_conformal_uq_metrics.csv"),
        ("PROJECTION PARITY", trust_dir / "projection_control_metrics.csv"),
        (
            "HIERARCHICAL PAIRED BOOTSTRAP",
            trust_dir / "statistics" / "paired_bootstrap_rmse.csv",
        ),
        ("PRIMARY 15-VARIANT SafeGrip-CI v1.0 ABLATION", ablation_dir / "ablation_metrics.csv"),
        ("ABLATION BY SEED", ablation_dir / "ablation_metrics_by_seed.csv"),
    ]:
        print("\n" + "=" * 112)
        print(title)
        print("=" * 112)
        print_file(p)

    trust_pass = (
        health.get("status") == "PASS"
        and fairness.get("status") == "PASS"
        and statistics_audit.get("status") == "PASS"
        and ablation_semantic.get("status") == "PASS"
        and set(PRIMARY_ABLATIONS).issubset(set(ablation_models))
    )

    review_gate = {
        "status": "PASS" if trust_pass else "REVIEW",
        "scientific_health": health.get("status"),
        "fairness_audit": fairness.get("status"),
        "statistical_protocol_audit": statistics_audit.get("status"),
        "ablation_semantic_audit": ablation_semantic.get("status"),
        "primary_ablations_complete": set(PRIMARY_ABLATIONS).issubset(
            set(ablation_models)
        ),
        "release_hash_audit": release_hash_audit.get("status"),
        "git_commit": GIT_COMMIT,
        "version": safegrip.__version__,
        "ci_diagnostics_file": "ci_diagnostics_v100.json",
        "note": (
            "PASS clears automatic trust/fairness/statistics/ablation gates. "
            "It is not itself a novelty claim or external-validation result."
        ),
    }
    trust_gate_path = trust_dir / "trust_gate_v100.json"
    trust_gate_path.write_text(json.dumps(review_gate, indent=2), encoding="utf-8")
    print("\nTRUST v1.0 GATE:\n", json.dumps(review_gate, indent=2))

    # Always export diagnostics/results, even when status is REVIEW.
    make_review_or_release_zip(
        TRUST_ZIP,
        folders=[
            Path("results/lira_trust"),
            Path("results/lira_ablation_trust"),
        ],
        diagnostics_dir=proc,
        config_path=runtime_cfg,
        source_kind=SOURCE_KIND,
        safegrip_module=safegrip,
        git_commit=GIT_COMMIT,
        release_hash_audit=release_hash_audit,
    )

    print("\nTrust result ZIP:", TRUST_ZIP)
    try:
        from IPython.display import FileLink, display

        display(FileLink(str(TRUST_ZIP)))
    except Exception:
        pass

    if not trust_pass:
        raise RuntimeError(
            "\nTRUST RUN = REVIEW\n\n"
            "The run completed and the ZIP was exported, but at least one v1.0 "
            "scientific/fairness/statistics/ablation gate failed. Do not report "
            "these values as final paper results. Inspect result_health.json, "
            "ci_diagnostics_v100.json, statistical_protocol.json, and "
            "ablation_semantic_audit_v100.json first."
        )

    print("\n" + "#" * 112)
    print("TRUST RUN PASSED")
    print("#" * 112)
    print('\nNext: set MODE = "paper" and rerun the complete cell.')

# ============================================================
# 10. PAPER MODE
# ============================================================
elif MODE == "paper":
    CONFIG = "configs/default.yaml"
    SG = f"{sys.executable} -m safegrip.cli --config {CONFIG}"

    print("\n" + "#" * 112)
    print("RUNNING SAFEGRIP-CI v1.0.0 FULL PAPER WORKFLOW")
    print("#" * 112)

    # 10A. Download + prepare real LiRA.
    run(f"{SG} download --datasets lira", REPO)
    run(f"{SG} prepare --dataset lira", REPO)

    proc = REPO / "data" / "processed" / "lira"
    for name in [
        "lira_signal_audit.csv",
        "lira_physics_audit.json",
        "lira_friction_schema_report.csv",
        "lira_stream_assembly_report.json",
        "lira_alignment_report.csv",
        "lira_preprocessing_report.json",
    ]:
        print_file(proc / name)

    # 10B. Validation-only proposal tuning. Test is locked.
    run(
        f"{SG} tune --dataset lira --trials {PAPER_TRIALS} --no-test",
        REPO,
    )

    # 10C. Every literature comparator receives the same search-trial budget.
    run(
        f"{SG} tune-baselines --dataset lira --trials {BASELINE_TRIALS}",
        REPO,
    )

    proposal_tune_dir = REPO / "results" / "lira_tuning"
    baseline_tune_dir = REPO / "results" / "lira_baseline_tuning"

    tuning_parity = verify_tuning_endpoint_parity(
        proposal_tune_dir,
        baseline_tune_dir,
        REPO / "results" / "tuning_endpoint_parity_v100.json",
    )

    proposal_hp = "results/lira_tuning/best_hparams.yaml"
    baseline_hp = "results/lira_baseline_tuning/best_hparams.yaml"

    # v1.0 reviewer-facing stability analysis. This stays on locked validation
    # endpoints and does not touch test labels.
    sensitivity_dir = REPO / "results" / "lira_hyperparameter_sensitivity"
    if RUN_SENSITIVITY:
        sensitivity_args = ""
        if SENSITIVITY_PARAMETERS:
            sensitivity_args += " --parameters " + ",".join(SENSITIVITY_PARAMETERS)
        if SENSITIVITY_EPOCHS is not None:
            sensitivity_args += f" --epochs {int(SENSITIVITY_EPOCHS)}"
        run(
            f"{SG} sensitivity --dataset lira --proposal-hparams {proposal_hp}" + sensitivity_args,
            REPO,
        )
        require_files(
            [
                sensitivity_dir / "hyperparameter_sensitivity.csv",
                sensitivity_dir / "hyperparameter_sensitivity.json",
            ],
            "v1.0 hyperparameter sensitivity",
        )
        sensitivity_protocol = load_json(sensitivity_dir / "hyperparameter_sensitivity.json")
        if bool(sensitivity_protocol.get("test_labels_used", True)):
            raise RuntimeError("Hyperparameter sensitivity protocol touched test labels; abort paper workflow.")

    # 10D. Five-seed final benchmark with the full paper baseline set.
    run(
        f"{SG} benchmark --dataset lira --preset paper "
        f"--proposal-hparams {proposal_hp} "
        f"--baseline-hparams {baseline_hp}",
        REPO,
    )

    paper_dir = REPO / "results" / "lira_paper"
    fairness = verify_fairness_outputs(paper_dir, expect_feature_parity=True)

    # 10E. Correct hierarchical statistics on matched per-seed predictions.
    run(
        f"{SG} statistics --results results/lira_paper "
        f"--bootstrap {PAPER_BOOTSTRAP_REPLICATES}",
        REPO,
    )
    statistics_audit = verify_statistics_outputs(paper_dir)

    # 10F. Controlled/frozen-hyperparameter ablation.
    run(
        f"{SG} ablation --dataset lira --preset paper "
        f"--proposal-hparams {proposal_hp} "
        f"--variants {PAPER_ABLATIONS_CSV}",
        REPO,
    )

    ablation_dir = REPO / "results" / "lira_ablation_paper"
    ablation_models, ablation_semantic = verify_primary_ablation(
        ablation_dir,
        expected_seed_count=5,
        expected_variants=PAPER_ABLATIONS,
    )

    # 10G. Reviewer-oriented frozen-prediction analyses.
    # Excitation remains a diagnostic/legacy-proxy stratification only.
    run(
        f"{SG} experiment --dataset lira --study excitation "
        "--results results/lira_paper",
        REPO,
    )
    run(
        f"{SG} experiment --dataset lira --study robustness "
        "--results results/lira_paper",
        REPO,
    )

    # Proposal-specific diagnostics on the frozen final predictions.
    ci_diag = write_ci_diagnostics(paper_dir, ablation_dir)

    # 10H. Expensive supplementary controls (optional).
    if RUN_RETUNED_ABLATION:
        run(
            f"{SG} tune-ablation --dataset lira "
            f"--variants {PAPER_ABLATIONS_CSV} "
            f"--trials {RETUNED_ABLATION_TRIALS}",
            REPO,
        )

    if RUN_CROSS_ROUTE:
        run(
            f"{SG} experiment --dataset lira --study cross-route "
            f"--preset paper --proposal-hparams {proposal_hp}",
            REPO,
        )

    if RUN_SCARCITY:
        run(
            f"{SG} experiment --dataset lira --study scarcity "
            f"--preset paper --proposal-hparams {proposal_hp}",
            REPO,
        )

    if RUN_EXTENDED_SCRIPT:
        run("bash scripts/run_extended_experiments.sh", REPO)

    if DOWNLOAD_AUX:
        run("bash scripts/download_all_datasets.sh", REPO)

    # --------------------------------------------------------
    # 10I. Strong v1.0 paper-readiness gate
    # --------------------------------------------------------
    required_main = [
        paper_dir / "metrics.csv",
        paper_dir / "metrics_by_seed.csv",
        paper_dir / "predictions.csv",
        paper_dir / "predictions_by_seed.csv",
        paper_dir / "prediction_manifest.json",
        paper_dir / "result_health.json",
        paper_dir / "sanity_baselines.csv",
        paper_dir / "evaluation_protocol.json",
        paper_dir / "reproducibility_manifest.json",
        paper_dir / "baseline_manifest.csv",
        paper_dir / "proposal_hparams.json",
        paper_dir / "proposal_reliability.json",
        paper_dir / "proposal_uq.json",
        paper_dir / "baseline_selected_hparams.json",
        paper_dir / "fairness_audit.json",
        paper_dir / "feature_parity_metrics.csv",
        paper_dir / "label_budget_parity_metrics.csv",
        paper_dir / "common_conformal_uq_metrics.csv",
        paper_dir / "projection_control_metrics.csv",
        paper_dir / "statistics" / "paired_bootstrap_rmse.csv",
        paper_dir / "statistics" / "statistical_protocol.json",
        paper_dir / "ci_diagnostics_v100.json",
    ]

    required_processed = [
        proc / "lira_aligned.csv",
        proc / "lira_signal_audit.csv",
        proc / "lira_physics_audit.json",
        proc / "lira_friction_schema_report.csv",
        proc / "lira_alignment_report.csv",
        proc / "lira_preprocessing_report.json",
    ]

    required_ablation = [
        ablation_dir / "ablation_metrics.csv",
        ablation_dir / "ablation_metrics_by_seed.csv",
        ablation_dir / "ablation_predictions.csv",
        ablation_dir / "ablation_design.json",
        ablation_dir / "ablation_semantic_audit_v100.json",
    ]

    required_analysis = [
        REPO / "results" / "lira_excitation" / "excitation_metrics.csv",
        REPO / "results" / "lira_robustness" / "physics_robustness_metrics.csv",
    ]

    required_tuning = [
        proposal_tune_dir / "best_hparams.yaml",
        proposal_tune_dir / "tuning_summary.json",
        proposal_tune_dir / "tuning_endpoint_manifest.json",
        baseline_tune_dir / "best_hparams.yaml",
        baseline_tune_dir / "tuning_summary.json",
        REPO / "results" / "tuning_endpoint_parity_v100.json",
    ]

    require_files(required_main, "paper main outputs")
    require_files(required_processed, "paper processed outputs")
    require_files(required_ablation, "paper ablation outputs")
    require_files(required_analysis, "paper analysis outputs")
    require_files(required_tuning, "paper tuning outputs")
    if RUN_SENSITIVITY:
        require_files(
            [
                sensitivity_dir / "hyperparameter_sensitivity.csv",
                sensitivity_dir / "hyperparameter_sensitivity.json",
            ],
            "paper hyperparameter sensitivity outputs",
        )

    paper_health = load_json(paper_dir / "result_health.json")

    # Main model/seed completeness.
    main_metrics = pd.read_csv(paper_dir / "metrics_by_seed.csv")
    observed_models = set(main_metrics["model"].astype(str))
    main_seed_counts = (
        main_metrics.groupby("model")["seed"].nunique().to_dict()
        if "seed" in main_metrics.columns
        else {}
    )

    # Primary ablation/seed completeness.
    abl_metrics = pd.read_csv(ablation_dir / "ablation_metrics_by_seed.csv")
    observed_ablations = set(abl_metrics["model"].astype(str))
    abl_seed_counts = (
        abl_metrics.groupby("model")["seed"].nunique().to_dict()
        if "seed" in abl_metrics.columns
        else {}
    )

    # Physics support audit.
    physics_audit = load_json(proc / "lira_physics_audit.json")
    physics_support_ok = float(
        physics_audit.get("bound_above_upper_rate", 1.0)
    ) <= float(physics_audit.get("max_allowed_bound_above_upper_rate", 0.0))

    # Preprocessing/leakage controls.
    prep_report = load_json(proc / "lira_preprocessing_report.json")
    prep = prep_report.get("preprocessing", {})
    preprocessing_ok = all(
        [
            bool(prep.get("split_before_imputation")),
            bool(prep.get("partition_local_imputation")),
            bool(prep.get("trajectory_aware_splitting")),
            bool(prep.get("segment_safe_windows")),
        ]
    )

    v100_checks = {
        "scientific_health_pass": paper_health.get("status") == "PASS",
        "fairness_audit_pass": fairness.get("status") == "PASS",
        "statistical_protocol_pass": statistics_audit.get("status") == "PASS",
        "tuning_endpoint_parity_pass": tuning_parity.get("status") == "PASS",
        "ablation_semantic_audit_pass": ablation_semantic.get("status") == "PASS",
        "release_hash_audit_pass": release_hash_audit.get("status") == "PASS",
        "required_main_outputs_present": all(p.exists() for p in required_main),
        "required_processed_outputs_present": all(p.exists() for p in required_processed),
        "required_ablation_outputs_present": all(p.exists() for p in required_ablation),
        "required_analysis_outputs_present": all(p.exists() for p in required_analysis),
        "required_tuning_outputs_present": all(p.exists() for p in required_tuning),
        "hyperparameter_sensitivity_present_if_requested": (
            (sensitivity_dir / "hyperparameter_sensitivity.csv").exists()
            and (sensitivity_dir / "hyperparameter_sensitivity.json").exists()
            if RUN_SENSITIVITY else True
        ),
        "hyperparameter_sensitivity_validation_only": (
            not bool(load_json(sensitivity_dir / "hyperparameter_sensitivity.json").get("test_labels_used", True))
            if RUN_SENSITIVITY and (sensitivity_dir / "hyperparameter_sensitivity.json").exists()
            else (not RUN_SENSITIVITY)
        ),
        "all_paper_models_present": PAPER_MODELS.issubset(observed_models),
        "five_seeds_each_main_model": all(
            int(main_seed_counts.get(name, 0)) >= 5 for name in PAPER_MODELS
        ),
        "all_paper_v100_ablations_present": set(PAPER_ABLATIONS).issubset(
            observed_ablations
        ),
        "five_seeds_each_paper_ablation": all(
            int(abl_seed_counts.get(name, 0)) >= 5 for name in PAPER_ABLATIONS
        ),
        "physics_support_audit_pass": physics_support_ok,
        "preprocessing_leakage_controls_pass": preprocessing_ok,
        "retuned_ablation_present_if_requested": (
            (REPO / "results/lira_ablation_retuned/retuned_ablation_summary.json").exists()
            if RUN_RETUNED_ABLATION
            else True
        ),
        "cross_route_present_if_requested": (
            (REPO / "results/lira_cross_route").exists() if RUN_CROSS_ROUTE else True
        ),
        "scarcity_present_if_requested": (
            (REPO / "results/lira_scarcity").exists() if RUN_SCARCITY else True
        ),
    }

    v100_status = "PAPER_READY" if all(v100_checks.values()) else "REVIEW"
    v100_gate = {
        "status": v100_status,
        "checks": v100_checks,
        "scientific_health": paper_health,
        "safegrip_version": safegrip.__version__,
        "git_commit": GIT_COMMIT,
        "primary_ablations": PRIMARY_ABLATIONS,
        "supplementary_ablations": SUPPLEMENTARY_ABLATIONS,
        "paper_ablations": PAPER_ABLATIONS,
        "proposal": (
            "SafeGrip-CI v1.0 persistent state + supervised raw innovation + "
            "multi-scale counterfactual observability + local-linearity consistency + "
            "residual veto + inverse-dynamics agreement veto"
        ),
        "key_novelty_controls": [
            "safegrip_excitation_proxy vs safegrip",
            "safegrip_no_acceptance vs safegrip",
            "safegrip_no_cf_agreement vs safegrip",
            "safegrip_single_scale_cf vs safegrip",
            "safegrip_no_linearity_consistency vs safegrip",
            "safegrip_no_agreement_veto vs safegrip",
            "safegrip_no_counterfactual_ranking vs safegrip",
        ],
        "statistics": "paired hierarchical bootstrap over matched seeds and trajectory segments",
        "note": (
            "PAPER_READY means automatic v1.0 scientific-health, fairness, "
            "endpoint-parity, multi-seed, ablation, preprocessing, statistics, "
            "and reproducibility gates passed. It does not prove novelty, "
            "external validity, or guarantee acceptance."
        ),
    }
    paper_gate_path = paper_dir / "paper_readiness_v100.json"
    paper_gate_path.write_text(json.dumps(v100_gate, indent=2), encoding="utf-8")

    # Also execute the repository-maintained readiness checker. It writes
    # results/lira_paper/paper_readiness.json even when it exits REVIEW.
    readiness_proc = run(
        f"{sys.executable} scripts/check_paper_readiness.py",
        REPO,
        check=False,
    )
    repo_readiness_path = paper_dir / "paper_readiness.json"
    repo_readiness = (
        load_json(repo_readiness_path) if repo_readiness_path.exists() else {"status": "MISSING"}
    )
    if repo_readiness.get("status") != "PAPER_READY":
        v100_status = "REVIEW"
        v100_gate["status"] = "REVIEW"
        v100_gate["repository_readiness_status"] = repo_readiness.get("status")
        paper_gate_path.write_text(json.dumps(v100_gate, indent=2), encoding="utf-8")

    print("\n" + "=" * 112)
    print("v1.0 PAPER READINESS / FAIRNESS / STATISTICS / ABLATION GATE")
    print("=" * 112)
    print(json.dumps(v100_gate, indent=2))
    print("\nREPOSITORY READINESS CHECK:\n", json.dumps(repo_readiness, indent=2))

    # --------------------------------------------------------
    # 10J. Print final tables
    # --------------------------------------------------------
    for title, p in [
        ("FINAL MAIN TABLE", paper_dir / "metrics.csv"),
        ("FINAL PER-SEED TABLE", paper_dir / "metrics_by_seed.csv"),
        ("SANITY BASELINES", paper_dir / "sanity_baselines.csv"),
        ("FEATURE PARITY", paper_dir / "feature_parity_metrics.csv"),
        ("EQUAL LABEL BUDGET", paper_dir / "label_budget_parity_metrics.csv"),
        ("COMMON CONFORMAL UQ", paper_dir / "common_conformal_uq_metrics.csv"),
        ("PROJECTION PARITY", paper_dir / "projection_control_metrics.csv"),
        (
            "HIERARCHICAL PAIRED BOOTSTRAP",
            paper_dir / "statistics" / "paired_bootstrap_rmse.csv",
        ),
        ("FINAL CONTROLLED CI ABLATION (15 PRIMARY + 3 TRAINING CONTROLS)", ablation_dir / "ablation_metrics.csv"),
        ("FINAL CI ABLATION BY SEED", ablation_dir / "ablation_metrics_by_seed.csv"),
        *(([("HYPERPARAMETER SENSITIVITY", sensitivity_dir / "hyperparameter_sensitivity.csv")] if RUN_SENSITIVITY else [])),
        (
            "EXCITATION-PROXY STRATIFICATION",
            REPO / "results/lira_excitation/excitation_metrics.csv",
        ),
        (
            "PHYSICS ROBUSTNESS",
            REPO / "results/lira_robustness/physics_robustness_metrics.csv",
        ),
    ]:
        print("\n" + "=" * 112)
        print(title)
        print("=" * 112)
        print_file(p)

    # --------------------------------------------------------
    # 10K. Package. Never label REVIEW outputs as paper release.
    # --------------------------------------------------------
    package_folders = [
        Path("results/lira_paper"),
        Path("results/lira_ablation_paper"),
        Path("results/lira_tuning"),
        Path("results/lira_baseline_tuning"),
        *([Path("results/lira_hyperparameter_sensitivity")] if RUN_SENSITIVITY else []),
        Path("results/lira_excitation"),
        Path("results/lira_robustness"),
    ]
    if RUN_RETUNED_ABLATION:
        package_folders.append(Path("results/lira_ablation_retuned"))
    if RUN_CROSS_ROUTE:
        package_folders.append(Path("results/lira_cross_route"))
    if RUN_SCARCITY:
        package_folders.append(Path("results/lira_scarcity"))

    if v100_status == "PAPER_READY":
        target_zip = PAPER_ZIP
        print("\n" + "#" * 112)
        print("PAPER_READY")
        print("#" * 112)
    else:
        target_zip = PAPER_REVIEW_ZIP
        print("\n" + "#" * 112)
        print("PAPER RESULT = REVIEW")
        print("#" * 112)

    make_review_or_release_zip(
        target_zip,
        folders=package_folders,
        diagnostics_dir=proc,
        config_path=REPO / CONFIG,
        source_kind=SOURCE_KIND,
        safegrip_module=safegrip,
        git_commit=GIT_COMMIT,
        release_hash_audit=release_hash_audit,
        extra_files=[REPO / "results/tuning_endpoint_parity_v100.json"],
    )

    print("\nExported result ZIP:", target_zip)
    try:
        from IPython.display import FileLink, display

        display(FileLink(str(target_zip)))
    except Exception:
        pass

    if v100_status != "PAPER_READY":
        raise RuntimeError(
            "\nPAPER RESULT = REVIEW\n\n"
            "The complete experiment finished and a review ZIP was exported, "
            "but at least one v1.0 scientific/fairness/statistics/semantic/"
            "reproducibility gate did not pass. Do not use these numbers as "
            "final manuscript claims until the failed checks are resolved."
        )

print("\nSafeGrip-CI v1.0.0 Kaggle workflow finished.")

