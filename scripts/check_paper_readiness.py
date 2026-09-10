#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "results" / "lira_paper"
ABL = ROOT / "results" / "lira_ablation_paper"
PROC = ROOT / "data" / "processed" / "lira"
TUNE = ROOT / "results" / "lira_tuning"
BTUNE = ROOT / "results" / "lira_baseline_tuning"

required_main = [
    "metrics.csv", "metrics_by_seed.csv", "predictions.csv", "predictions_by_seed.csv",
    "result_health.json", "sanity_baselines.csv", "evaluation_protocol.json",
    "reproducibility_manifest.json", "baseline_manifest.csv", "proposal_hparams.json",
    "proposal_reliability.json", "proposal_uq.json", "baseline_selected_hparams.json",
    "fairness_audit.json", "feature_parity_metrics.csv", "label_budget_parity_metrics.csv",
    "common_conformal_uq_metrics.csv", "projection_control_metrics.csv",
    "statistics/paired_bootstrap_rmse.csv",
]
required_proc = [
    "lira_aligned.csv", "lira_signal_audit.csv", "lira_physics_audit.json",
    "lira_friction_schema_report.csv", "lira_alignment_report.csv", "lira_preprocessing_report.json",
]
required_abl = [
    "ablation_metrics.csv", "ablation_metrics_by_seed.csv", "ablation_predictions.csv",
    "ablation_design.json",
]
PRIMARY_ABLATIONS = {
    "safegrip_backbone_raw", "safegrip_persistent", "safegrip_neural_innovation",
    "safegrip_no_identifiability", "safegrip_excitation_proxy", "safegrip_no_acceptance",
    "safegrip_no_cf_agreement", "safegrip_no_innovation_supervision", "safegrip_no_bound",
    "safegrip_no_uq", "safegrip",
}
PAPER_MODELS = {
    "todorovic2022_cnn", "lampe2023_lstm", "lampe2023_gru",
    "schaefke2023_transformer", "chen2025_svdkl", "safegrip",
}

checks: dict[str, bool] = {}
for f in required_main:
    checks[f"main:{f}"] = (MAIN / f).exists()
for f in required_proc:
    checks[f"processed:{f}"] = (PROC / f).exists()
for f in required_abl:
    checks[f"ablation:{f}"] = (ABL / f).exists()

health = {}
if (MAIN / "result_health.json").exists():
    health = json.loads((MAIN / "result_health.json").read_text())
    checks["scientific_health_pass"] = health.get("status") == "PASS"
else:
    checks["scientific_health_pass"] = False

if (MAIN / "fairness_audit.json").exists():
    fairness = json.loads((MAIN / "fairness_audit.json").read_text())
    checks["fairness_audit_pass"] = fairness.get("status") == "PASS"
else:
    checks["fairness_audit_pass"] = False

if (MAIN / "metrics_by_seed.csv").exists():
    m = pd.read_csv(MAIN / "metrics_by_seed.csv")
    observed = set(m["model"].astype(str))
    counts = m.groupby("model")["seed"].nunique().to_dict() if "seed" in m else {}
    checks["all_paper_models_present"] = PAPER_MODELS.issubset(observed)
    checks["five_seeds_each_main_model"] = all(int(counts.get(name, 0)) >= 5 for name in PAPER_MODELS)
else:
    checks["all_paper_models_present"] = False
    checks["five_seeds_each_main_model"] = False

if (ABL / "ablation_metrics_by_seed.csv").exists():
    a = pd.read_csv(ABL / "ablation_metrics_by_seed.csv")
    observed = set(a["model"].astype(str))
    counts = a.groupby("model")["seed"].nunique().to_dict() if "seed" in a else {}
    checks["all_ablation_variants_present"] = PRIMARY_ABLATIONS.issubset(observed)
    checks["five_seeds_each_ablation"] = all(int(counts.get(name, 0)) >= 5 for name in PRIMARY_ABLATIONS)
else:
    checks["all_ablation_variants_present"] = False
    checks["five_seeds_each_ablation"] = False

if (PROC / "lira_physics_audit.json").exists():
    pa = json.loads((PROC / "lira_physics_audit.json").read_text())
    checks["physics_support_audit_pass"] = (
        float(pa.get("bound_above_upper_rate", 1.0))
        <= float(pa.get("max_allowed_bound_above_upper_rate", 0.0))
    )
else:
    checks["physics_support_audit_pass"] = False

if (PROC / "lira_preprocessing_report.json").exists():
    pr = json.loads((PROC / "lira_preprocessing_report.json").read_text())
    prep = pr.get("preprocessing", {})
    checks["split_before_imputation"] = bool(prep.get("split_before_imputation"))
    checks["partition_local_imputation"] = bool(prep.get("partition_local_imputation"))
    checks["trajectory_aware_splitting"] = bool(prep.get("trajectory_aware_splitting"))
    checks["segment_safe_windows"] = bool(prep.get("segment_safe_windows"))
else:
    for key in ("split_before_imputation", "partition_local_imputation", "trajectory_aware_splitting", "segment_safe_windows"):
        checks[key] = False

checks["proposal_tuning_present"] = all((TUNE / f).exists() for f in [
    "best_hparams.yaml", "tuning_summary.json", "tuning_endpoint_manifest.json"
])
checks["baseline_tuning_present"] = all((BTUNE / f).exists() for f in [
    "best_hparams.yaml", "tuning_summary.json"
])

# Proposal and all baseline tuning runs must score the exact same validation endpoints.
checks["tuning_endpoint_parity"] = False
p_manifest = TUNE / "tuning_endpoint_manifest.json"
b_manifests = sorted(BTUNE.glob("*/tuning_endpoint_manifest.json"))
if p_manifest.exists() and b_manifests:
    p = json.loads(p_manifest.read_text())
    checks["tuning_endpoint_parity"] = all(
        int(json.loads(m.read_text()).get("eval_start", -1)) == int(p.get("eval_start", -2))
        and int(json.loads(m.read_text()).get("n", -1)) == int(p.get("n", -2))
        and str(json.loads(m.read_text()).get("sha256")) == str(p.get("sha256"))
        for m in b_manifests
    )

status = "PAPER_READY" if all(checks.values()) else "REVIEW"
report = {
    "status": status,
    "checks": checks,
    "health": health,
    "proposal": "SafeGrip-CI v0.9",
    "note": (
        "PAPER_READY means the automatic scientific-health, fairness, endpoint-parity, "
        "multi-seed, ablation, preprocessing and statistics gates passed. It does not prove "
        "novelty or guarantee acceptance; manuscript claims must match measured results and limitations."
    ),
}
MAIN.mkdir(parents=True, exist_ok=True)
(MAIN / "paper_readiness.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
if status != "PAPER_READY":
    sys.exit(2)
