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
    "metrics.csv", "metrics_by_seed.csv", "predictions.csv", "result_health.json",
    "sanity_baselines.csv", "evaluation_protocol.json", "reproducibility_manifest.json",
    "baseline_manifest.csv", "proposal_hparams.json", "baseline_selected_hparams.json",
]
required_proc = [
    "lira_aligned.csv", "lira_signal_audit.csv", "lira_physics_audit.json",
    "lira_friction_schema_report.csv", "lira_alignment_report.csv", "lira_preprocessing_report.json",
]
required_abl = ["ablation_metrics.csv", "ablation_metrics_by_seed.csv", "ablation_predictions.csv", "ablation_design.json"]

checks = {}
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

if (MAIN / "metrics_by_seed.csv").exists():
    m = pd.read_csv(MAIN / "metrics_by_seed.csv")
    expected = {"todorovic2022_cnn", "lampe2023_lstm", "lampe2023_gru", "schaefke2023_transformer", "chen2025_svdkl", "safegrip"}
    checks["all_paper_models_present"] = expected.issubset(set(m["model"].astype(str)))
    counts = m.groupby("model")["seed"].nunique().to_dict() if "seed" in m else {}
    checks["five_seeds_each_main_model"] = all(int(counts.get(name, 0)) >= 5 for name in expected)
else:
    checks["all_paper_models_present"] = False
    checks["five_seeds_each_main_model"] = False

if (ABL / "ablation_metrics_by_seed.csv").exists():
    a = pd.read_csv(ABL / "ablation_metrics_by_seed.csv")
    counts = a.groupby("model")["seed"].nunique().to_dict() if "seed" in a else {}
    expected_abl = {"safegrip_data_only", "safegrip_no_projection", "safegrip_no_uq", "safegrip_no_physics_loss", "safegrip_no_calibration", "safegrip_no_temporal", "safegrip"}
    checks["all_ablation_variants_present"] = expected_abl.issubset(set(a["model"].astype(str)))
    checks["five_seeds_each_ablation"] = all(int(counts.get(name, 0)) >= 5 for name in expected_abl)
else:
    checks["all_ablation_variants_present"] = False
    checks["five_seeds_each_ablation"] = False

if (PROC / "lira_physics_audit.json").exists():
    pa = json.loads((PROC / "lira_physics_audit.json").read_text())
    checks["physics_support_audit_pass"] = float(pa.get("bound_above_upper_rate", 1.0)) <= float(pa.get("max_allowed_bound_above_upper_rate", 0.0))
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

checks["proposal_tuning_present"] = (TUNE / "best_hparams.yaml").exists() and (TUNE / "tuning_summary.json").exists()
checks["baseline_tuning_present"] = (BTUNE / "best_hparams.yaml").exists() and (BTUNE / "tuning_summary.json").exists()

status = "PAPER_READY" if all(checks.values()) else "REVIEW"
report = {
    "status": status,
    "checks": checks,
    "health": health,
    "note": "PAPER_READY means the repository's automatic reproducibility/sanity gates passed. It does not prove novelty or guarantee acceptance; manuscript claims must still match the measured results and limitations.",
}
MAIN.mkdir(parents=True, exist_ok=True)
(MAIN / "paper_readiness.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
if status != "PAPER_READY":
    sys.exit(2)
