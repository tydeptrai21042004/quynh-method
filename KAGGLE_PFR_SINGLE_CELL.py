# SafeGrip-PFR-ECR corrected Kaggle single-cell driver.
# Paste this file into one Kaggle Python cell or run it as a script.
# Internet must be ON for cloning/downloading LiRA.

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

MODE = os.environ.get("SAFEGRIP_MODE", "quick")  # quick | trust | paper
REPO_URL = os.environ.get("SAFEGRIP_REPO", "https://github.com/tydeptrai21042004/quynh-method.git")
WORK = Path("/kaggle/working")
REPO = WORK / "quynh-method"
RESULT_ZIP = WORK / f"safegrip_pfr_{MODE}_results.zip"


def run(*args: str) -> None:
    print("+", " ".join(args), flush=True)
    subprocess.run(args, cwd=REPO, check=True)


if MODE not in {"quick", "trust", "paper"}:
    raise ValueError("SAFEGRIP_MODE must be quick, trust, or paper")

shutil.rmtree(REPO, ignore_errors=True)
if RESULT_ZIP.exists():
    RESULT_ZIP.unlink()
subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(REPO)], check=True)

run(sys.executable, "-m", "pip", "install", "-q", "-e", ".[paper,dev]")
run(sys.executable, "-m", "pytest", "-q")
run(sys.executable, "-m", "safegrip.cli", "download", "--datasets", "lira")
run(sys.executable, "-m", "safegrip.cli", "prepare", "--dataset", "lira")

proposal_args: list[str] = []
baseline_args: list[str] = []
if MODE == "paper":
    run(sys.executable, "-m", "safegrip.cli", "tune", "--method", "pfr", "--dataset", "lira", "--trials", "20")
    run(sys.executable, "-m", "safegrip.cli", "tune-baselines", "--dataset", "lira", "--protocol", "controlled", "--trials", "30")
    proposal_args = ["--proposal-hparams", "results/lira_pfr_tuning/best_hparams.yaml"]
    baseline_args = ["--baseline-hparams", "results/lira_baseline_tuning/best_hparams.yaml"]

run(
    sys.executable, "-m", "safegrip.cli", "benchmark",
    "--dataset", "lira", "--preset", MODE,
    "--proposal", "pfr", "--protocol", "controlled",
    *proposal_args, *baseline_args,
)

result_dir = REPO / "results" / f"lira_{MODE}"
run(
    sys.executable, "-m", "safegrip.cli", "statistics",
    "--results", str(result_dir.relative_to(REPO)),
    "--proposal", "safegrip_pfr",
    "--bootstrap", "500" if MODE == "quick" else "2000",
)

archive_base = RESULT_ZIP.with_suffix("")
shutil.make_archive(str(archive_base), "zip", root_dir=result_dir)

print("\nSafeGrip-PFR-ECR run complete")
print("Metrics:", result_dir / "metrics.csv")
print("Ablation:", result_dir / "pfr_component_ablation.csv")
print("Safety metrics:", result_dir / "pfr_safety_metrics.csv")
print("Risk-split sensitivity:", result_dir / "pfr_risk_split_sensitivity.csv")
print("Safety audit:", result_dir / "pfr_safety_fusion_audit.csv")
print("Theorem audit:", result_dir / "pfr_theorem_audit.json")
print("Result ZIP:", RESULT_ZIP)
