# ============================================================
# SafeGrip-PFR-ECR — REAL-DATA-ONLY KAGGLE DRIVER (10 epochs)
#
# Primary real friction benchmarks:
#   1. LiRA-CD platoon friction test (automatic download)
#   2. Guo et al. MSSP-2023 real dynamics/friction dataset
#      (runs only when the real author-supplied payload is mounted)
#
# Real auxiliary validation sources:
#   - KIT tire-force dataset
#   - KU Leuven LMSD Concept Car wheel-force dataset
#   - Mendeley tire-pavement friction coefficient dataset
#
# No synthetic/simulated dataset is generated or used.
# ============================================================

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

MODE = os.environ.get("SAFEGRIP_MODE", "trust")  # quick | trust | paper
EPOCHS = 10
REPO_URL = os.environ.get("SAFEGRIP_REPO", "https://github.com/tydeptrai21042004/quynh-method.git")
WORK = Path("/kaggle/working")
REPO = WORK / "quynh-method"
CONFIG = REPO / "configs" / "kaggle_real_10epochs.yaml"
FINAL_DIR = WORK / f"safegrip_real_only_{MODE}_10epochs"
FINAL_ZIP = WORK / f"safegrip_real_only_{MODE}_10epochs_results.zip"

BASELINES = [
    "du2023_inceptiontime",
    "todorovic2022_cnn",
    "lampe2023_gru",
    "levenberg2023_stft",
]

# Optional real MSSP-2023 payload.  Set SAFEGRIP_MSSP2023_INPUT to a Kaggle
# dataset directory containing the authors' extracted dynamics tables.
MSSP_INPUT = Path(os.environ.get("SAFEGRIP_MSSP2023_INPUT", "/kaggle/input/mssp2023-friction"))
RUN_AUX_REAL = os.environ.get("SAFEGRIP_RUN_AUX_REAL", "1") not in {"0", "false", "False"}


def run(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    print("\n+", " ".join(map(str, args)), flush=True)
    return subprocess.run(list(map(str, args)), cwd=REPO, check=check)


def copy_results(dataset: str) -> None:
    src = REPO / "results" / f"{dataset}_{MODE}"
    if src.exists():
        dst = FINAL_DIR / f"{dataset}_{MODE}_10epochs"
        shutil.rmtree(dst, ignore_errors=True)
        shutil.copytree(src, dst)


def run_primary(dataset: str) -> None:
    run(sys.executable, "-m", "safegrip.cli", "--config", str(CONFIG), "prepare", "--dataset", dataset)
    run(
        sys.executable, "-m", "safegrip.cli", "--config", str(CONFIG), "benchmark",
        "--dataset", dataset,
        "--preset", MODE,
        "--proposal", "pfr",
        "--protocol", "controlled",
        "--models", ",".join(BASELINES),
    )
    result_dir = REPO / "results" / f"{dataset}_{MODE}"
    run(
        sys.executable, "-m", "safegrip.cli", "--config", str(CONFIG), "statistics",
        "--results", str(result_dir.relative_to(REPO)),
        "--proposal", "safegrip_pfr",
        "--bootstrap", "500" if MODE == "quick" else "2000",
    )
    run(
        sys.executable, "-m", "safegrip.cli", "--config", str(CONFIG), "plots",
        "--results", str(result_dir.relative_to(REPO)),
        check=False,
    )
    copy_results(dataset)


if MODE not in {"quick", "trust", "paper"}:
    raise ValueError("SAFEGRIP_MODE must be quick, trust, or paper")

shutil.rmtree(REPO, ignore_errors=True)
shutil.rmtree(FINAL_DIR, ignore_errors=True)
if FINAL_ZIP.exists():
    FINAL_ZIP.unlink()

subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(REPO)], check=True)
run(sys.executable, "-m", "pip", "install", "-q", "-e", ".[paper,dev]")
run(sys.executable, "-m", "pytest", "-q")

# Load the complete repository config and change only the training budget.
# This preserves required sections such as vehicle, LiRA alignment, split,
# physics, baselines, and evaluation settings.
default_cfg = yaml.safe_load((REPO / "configs" / "default.yaml").read_text(encoding="utf-8"))
default_cfg.setdefault("training", {})["epochs_quick"] = EPOCHS
default_cfg["training"]["epochs_trust"] = EPOCHS
default_cfg["training"]["epochs_paper"] = EPOCHS
default_cfg["training"]["patience"] = EPOCHS
default_cfg["training"]["patience_trust"] = EPOCHS
default_cfg.setdefault("comparison", {}).setdefault("controlled", {})["epochs"] = EPOCHS
default_cfg["comparison"]["controlled"]["patience"] = EPOCHS
default_cfg.setdefault("pfr", {})["epochs"] = EPOCHS
default_cfg["pfr"]["patience"] = EPOCHS
default_cfg["pfr"]["extended_ablations"] = True
CONFIG.write_text(yaml.safe_dump(default_cfg, sort_keys=False), encoding="utf-8")

FINAL_DIR.mkdir(parents=True, exist_ok=True)
shutil.copy2(CONFIG, FINAL_DIR / CONFIG.name)

# ---------------------------------------------------------------------------
# Primary real benchmark 1: LiRA
# ---------------------------------------------------------------------------
run(sys.executable, "-m", "safegrip.cli", "--config", str(CONFIG), "download", "--datasets", "lira")
run_primary("lira")

# ---------------------------------------------------------------------------
# Primary real benchmark 2: MSSP-2023, only if the REAL payload is available.
# No generated replacement data is permitted.
# ---------------------------------------------------------------------------
if MSSP_INPUT.exists() and any(p.is_file() for p in MSSP_INPUT.rglob("*")):
    raw_mssp = REPO / "data" / "raw" / "mssp2023_friction"
    shutil.rmtree(raw_mssp, ignore_errors=True)
    shutil.copytree(MSSP_INPUT, raw_mssp)
    try:
        run_primary("mssp2023_friction")
    except subprocess.CalledProcessError as exc:
        print(f"[MSSP-2023] Real payload found but preparation/benchmark failed: {exc}")
        print("Inspect data/processed/mssp2023_friction/mssp2023_schema_audit.csv.")
else:
    print("\n[MSSP-2023] Real payload not mounted; skipping rather than generating simulated data.")
    print("Set SAFEGRIP_MSSP2023_INPUT to the extracted author-supplied real dataset directory.")

# ---------------------------------------------------------------------------
# Additional real-data validation. These are deliberately NOT forced into the
# same point-prediction table because their targets differ from LiRA/MSSP.
# ---------------------------------------------------------------------------
if RUN_AUX_REAL:
    # KIT: measured force-utilization validation.
    try:
        run(sys.executable, "-m", "safegrip.cli", "--config", str(CONFIG), "download", "--datasets", "kit")
        run(sys.executable, "-m", "safegrip.cli", "--config", str(CONFIG), "prepare", "--dataset", "kit")
        run(sys.executable, "-m", "safegrip.cli", "--config", str(CONFIG), "force-validate", "--dataset", "kit")
        src = REPO / "results" / "kit_force_validation"
        if src.exists(): shutil.copytree(src, FINAL_DIR / "kit_force_validation", dirs_exist_ok=True)
    except Exception as exc:
        print(f"[KIT] optional real-data validation skipped: {exc}")

    # KU Leuven: real vehicle wheel-force validation. Public repository terms may
    # require user acceptance, so failure is non-fatal and never replaced by fake data.
    try:
        run(sys.executable, "-m", "safegrip.cli", "--config", str(CONFIG), "download", "--datasets", "kuleuven")
        run(sys.executable, "-m", "safegrip.cli", "--config", str(CONFIG), "prepare", "--dataset", "kuleuven")
        run(sys.executable, "-m", "safegrip.cli", "--config", str(CONFIG), "force-validate", "--dataset", "kuleuven")
        src = REPO / "results" / "kuleuven_force_validation"
        if src.exists(): shutil.copytree(src, FINAL_DIR / "kuleuven_force_validation", dirs_exist_ok=True)
    except Exception as exc:
        print(f"[KU Leuven] optional real-data validation skipped: {exc}")

    # Mendeley: real measured friction/speed/surface reference. This is an
    # external reference validation, not a fake sensor sequence for the PFR model.
    try:
        run(sys.executable, "-m", "safegrip.cli", "--config", str(CONFIG), "download", "--datasets", "mendeley_friction")
        run(sys.executable, "-m", "safegrip.cli", "--config", str(CONFIG), "prepare", "--dataset", "mendeley_friction")
        run(sys.executable, "-m", "safegrip.cli", "--config", str(CONFIG), "friction-reference-validate", "--dataset", "mendeley_friction")
        src = REPO / "results" / "mendeley_friction_reference_validation"
        if src.exists(): shutil.copytree(src, FINAL_DIR / "mendeley_friction_reference_validation", dirs_exist_ok=True)
    except Exception as exc:
        print(f"[Mendeley friction] optional real-data validation skipped: {exc}")

# Package only real-data results.
shutil.make_archive(str(FINAL_ZIP.with_suffix("")), "zip", root_dir=FINAL_DIR)

print("\n" + "=" * 90)
print("SafeGrip-PFR-ECR REAL-DATA-ONLY run complete")
print("Primary LiRA results:", FINAL_DIR / f"lira_{MODE}_10epochs")
if (FINAL_DIR / f"mssp2023_friction_{MODE}_10epochs").exists():
    print("Primary MSSP-2023 results:", FINAL_DIR / f"mssp2023_friction_{MODE}_10epochs")
print("Result ZIP:", FINAL_ZIP)
print("No synthetic/simulated dataset was generated or benchmarked.")
