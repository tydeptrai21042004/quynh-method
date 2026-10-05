# ================================================================
# Universal SafeGrip NPI-v3 — controlled real-data ablation runner
# ================================================================
# NPI-v3 is the ONLY proposal. Every alternate configuration executed here is
# a one-factor controlled ablation of that proposal.
#
# Default primary table (3 seeds):
#   direct_normalized
#   reference_only
#   old_physical_innovation
#   npi_no_center
#   npi_no_scale
#   mean_pool
#   channel_id_only
#   npi_v3
#
# Optional supplementary controls are enabled with
#   SAFEGRIP_INCLUDE_SUPPLEMENTARY=1
# or an explicit comma-separated SAFEGRIP_ABLATIONS list.
#
# The underlying one-proposal runner is reused with baselines disabled, so
# dataset preparation, splits, seeds, epochs, checkpoint selection and metrics
# remain identical across variants.
# ================================================================
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd


WORK = Path("/kaggle/working")
REPO = WORK / "quynh-method"
RUNNER = REPO / "scripts/KAGGLE_ALL_REAL_DATASETS_ONE_PROPOSAL_1SEED_10EPOCHS.py"
MASTER_ZIP = WORK / "safegrip_npi_v3_ablation_master.zip"
MASTER_DIR = WORK / "safegrip_npi_v3_ablation_master"

PRIMARY = (
    "direct_normalized",
    "reference_only",
    "old_physical_innovation",
    "npi_no_center",
    "npi_no_scale",
    "mean_pool",
    "channel_id_only",
    "npi_v3",
)
SUPPLEMENTARY = (
    "no_physical_metadata",
    "no_spectrum",
    "no_sensor_dropout",
    "dataset_id_conditioning",
    "path_length_reference",
    "no_learned_scale",
)

SEEDS = tuple(
    int(x) for x in os.environ.get("SAFEGRIP_ABLATION_SEEDS", "3101,3102,3103").split(",")
    if x.strip()
)
EPOCHS = int(os.environ.get("SAFEGRIP_EPOCHS", "10"))
GLOBAL_HOURS = float(os.environ.get("SAFEGRIP_MAX_HOURS", "11"))
BUFFER_MIN = int(os.environ.get("SAFEGRIP_SAVE_BUFFER_MIN", "12"))
INCLUDE_SUPPLEMENTARY = os.environ.get("SAFEGRIP_INCLUDE_SUPPLEMENTARY", "0") in {"1", "true", "True"}

requested = os.environ.get("SAFEGRIP_ABLATIONS", "").strip()
if requested:
    VARIANTS = tuple(x.strip().lower() for x in requested.split(",") if x.strip())
else:
    VARIANTS = PRIMARY + (SUPPLEMENTARY if INCLUDE_SUPPLEMENTARY else ())

allowed = set(PRIMARY + SUPPLEMENTARY)
unknown = [v for v in VARIANTS if v not in allowed]
if unknown:
    raise ValueError(f"Unknown ablation(s): {unknown}. Allowed: {sorted(allowed)}")
if not SEEDS:
    raise ValueError("SAFEGRIP_ABLATION_SEEDS must contain at least one seed")
if not RUNNER.exists():
    raise FileNotFoundError(RUNNER)

# path_length_reference changes only D4. Running D1-D3 would intentionally
# duplicate NPI-v3 and waste paper-compute budget.
def datasets_for(variant: str) -> str:
    return "io_vnbd" if variant == "path_length_reference" else "lira_cd,uc3m_tire,deep_dynamics_iac,io_vnbd"


def restore_master() -> None:
    if MASTER_DIR.exists():
        return
    candidates = sorted(Path("/kaggle/input").rglob(MASTER_ZIP.name))
    if candidates:
        print(f"[ablation resume] restoring {candidates[-1]}")
        shutil.unpack_archive(str(candidates[-1]), str(WORK))


def collect_metrics() -> pd.DataFrame:
    frames = []
    for variant in VARIANTS:
        for seed in SEEDS:
            run_name = f"safegrip_npi_v3_ablation_{variant}_seed{seed}_{EPOCHS}epochs"
            p = WORK / run_name / "all_metrics.csv"
            if not p.exists():
                continue
            df = pd.read_csv(p)
            if "ablation" in df.columns:
                df = df[df["ablation"].fillna("") == variant]
            else:
                df = df[df["model"].astype(str).str.contains("universal_safegrip", na=False)]
                df["ablation"] = variant
            if not df.empty:
                df["seed"] = seed
                frames.append(df)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def save_master() -> None:
    MASTER_DIR.mkdir(parents=True, exist_ok=True)
    all_runs = collect_metrics()
    if not all_runs.empty:
        all_runs.to_csv(MASTER_DIR / "npi_v3_ablation_by_seed.csv", index=False)
        metric_cols = [c for c in ("mae", "rmse", "r2") if c in all_runs.columns]
        keys = [c for c in ("ablation", "dataset", "target", "innovation_mode", "localization_reference") if c in all_runs.columns]
        summary = all_runs.groupby(keys, dropna=False)[metric_cols].agg(["mean", "std", "count"]).reset_index()
        summary.columns = ["_".join(x).rstrip("_") if isinstance(x, tuple) else x for x in summary.columns]
        summary.to_csv(MASTER_DIR / "npi_v3_ablation_summary.csv", index=False)

    # Store each resumable run folder, not only the CSVs. On the next Kaggle
    # session the master archive recreates /kaggle/working/<run_name>, allowing
    # the underlying runner to continue from its checkpoint/state.
    runs_dir = MASTER_DIR / "runs"
    runs_dir.mkdir(exist_ok=True)
    for variant in VARIANTS:
        for seed in SEEDS:
            run_name = f"safegrip_npi_v3_ablation_{variant}_seed{seed}_{EPOCHS}epochs"
            src = WORK / run_name
            dst = runs_dir / run_name
            if src.exists():
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)

    if MASTER_ZIP.exists():
        MASTER_ZIP.unlink()
    shutil.make_archive(str(MASTER_ZIP.with_suffix("")), "zip", root_dir=MASTER_DIR.parent, base_dir=MASTER_DIR.name)
    print("[ablation checkpoint]", MASTER_ZIP)


def restore_run_folders_from_master() -> None:
    runs_dir = MASTER_DIR / "runs"
    if not runs_dir.exists():
        return
    for src in runs_dir.iterdir():
        if not src.is_dir():
            continue
        dst = WORK / src.name
        if not dst.exists():
            shutil.copytree(src, dst)


restore_master()
restore_run_folders_from_master()

start = time.time()
deadline = start + GLOBAL_HOURS * 3600 - BUFFER_MIN * 60
jobs = [(v, s) for v in VARIANTS for s in SEEDS]

try:
    for pos, (variant, seed) in enumerate(jobs):
        run_name = f"safegrip_npi_v3_ablation_{variant}_seed{seed}_{EPOCHS}epochs"
        metric = WORK / run_name / "all_metrics.csv"
        if metric.exists():
            try:
                done = pd.read_csv(metric)
                expected_datasets = set(datasets_for(variant).split(","))
                got = set(done.loc[done.get("ablation", pd.Series(index=done.index, dtype=str)).fillna("") == variant, "dataset"])
                if expected_datasets.issubset(got):
                    print(f"[skip complete] {variant} seed={seed}")
                    continue
            except Exception:
                pass

        seconds_left = deadline - time.time()
        jobs_left = len(jobs) - pos
        per_job_hours = max(0.0, seconds_left / 3600 / max(1, jobs_left))
        if per_job_hours < 0.30:
            print(f"[time] stopping before {variant}/seed{seed}: {seconds_left/60:.1f} min remain")
            break

        env = os.environ.copy()
        env.update({
            "SAFEGRIP_SEED": str(seed),
            "SAFEGRIP_EPOCHS": str(EPOCHS),
            "SAFEGRIP_RUN_NAME": run_name,
            "SAFEGRIP_ABLATION": variant,
            "SAFEGRIP_DATASETS": datasets_for(variant),
            "SAFEGRIP_RUN_BASELINES": "0",
            "SAFEGRIP_MAX_HOURS": f"{per_job_hours:.6f}",
            "SAFEGRIP_SAVE_BUFFER_MIN": "5",
            # Repository tests are identical across variants; run them in the
            # normal proposal job, not repeatedly for every ablation.
            "SAFEGRIP_RUN_TESTS": "0",
        })
        print("\n" + "=" * 100)
        print(f"NPI-v3 ABLATION {pos+1}/{len(jobs)}: {variant}, seed={seed}, budget={per_job_hours:.2f}h")
        print("=" * 100)
        proc = subprocess.run([sys.executable, str(RUNNER)], cwd=REPO, env=env)
        save_master()

        # If the current job was only partially completed, preserve it and stop.
        # This keeps the next session focused on the same controlled comparison.
        if proc.returncode != 0 or not metric.exists():
            print(f"[ablation] incomplete {variant}/seed{seed}; resume from master ZIP next session")
            break
finally:
    save_master()
    all_runs = collect_metrics()
    if not all_runs.empty:
        print("\n=== NPI-v3 ABLATION RESULTS COLLECTED ===")
        cols = [c for c in ("ablation", "dataset", "target", "seed", "mae", "rmse", "r2") if c in all_runs.columns]
        print(all_runs[cols].to_string(index=False))
    print("\nMaster checkpoint/result ZIP:", MASTER_ZIP)
