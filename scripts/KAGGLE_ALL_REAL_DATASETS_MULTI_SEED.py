# ================================================================
# UniversalSafeGrip — all real datasets, repeated-seed evidence
# Runs the exact same one-proposal protocol for each requested seed,
# preserving per-seed checkpoints and aggregating mean/std metrics.
# ================================================================
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

WORK = Path("/kaggle/working")
REPO = WORK / "quynh-method"
RUNNER = REPO / "scripts/KAGGLE_ALL_REAL_DATASETS_ONE_PROPOSAL_1SEED_10EPOCHS.py"
SEEDS = tuple(int(x) for x in os.environ.get("SAFEGRIP_SEEDS", "3101,3102,3103").split(",") if x.strip())
EPOCHS = int(os.environ.get("SAFEGRIP_EPOCHS", "10"))
GLOBAL_HOURS = float(os.environ.get("SAFEGRIP_MAX_HOURS", "11"))
GLOBAL_BUFFER_MIN = int(os.environ.get("SAFEGRIP_SAVE_BUFFER_MIN", "12"))
START = time.time()
DEADLINE = START + GLOBAL_HOURS * 3600 - GLOBAL_BUFFER_MIN * 60

if not SEEDS:
    raise ValueError("SAFEGRIP_SEEDS must contain at least one integer seed")
if not RUNNER.exists():
    raise FileNotFoundError(RUNNER)

completed = []
for pos, seed in enumerate(SEEDS):
    seconds_left = DEADLINE - time.time()
    seeds_left = len(SEEDS) - pos
    # Split the remaining budget fairly, so one seed cannot consume the whole
    # Kaggle session and prevent any repeated-seed evidence.
    per_seed_hours = max(0.0, seconds_left / 3600.0 / max(1, seeds_left))
    if per_seed_hours < 0.35:
        print(f"[multi-seed] stopping before seed {seed}: only {seconds_left/60:.1f} min remain")
        break

    run_name = f"safegrip_real_seed{seed}_{EPOCHS}epochs"
    env = os.environ.copy()
    env.update({
        "SAFEGRIP_SEED": str(seed),
        "SAFEGRIP_EPOCHS": str(EPOCHS),
        "SAFEGRIP_RUN_NAME": run_name,
        "SAFEGRIP_MAX_HOURS": f"{per_seed_hours:.6f}",
        # The global wrapper already reserves its final serialization buffer.
        "SAFEGRIP_SAVE_BUFFER_MIN": "8",
    })
    print("\n" + "=" * 100)
    print(f"MULTI-SEED RUN {pos+1}/{len(SEEDS)}: seed={seed}, budget={per_seed_hours:.2f} h")
    print("=" * 100)
    proc = subprocess.run([sys.executable, str(RUNNER)], cwd=REPO, env=env)
    if proc.returncode != 0:
        print(f"[multi-seed] seed {seed} exited with code {proc.returncode}; checkpoint retained")
    metric = WORK / run_name / "all_metrics.csv"
    if metric.exists():
        completed.append((seed, metric))

frames = []
for seed, path in completed:
    df = pd.read_csv(path)
    df["seed"] = seed
    frames.append(df)

if frames:
    all_runs = pd.concat(frames, ignore_index=True, sort=False)
    all_path = WORK / "safegrip_multiseed_all_metrics.csv"
    all_runs.to_csv(all_path, index=False)

    keys = [c for c in ("dataset", "model", "target", "proposal_revision", "split_policy", "data_kind") if c in all_runs.columns]
    metrics = [c for c in ("mae", "rmse", "r2") if c in all_runs.columns]
    summary = (
        all_runs.groupby(keys, dropna=False)[metrics]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    summary.columns = ["_".join(x).rstrip("_") if isinstance(x, tuple) else x for x in summary.columns]
    summary_path = WORK / "safegrip_multiseed_summary.csv"
    summary.to_csv(summary_path, index=False)
    print("\n=== MULTI-SEED SUMMARY ===")
    print(summary.to_string(index=False))
    print("\nSaved:", all_path)
    print("Saved:", summary_path)
else:
    print("[multi-seed] no completed metrics files yet; attach per-seed checkpoints and rerun")
