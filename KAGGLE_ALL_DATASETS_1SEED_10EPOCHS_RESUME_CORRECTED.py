# ================================================================
# SafeGrip / PFR-ECR — ALL REGISTERED DATASETS, 1 SEED, 10 EPOCHS
# Kaggle single-cell driver with checkpoint/resume and 11-hour guard.
#
# D1 lira_cd:
#   SafeGrip-PFR-ECR + the 4 registered LiRA baselines.
# D2-D4:
#   UniversalSafeGrip proposal + the registered per-dataset baselines.
#
# Scientific rule: missing real targets are NEVER fabricated. If an upstream
# dataset does not expose the task target required by its adapter, that dataset
# is logged as SKIPPED_NO_EXPLICIT_TARGETS rather than silently synthesised.
#
# Resume:
#   - during one Kaggle session, rerun this cell: it resumes from state.json.
#   - across Kaggle sessions, save safegrip_all_datasets_checkpoint.zip as a
#     notebook output / Kaggle Dataset and attach it to the next notebook.
#     This cell auto-discovers it anywhere under /kaggle/input.
# ================================================================

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import pickle
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

# ----------------------------- USER SETTINGS -----------------------------
REPO_URL = os.environ.get(
    "SAFEGRIP_REPO",
    "https://github.com/tydeptrai21042004/quynh-method.git",
)
BRANCH = os.environ.get("SAFEGRIP_BRANCH", "main")
SEED = int(os.environ.get("SAFEGRIP_SEED", "3101"))
EPOCHS = int(os.environ.get("SAFEGRIP_EPOCHS", "10"))
MAX_WALL_HOURS = float(os.environ.get("SAFEGRIP_MAX_HOURS", "11"))
# Reserve time to serialize outputs before Kaggle cuts the session.
SAVE_BUFFER_MIN = int(os.environ.get("SAFEGRIP_SAVE_BUFFER_MIN", "12"))
# 0 means use all valid records. Set e.g. 5000 only for a deliberate pilot.
MAX_RECORDS_PER_DATASET = int(os.environ.get("SAFEGRIP_MAX_RECORDS", "0"))
UNIVERSAL_BATCH = int(os.environ.get("SAFEGRIP_BATCH", "32"))
BASELINE_BATCH = int(os.environ.get("SAFEGRIP_BASELINE_BATCH", "128"))
RUN_TESTS = os.environ.get("SAFEGRIP_RUN_TESTS", "1") not in {"0", "false", "False"}

DATASETS = ("lira_cd", "uc3m_tire", "deep_dynamics_iac", "io_vnbd")
D2D4_INPUT_DIMS = {"uc3m_tire": 3, "deep_dynamics_iac": 5, "io_vnbd": 8}

WORK = Path("/kaggle/working")
REPO = WORK / "quynh-method"
RUN_ROOT = WORK / "safegrip_all_datasets_1seed_10epochs"
RESULTS_ROOT = RUN_ROOT / "results"
STATE_PATH = RUN_ROOT / "state.json"
CONFIG_PATH = RUN_ROOT / "kaggle_1seed_10epochs.yaml"
CHECKPOINT_ZIP = WORK / "safegrip_all_datasets_checkpoint.zip"
FINAL_ZIP = WORK / "safegrip_all_datasets_1seed_10epochs_results.zip"

START_TIME = time.time()
HARD_DEADLINE = START_TIME + MAX_WALL_HOURS * 3600.0
SAFE_DEADLINE = HARD_DEADLINE - SAVE_BUFFER_MIN * 60.0


# ----------------------------- SMALL UTILITIES ----------------------------
class TimeBudgetReached(RuntimeError):
    pass


def sh(cmd, *, cwd=None, check=True, timeout=None, env=None):
    if isinstance(cmd, (list, tuple)):
        printable = " ".join(map(str, cmd))
    else:
        printable = str(cmd)
    print("\n" + "=" * 100)
    print("RUN:", printable, flush=True)
    print("=" * 100, flush=True)
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        check=check,
        timeout=timeout,
        env=env,
        shell=isinstance(cmd, str),
        executable="/bin/bash" if isinstance(cmd, str) else None,
    )


def remaining_seconds():
    return max(0.0, SAFE_DEADLINE - time.time())


def time_guard(label="next stage", minimum_seconds=120.0):
    left = remaining_seconds()
    if left < minimum_seconds:
        raise TimeBudgetReached(
            f"Stopping before 11h limit: only {left/60:.1f} safe minutes remain before {label}."
        )


def atomic_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def restore_previous_checkpoint():
    if STATE_PATH.exists():
        return
    candidates = sorted(Path("/kaggle/input").rglob("safegrip_all_datasets_checkpoint.zip"))
    if not candidates:
        return
    src = candidates[-1]
    print(f"[resume] Restoring previous checkpoint: {src}")
    shutil.unpack_archive(str(src), str(WORK))


def load_state():
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    return {
        "version": 2,
        "seed": SEED,
        "epochs": EPOCHS,
        "repo_url": REPO_URL,
        "git_commit": None,
        "completed": [],
        "progress": {},
        "errors": {},
        "skipped": {},
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def save_state():
    STATE["updated_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    atomic_json(STATE_PATH, STATE)


def make_checkpoint_zip():
    save_state()
    base = CHECKPOINT_ZIP.with_suffix("")
    if CHECKPOINT_ZIP.exists():
        CHECKPOINT_ZIP.unlink()
    shutil.make_archive(str(base), "zip", root_dir=RUN_ROOT.parent, base_dir=RUN_ROOT.name)
    print(f"[checkpoint] {CHECKPOINT_ZIP} ({CHECKPOINT_ZIP.stat().st_size/1e6:.1f} MB)")


def is_done(stage: str) -> bool:
    return stage in set(STATE.get("completed", []))


def mark_done(stage: str):
    if not is_done(stage):
        STATE.setdefault("completed", []).append(stage)
    STATE.setdefault("progress", {}).pop(stage, None)
    STATE.setdefault("errors", {}).pop(stage, None)
    save_state()
    make_checkpoint_zip()


def mark_error(stage: str, exc: Exception):
    STATE.setdefault("errors", {})[stage] = f"{type(exc).__name__}: {exc}"
    save_state()
    make_checkpoint_zip()


def seed_all(seed: int):
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def metric_dict(y, pred):
    import numpy as np
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    m = np.isfinite(y) & np.isfinite(pred)
    y, pred = y[m], pred[m]
    if len(y) == 0:
        return {"n": 0, "rmse": float("nan"), "mae": float("nan"), "r2": float("nan")}
    rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
    mae = float(np.mean(np.abs(pred - y)))
    den = float(np.sum((y - np.mean(y)) ** 2))
    r2 = float(1.0 - np.sum((pred - y) ** 2) / den) if den > 1e-12 else float("nan")
    return {"n": int(len(y)), "rmse": rmse, "mae": mae, "r2": r2}


# ----------------------------- RESTORE / CLONE ----------------------------
restore_previous_checkpoint()
RUN_ROOT.mkdir(parents=True, exist_ok=True)
RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
STATE = load_state()

if STATE.get("seed") != SEED or STATE.get("epochs") != EPOCHS:
    raise RuntimeError(
        f"Checkpoint was created with seed={STATE.get('seed')} epochs={STATE.get('epochs')}, "
        f"but this run requests seed={SEED} epochs={EPOCHS}. Use matching settings or a fresh output folder."
    )

if not REPO.exists():
    sh(["git", "clone", "--depth", "1", "--branch", BRANCH, REPO_URL, str(REPO)])
else:
    print(f"[repo] Reusing {REPO}")

# Resume on the exact commit if a previous checkpoint recorded one.
if STATE.get("git_commit"):
    wanted = STATE["git_commit"]
    current = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    if current != wanted:
        print(f"[repo] Restoring checkpoint commit {wanted}")
        sh(["git", "fetch", "--depth", "1", "origin", wanted], cwd=REPO, check=False)
        sh(["git", "checkout", wanted], cwd=REPO)
else:
    STATE["git_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    save_state()

# Install current checkout.
sh([sys.executable, "-m", "pip", "install", "-q", "-e", ".[paper,dev]"], cwd=REPO)
sys.path.insert(0, str(REPO / "src"))

# Optional compact correctness gate; experiment runtime is the priority.
if RUN_TESTS and not is_done("repo_tests"):
    try:
        time_guard("repository tests", 300)
        sh(
            [sys.executable, "-m", "pytest", "-q",
             "tests/test_pfr.py", "tests/test_registry.py", "tests/test_paper_benchmark.py",
             "tests/test_universal_training.py", "tests/test_d2_d4_baselines.py"],
            cwd=REPO,
            timeout=max(300, int(remaining_seconds() - 60)),
        )
        mark_done("repo_tests")
    except Exception as exc:
        mark_error("repo_tests", exc)
        raise

# ----------------------------- 10-EPOCH CONFIG ----------------------------
import yaml

cfg = yaml.safe_load((REPO / "configs" / "default.yaml").read_text(encoding="utf-8"))
# Prefer trajectory-level holdout, but public LiRA downloads can contain fewer
# than four independent trajectory/trip groups. Newer repo revisions understand
# this explicit fallback policy; the prepare retry below also supports older
# revisions by rewriting the effective mode to spatial_within_trajectory.
cfg.setdefault("split", {}).setdefault("mode", "group_holdout")
cfg["split"]["insufficient_group_policy"] = "fallback_spatial_within_trajectory"
cfg.setdefault("evaluation", {})["seeds"] = [SEED]
cfg["evaluation"]["trust_seeds"] = [SEED]
cfg.setdefault("comparison", {}).setdefault("controlled", {})["evaluation_seeds"] = [SEED]
cfg["comparison"]["controlled"]["epochs"] = EPOCHS
cfg["comparison"]["controlled"]["patience"] = EPOCHS
cfg.setdefault("training", {})["epochs_quick"] = EPOCHS
cfg["training"]["epochs_trust"] = EPOCHS
cfg["training"]["epochs_paper"] = EPOCHS
cfg["training"]["patience"] = EPOCHS
cfg["training"]["patience_trust"] = EPOCHS
cfg.setdefault("pfr", {})["epochs"] = EPOCHS
cfg["pfr"]["patience"] = EPOCHS
# We run only the proposal + paper baselines here; no extra PFR ablation training.
cfg["pfr"]["extended_ablations"] = False
CONFIG_PATH.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

from safegrip.config import load_config
from safegrip.datasets import PAPER_DATASETS
from safegrip.literature import baselines_for_dataset, LITERATURE_BASELINES

if tuple(PAPER_DATASETS) != DATASETS:
    print("[warning] Repository PAPER_DATASETS differs from this cell:", PAPER_DATASETS)


# ----------------------------- DOWNLOAD / PREPARE --------------------------
def ensure_dataset(dataset: str):
    """Download and prepare one dataset with an auditable LiRA split fallback.

    The public LiRA DTU bulk artifact can expose fewer than four independent
    trajectory/trip groups. group_holdout is then mathematically impossible.
    We first try the configured stronger protocol; if LiRA preparation fails,
    we retry only LiRA with spatial_within_trajectory and persist that effective
    mode in this run's config instead of aborting the whole 11-hour job.
    """
    stage = f"data:{dataset}"
    raw = REPO / "data" / "raw" / dataset
    proc = REPO / "data" / "processed" / dataset
    if is_done(stage) and raw.exists() and proc.exists():
        return
    time_guard(f"download/prepare {dataset}", 300)
    try:
        # Avoid downloading the 220 MB LiRA archive again when rerunning in the
        # same Kaggle session after a preparation-only failure.
        has_raw_files = raw.exists() and any(p.is_file() for p in raw.rglob("*"))
        if not has_raw_files:
            sh([sys.executable, "-m", "safegrip.cli", "--config", str(CONFIG_PATH),
                "download", "--datasets", dataset], cwd=REPO,
               timeout=max(300, int(remaining_seconds() - 60)))
        else:
            print(f"[{dataset}] raw files already exist; skipping re-download")

        prepare_cmd = [sys.executable, "-m", "safegrip.cli", "--config", str(CONFIG_PATH),
                       "prepare", "--dataset", dataset]
        try:
            sh(prepare_cmd, cwd=REPO, timeout=max(300, int(remaining_seconds() - 60)))
        except subprocess.CalledProcessError as first_exc:
            if dataset != "lira_cd":
                raise
            # Compatibility path for a checkout that predates the repository
            # fallback patch. Persist the weaker effective protocol explicitly.
            print("[lira_cd] group_holdout preparation failed. Retrying with an explicit "
                  "spatial_within_trajectory split because the public bundle may have <4 groups.")
            effective_cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
            effective_cfg.setdefault("split", {})["mode"] = "spatial_within_trajectory"
            effective_cfg["split"]["fallback_from"] = "group_holdout"
            effective_cfg["split"]["fallback_reason"] = "public LiRA bundle has insufficient independent trajectory/trip groups"
            CONFIG_PATH.write_text(yaml.safe_dump(effective_cfg, sort_keys=False), encoding="utf-8")
            sh(prepare_cmd, cwd=REPO, timeout=max(300, int(remaining_seconds() - 60)))
            STATE.setdefault("protocol_fallbacks", {})[dataset] = {
                "requested": "group_holdout",
                "effective": "spatial_within_trajectory",
                "reason": "initial LiRA preparation could not satisfy group_holdout; explicit audited retry",
                "first_error": f"{type(first_exc).__name__}: {first_exc}",
            }
            save_state()

        mark_done(stage)
    except Exception as exc:
        mark_error(stage, exc)
        raise


# ----------------------------- D1 PRIMARY-ONLY RUNNER ----------------------
def run_d1_lira():
    """Run only PFR-ECR + the four registered LiRA baselines (no retrained ablations)."""
    import numpy as np
    import pandas as pd
    from safegrip.benchmark import make_bundle, fit_literature, predict_literature, regression_metrics
    from safegrip.pfr_benchmark import (
        _pfr_hparams, _pfr_bundle_cfg, _effective_baseline_hparams,
        _common_eval_start_pfr, _fit_pfr_gru, _predict_pfr,
    )
    from safegrip.pfr import compose_raw_prediction, conformal_safe_correction, statistical_safe_lower, fuse_safe_lower
    from safegrip.physics import conformal_lower_correction, apply_lower_correction
    from safegrip.conformal import resolved_block_size
    from safegrip.utils import seed_everything

    dataset = "lira_cd"
    ensure_dataset(dataset)
    out = RESULTS_ROOT / dataset
    out.mkdir(parents=True, exist_ok=True)
    csv_path = REPO / "data" / "processed" / dataset / "lira_aligned.csv"
    names = list(baselines_for_dataset(dataset))
    local_cfg = load_config(str(CONFIG_PATH))
    hp = _pfr_hparams(local_cfg)
    hp["epochs"] = EPOCHS
    hp["patience"] = EPOCHS
    baseline_selected = {}
    eval_start = _common_eval_start_pfr(local_cfg, hp, names, baseline_selected, "controlled", "paper")
    pfr_cfg = _pfr_bundle_cfg(local_cfg, hp)

    # Build proposal bundle once.
    pfr_bundle = make_bundle(
        csv_path, pfr_cfg, sequence_length=int(hp["sequence_length"]),
        scaler_kind=str(hp["scaler"]), eval_start=eval_start, feature_mode="pfr",
    )
    total_alpha = float(local_cfg.get("alpha", 0.05))
    risk_split = float(hp.get("risk_split", 0.5))
    physics_alpha = total_alpha * risk_split
    statistical_alpha = total_alpha - physics_alpha
    uq_cfg = local_cfg.get("uq", {})
    block_size = resolved_block_size(
        uq_cfg, sequence_length=int(hp["sequence_length"]), stride=int(local_cfg.get("stride", 1))
    ) if str(uq_cfg.get("method", "block_max_split_conformal")) == "block_max_split_conformal" else 1
    q_physics = conformal_lower_correction(
        pfr_bundle.raw_loc, pfr_bundle.yc, physics_alpha,
        endpoint_ids=pfr_bundle.idc, block_size=block_size,
    )
    mechanics_lower_test = apply_lower_correction(pfr_bundle.raw_lot, q_physics).astype(np.float32)

    metrics_rows = []

    # Proposal stage.
    stage = f"model:{dataset}:safegrip_pfr"
    proposal_metrics_file = out / "safegrip_pfr_metrics.json"
    if is_done(stage) and proposal_metrics_file.exists():
        metrics_rows.append(json.loads(proposal_metrics_file.read_text()))
    else:
        time_guard(stage, 900)
        seed_everything(SEED)
        t0 = time.time()
        fit = _fit_pfr_gru(pfr_bundle, local_cfg, hp)
        residual_test, sigma_test, *_ = _predict_pfr(fit, pfr_bundle.Xt)
        point_test = compose_raw_prediction(pfr_bundle.raw_lot, residual_test).astype(np.float32)
        residual_cal, sigma_cal, *_ = _predict_pfr(fit, pfr_bundle.Xc)
        point_cal = compose_raw_prediction(pfr_bundle.raw_loc, residual_cal).astype(np.float32)
        q_safe = conformal_safe_correction(
            point_cal, pfr_bundle.yc, sigma_cal, alpha=statistical_alpha,
            scale_floor=float(hp["scale_floor"]), endpoint_ids=pfr_bundle.idc, block_size=block_size,
        )
        stat_lower = statistical_safe_lower(point_test, sigma_test, q_safe).astype(np.float32)
        safe = fuse_safe_lower(mechanics_lower_test, stat_lower, float(local_cfg.get("mu_upper", 1.3))).astype(np.float32)
        row = {
            "dataset": dataset, "model": "safegrip_pfr", "seed": SEED, "epochs": EPOCHS,
            **regression_metrics(pfr_bundle.yt, point_test),
            "safe_lower_coverage": float(np.mean(safe <= pfr_bundle.yt + 1e-8)),
            "seconds": float(time.time() - t0),
        }
        proposal_metrics_file.write_text(json.dumps(row, indent=2), encoding="utf-8")
        pd.DataFrame({
            "endpoint_id": pfr_bundle.idt, "y_true": pfr_bundle.yt,
            "prediction": point_test, "safe_prediction": safe,
            "mechanics_lower": mechanics_lower_test, "statistical_lower": stat_lower,
            "predicted_scale": sigma_test,
        }).to_csv(out / "safegrip_pfr_predictions.csv", index=False)
        torch_path = out / "safegrip_pfr_model.pt"
        import torch
        torch.save({"state_dict": fit.model.state_dict(), "seed": SEED, "epochs": EPOCHS}, torch_path)
        metrics_rows.append(row)
        mark_done(stage)

    # Each literature baseline is its own checkpointable stage.
    for name in names:
        stage = f"model:{dataset}:{name}"
        metric_file = out / f"{name}_metrics.json"
        if is_done(stage) and metric_file.exists():
            metrics_rows.append(json.loads(metric_file.read_text()))
            continue
        time_guard(stage, 600)
        bhp = _effective_baseline_hparams(name, local_cfg, baseline_selected, "controlled", "paper")
        bhp["epochs"] = EPOCHS
        bhp["patience"] = EPOCHS
        b = make_bundle(
            csv_path, pfr_cfg, sequence_length=int(bhp["sequence_length"]),
            scaler_kind=str(bhp["scaler"]), eval_start=eval_start, feature_mode="raw",
        )
        seed_everything(SEED)
        t0 = time.time()
        model = fit_literature(name, b, local_cfg, epochs=EPOCHS, preset="paper", hp_overrides=bhp)
        pred, sigma = predict_literature(model, name, b.Xt)
        row = {
            "dataset": dataset, "model": name, "seed": SEED, "epochs": EPOCHS,
            **regression_metrics(b.yt, pred, sigma=sigma),
            "seconds": float(time.time() - t0),
        }
        metric_file.write_text(json.dumps(row, indent=2), encoding="utf-8")
        pd.DataFrame({"endpoint_id": b.idt, "y_true": b.yt, "prediction": pred}).to_csv(
            out / f"{name}_predictions.csv", index=False
        )
        metrics_rows.append(row)
        mark_done(stage)
        del model, b
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        except Exception:
            pass

    pd.DataFrame(metrics_rows).to_csv(out / "metrics.csv", index=False)
    return metrics_rows


# ----------------------------- D2-D4 RECORD BUILDING -----------------------
def _infer_slip_deg(name: str):
    low = name.lower()
    if "13" in low and "slip" in low: return 13.0
    if "6" in low and "slip" in low: return 6.0
    if "long" in low or "0" in low: return 0.0
    return None


def build_real_records(dataset: str):
    """Build real SensorRecord windows using the repository's dataset adapters."""
    import pandas as pd
    from safegrip.data import read_table
    from safegrip.sensor_io.uc3m_tire import uc3m_tire_frame_to_record
    from safegrip.sensor_io.deep_dynamics_iac import deep_dynamics_iac_frame_to_record
    from safegrip.sensor_io.io_vnbd import io_vnbd_frame_to_record

    cache = RUN_ROOT / "record_cache" / f"{dataset}.pkl"
    if cache.exists():
        with open(cache, "rb") as f:
            records = pickle.load(f)
        print(f"[{dataset}] loaded {len(records)} cached records")
        return records

    raw = REPO / "data" / "raw" / dataset
    records = []

    def add_record(rec):
        if rec.targets and any(tv.mask for tv in rec.targets.values()):
            records.append(rec)
            if MAX_RECORDS_PER_DATASET > 0 and len(records) >= MAX_RECORDS_PER_DATASET:
                return True
        return False

    if dataset == "uc3m_tire":
        files = sorted(list(raw.rglob("*.xlsx")) + list(raw.rglob("*.xls")) + list(raw.rglob("*.csv")))
        for path in files:
            time_guard(f"building {dataset} records", 180)
            try:
                tables = pd.read_excel(path, sheet_name=None) if path.suffix.lower() in {".xlsx", ".xls"} else {"csv": read_table(path)}
            except Exception:
                continue
            for sheet, df in tables.items():
                if df is None or len(df) < 4: continue
                group = f"{path.relative_to(raw)}::{sheet}"
                win, stride = min(128, len(df)), 64
                starts = [0] if len(df) <= win else list(range(0, len(df)-win+1, stride))
                for j, s in enumerate(starts):
                    chunk = df.iloc[s:s+win].copy()
                    try:
                        rec = uc3m_tire_frame_to_record(
                            chunk, sequence_id=f"{group}#{j}", slip_angle_deg=_infer_slip_deg(path.name)
                        )
                        if add_record(rec): break
                    except Exception:
                        pass
                if MAX_RECORDS_PER_DATASET > 0 and len(records) >= MAX_RECORDS_PER_DATASET: break
            if MAX_RECORDS_PER_DATASET > 0 and len(records) >= MAX_RECORDS_PER_DATASET: break

    elif dataset == "deep_dynamics_iac":
        files = sorted(raw.rglob("*.csv"))
        for path in files:
            time_guard(f"building {dataset} records", 180)
            try: df = read_table(path)
            except Exception: continue
            if len(df) < 8: continue
            group = str(path.relative_to(raw))
            win, stride = min(65, len(df)), 16
            if win < 2: continue
            starts = [0] if len(df) <= win else list(range(0, len(df)-win+1, stride))
            for j, s in enumerate(starts):
                try:
                    rec = deep_dynamics_iac_frame_to_record(df.iloc[s:s+win].copy(), sequence_id=f"{group}#{j}")
                    if add_record(rec): break
                except Exception:
                    pass
            if MAX_RECORDS_PER_DATASET > 0 and len(records) >= MAX_RECORDS_PER_DATASET: break

    elif dataset == "io_vnbd":
        files = sorted(list(raw.rglob("*.csv")) + list(raw.rglob("*.txt")))
        for path in files:
            time_guard(f"building {dataset} records", 180)
            try: df = read_table(path)
            except Exception: continue
            if len(df) < 8: continue
            group = str(path.relative_to(raw))
            win, stride = min(64, len(df)), 32
            starts = [0] if len(df) <= win else list(range(0, len(df)-win+1, stride))
            for j, s in enumerate(starts):
                try:
                    rec = io_vnbd_frame_to_record(df.iloc[s:s+win].copy(), sequence_id=f"{group}#{j}")
                    if add_record(rec): break
                except Exception:
                    pass
            if MAX_RECORDS_PER_DATASET > 0 and len(records) >= MAX_RECORDS_PER_DATASET: break
    else:
        raise ValueError(dataset)

    cache.parent.mkdir(parents=True, exist_ok=True)
    with open(cache, "wb") as f:
        pickle.dump(records, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[{dataset}] built {len(records)} real labelled records")
    return records


def group_key(record):
    return str(record.sequence_id).split("#", 1)[0]


def split_records(records, seed=SEED):
    """Deterministic group split where possible; sample split only if <3 groups."""
    import numpy as np
    groups = sorted({group_key(r) for r in records})
    rng = np.random.default_rng(seed)
    if len(groups) >= 3:
        perm = list(np.asarray(groups)[rng.permutation(len(groups))])
        n = len(perm)
        n_train = max(1, int(round(0.6*n)))
        n_val = max(1, int(round(0.2*n)))
        if n_train + n_val >= n:
            n_train = max(1, n - 2); n_val = 1
        gtr = set(perm[:n_train]); gva = set(perm[n_train:n_train+n_val]); gte = set(perm[n_train+n_val:])
        tr = [r for r in records if group_key(r) in gtr]
        va = [r for r in records if group_key(r) in gva]
        te = [r for r in records if group_key(r) in gte]
        policy = "group_holdout"
    else:
        idx = rng.permutation(len(records))
        n = len(idx)
        a = max(1, int(0.6*n)); b = max(a+1, int(0.8*n))
        b = min(b, n-1) if n > 2 else b
        tr = [records[i] for i in idx[:a]]
        va = [records[i] for i in idx[a:b]]
        te = [records[i] for i in idx[b:]]
        policy = "sample_fallback_too_few_groups"
    if not tr or not te:
        raise RuntimeError(f"Cannot construct non-empty train/test split from {len(records)} records")
    return tr, va, te, policy


# ----------------------------- UNIVERSAL PROPOSAL --------------------------
def make_research_batches(records, tokenizer, dataset, batch_size):
    from safegrip.universal.dataset import collate_paper_records
    batches = []
    for s in range(0, len(records), batch_size):
        batches.append(collate_paper_records(records[s:s+batch_size], tokenizer, dataset))
    return batches


def train_universal_proposal(dataset, train_records, val_records, test_records, split_policy):
    import numpy as np
    import pandas as pd
    import torch
    from safegrip.ablations import get_ablation
    from safegrip.model.safegrip_universal import UniversalSafeGrip
    from safegrip.training.trainer import ResearchLossConfig, estimate_target_scales, train_step, research_losses
    from safegrip.universal.tokenizer import UniversalSensorTokenizer, TokenizerConfig
    from safegrip.universal.queries import queries_for_dataset

    stage = f"model:{dataset}:universal_safegrip"
    out = RESULTS_ROOT / dataset
    out.mkdir(parents=True, exist_ok=True)
    metric_file = out / "universal_safegrip_metrics.csv"
    if is_done(stage) and metric_file.exists():
        return pd.read_csv(metric_file).to_dict("records")

    seed_all(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ab = get_ablation("full")
    tokenizer = UniversalSensorTokenizer(TokenizerConfig(use_spectrum=ab.spectrum))
    print(f"[{dataset}] tokenizing proposal batches ...")
    train_batches = make_research_batches(train_records, tokenizer, dataset, UNIVERSAL_BATCH)
    val_batches = make_research_batches(val_records, tokenizer, dataset, UNIVERSAL_BATCH) if val_records else []
    test_batches = make_research_batches(test_records, tokenizer, dataset, UNIVERSAL_BATCH)
    scales = estimate_target_scales(train_batches)

    model = UniversalSafeGrip(tokenizer.feature_dim, ablation=ab).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_cfg = ResearchLossConfig(sensor_dropout=ab.sensor_dropout)
    ckpt = out / "universal_safegrip_train_state.pt"
    start_epoch = 0
    if ckpt.exists() and not is_done(stage):
        payload = torch.load(ckpt, map_location=device)
        model.load_state_dict(payload["model"])
        opt.load_state_dict(payload["optimizer"])
        start_epoch = int(payload.get("epoch", 0))
        print(f"[{dataset}] resuming UniversalSafeGrip at epoch {start_epoch}/{EPOCHS}")

    epoch_rows = []
    for epoch in range(start_epoch, EPOCHS):
        time_guard(f"{stage} epoch {epoch+1}", 180)
        model.train()
        order = list(range(len(train_batches)))
        random.Random(SEED + epoch).shuffle(order)
        losses = []
        for bi in order:
            b = train_batches[bi].to(device)
            ls = train_step(model, b, opt, scales, loss_cfg)
            losses.append(float(ls.total.detach().cpu()))
        val_loss = float("nan")
        if val_batches:
            model.eval(); vals = []
            with torch.no_grad():
                for b in val_batches:
                    ls = research_losses(model, b.to(device), scales, loss_cfg, training=False)
                    vals.append(float(ls.total.detach().cpu()))
            val_loss = float(np.mean(vals)) if vals else float("nan")
        epoch_rows.append({"epoch": epoch+1, "train_loss": float(np.mean(losses)), "val_loss": val_loss})
        pd.DataFrame(epoch_rows).to_csv(out / "universal_safegrip_training_log_current_session.csv", index=False)
        torch.save({"epoch": epoch+1, "model": model.state_dict(), "optimizer": opt.state_dict()}, ckpt)
        STATE.setdefault("progress", {})[stage] = {"epoch": epoch+1, "of": EPOCHS}
        save_state()
        if (epoch + 1) % 2 == 0:
            make_checkpoint_zip()

    # Test evaluation.
    model.eval()
    queries = queries_for_dataset(dataset)
    names = [q.name or q.quantity for q in queries]
    all_y = [[] for _ in names]; all_p = [[] for _ in names]
    pred_rows = []
    cursor = 0
    with torch.no_grad():
        for b in test_batches:
            bd = b.to(device)
            pred = model(bd.sensors, bd.queries, domains=bd.domains).point.detach().cpu().numpy()
            y = b.target_values.numpy(); mask = b.target_mask.numpy()
            for i in range(len(y)):
                row = {"sample": cursor+i}
                for j, name in enumerate(names):
                    row[f"true_{name}"] = float(y[i,j]) if mask[i,j] else np.nan
                    row[f"pred_{name}"] = float(pred[i,j]) if mask[i,j] else np.nan
                    if mask[i,j]:
                        all_y[j].append(float(y[i,j])); all_p[j].append(float(pred[i,j]))
                pred_rows.append(row)
            cursor += len(y)
    metrics = []
    for j, name in enumerate(names):
        metrics.append({
            "dataset": dataset, "model": "universal_safegrip", "target": name,
            "seed": SEED, "epochs": EPOCHS, "split_policy": split_policy,
            **metric_dict(all_y[j], all_p[j]),
        })
    pd.DataFrame(metrics).to_csv(metric_file, index=False)
    pd.DataFrame(pred_rows).to_csv(out / "universal_safegrip_predictions.csv", index=False)
    torch.save({"state_dict": model.state_dict(), "seed": SEED, "epochs": EPOCHS}, out / "universal_safegrip_model.pt")
    if ckpt.exists(): ckpt.unlink()
    mark_done(stage)
    del model, opt, train_batches, val_batches, test_batches
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return metrics


# ----------------------------- D2-D4 BASELINES -----------------------------
def record_to_matrix(record, input_dim, steps=64):
    import numpy as np
    if not record.channels:
        return None
    t0, t1 = record.start_time, record.end_time
    if not np.isfinite(t0) or not np.isfinite(t1): return None
    if t1 <= t0: t1 = t0 + 1.0
    grid = np.linspace(t0, t1, steps)
    x = np.zeros((steps, input_dim), dtype=np.float32)
    for j, ch in enumerate(record.channels[:input_dim]):
        v = np.asarray(ch.values, dtype=float); t = np.asarray(ch.timestamps, dtype=float)
        m = np.isfinite(v) & np.isfinite(t)
        if not np.any(m): continue
        tt, vv = t[m], v[m]
        order = np.argsort(tt); tt, vv = tt[order], vv[order]
        uu, idx = np.unique(tt, return_index=True); vv = vv[idx]
        if len(uu) == 1: x[:,j] = float(vv[0])
        else: x[:,j] = np.interp(grid, uu, vv).astype(np.float32)
    return x


def target_vector(record, target_names):
    import numpy as np
    vals = []
    for name in target_names:
        tv = record.targets.get(name)
        if tv is None or not tv.mask: return None
        arr = np.asarray(tv.value, dtype=float).reshape(-1)
        arr = arr[np.isfinite(arr)]
        if len(arr) == 0: return None
        vals.append(float(np.mean(arr)))
    return np.asarray(vals, dtype=np.float32)


def build_baseline_xy(records, target_names, input_dim):
    import numpy as np
    xs, ys = [], []
    for r in records:
        y = target_vector(r, target_names)
        if y is None: continue
        x = record_to_matrix(r, input_dim)
        if x is None or not np.isfinite(x).all(): continue
        xs.append(x); ys.append(y)
    if not xs:
        return None, None
    return np.stack(xs), np.stack(ys)


def train_one_baseline(dataset, name, train_records, test_records, split_policy):
    import numpy as np
    import pandas as pd
    import torch
    from safegrip.paper_baselines import make_paper_baseline
    from safegrip.baseline_training import baseline_loss

    stage = f"model:{dataset}:{name}"
    out = RESULTS_ROOT / dataset
    metric_file = out / f"{name}_metrics.csv"
    if is_done(stage) and metric_file.exists():
        return pd.read_csv(metric_file).to_dict("records")

    input_dim = D2D4_INPUT_DIMS[dataset]
    build = make_paper_baseline(name, input_dim, debug_scale=False)
    xtr, ytr = build_baseline_xy(train_records, build.target_names, input_dim)
    xte, yte = build_baseline_xy(test_records, build.target_names, input_dim)
    if xtr is None or xte is None or len(xtr) < 2 or len(xte) < 1:
        reason = f"Insufficient explicit targets for {name}: train={0 if xtr is None else len(xtr)}, test={0 if xte is None else len(xte)}"
        STATE.setdefault("skipped", {})[stage] = reason
        save_state(); make_checkpoint_zip()
        print("[skip]", reason)
        return []

    seed_all(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build.model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    ckpt = out / f"{name}_train_state.pt"
    start_epoch = 0
    if ckpt.exists() and not is_done(stage):
        payload = torch.load(ckpt, map_location=device)
        model.load_state_dict(payload["model"]); opt.load_state_dict(payload["optimizer"])
        start_epoch = int(payload.get("epoch", 0))
        print(f"[{dataset}/{name}] resuming epoch {start_epoch}/{EPOCHS}")

    X = torch.from_numpy(xtr); Y = torch.from_numpy(ytr)
    n = len(X)
    for epoch in range(start_epoch, EPOCHS):
        time_guard(f"{stage} epoch {epoch+1}", 180)
        model.train()
        order = torch.randperm(n, generator=torch.Generator().manual_seed(SEED + epoch))
        losses = []
        for s in range(0, n, BASELINE_BATCH):
            idx = order[s:s+BASELINE_BATCH]
            xb = X[idx].to(device); yb = Y[idx].to(device)
            opt.zero_grad(set_to_none=True)
            loss = baseline_loss(model, xb, yb)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss in {dataset}/{name} epoch {epoch+1}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); losses.append(float(loss.detach().cpu()))
        print(f"[{dataset}/{name}] epoch {epoch+1:02d}/{EPOCHS} loss={np.mean(losses):.6g}")
        torch.save({"epoch": epoch+1, "model": model.state_dict(), "optimizer": opt.state_dict()}, ckpt)
        STATE.setdefault("progress", {})[stage] = {"epoch": epoch+1, "of": EPOCHS}
        save_state()
        if (epoch + 1) % 2 == 0: make_checkpoint_zip()

    model.eval()
    preds = []
    with torch.no_grad():
        for s in range(0, len(xte), BASELINE_BATCH):
            xb = torch.from_numpy(xte[s:s+BASELINE_BATCH]).to(device)
            preds.append(model(xb).detach().cpu().numpy())
    pred = np.concatenate(preds, axis=0)
    rows = []
    for j, target in enumerate(build.target_names):
        rows.append({
            "dataset": dataset, "model": name, "target": target,
            "seed": SEED, "epochs": EPOCHS, "split_policy": split_policy,
            **metric_dict(yte[:,j], pred[:,j]),
        })
    pd.DataFrame(rows).to_csv(metric_file, index=False)
    pred_df = {}
    for j, target in enumerate(build.target_names):
        pred_df[f"true_{target}"] = yte[:,j]; pred_df[f"pred_{target}"] = pred[:,j]
    pd.DataFrame(pred_df).to_csv(out / f"{name}_predictions.csv", index=False)
    torch.save({"state_dict": model.state_dict(), "seed": SEED, "epochs": EPOCHS}, out / f"{name}_model.pt")
    if ckpt.exists(): ckpt.unlink()
    mark_done(stage)
    del model, opt, X, Y
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return rows


def run_d2d4_dataset(dataset):
    import pandas as pd
    ensure_dataset(dataset)
    stage_records = f"records:{dataset}"
    records = build_real_records(dataset)
    if not records:
        reason = "SKIPPED_NO_EXPLICIT_TARGETS: repository adapter found no real records carrying the registered task targets"
        STATE.setdefault("skipped", {})[dataset] = reason
        save_state(); make_checkpoint_zip()
        print(f"[{dataset}] {reason}")
        return []
    if not is_done(stage_records): mark_done(stage_records)

    train_records, val_records, test_records, policy = split_records(records)
    audit = {
        "dataset": dataset, "records": len(records), "train": len(train_records),
        "validation": len(val_records), "test": len(test_records), "split_policy": policy,
        "seed": SEED,
    }
    out = RESULTS_ROOT / dataset; out.mkdir(parents=True, exist_ok=True)
    (out / "split_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print("[split]", audit)

    rows = []
    rows.extend(train_universal_proposal(dataset, train_records, val_records, test_records, policy))
    for name in baselines_for_dataset(dataset):
        time_guard(f"baseline {dataset}/{name}", 300)
        rows.extend(train_one_baseline(dataset, name, train_records, test_records, policy))
    if rows:
        pd.DataFrame(rows).to_csv(out / "metrics.csv", index=False)
    return rows


# ----------------------------- MAIN EXPERIMENT -----------------------------
ALL_ROWS = []
try:
    seed_all(SEED)
    print("\nSafeGrip all-dataset run")
    print("commit:", STATE.get("git_commit"))
    print("seed:", SEED, "epochs:", EPOCHS)
    print("device will be selected per stage; CUDA available check happens after torch import")

    # D1 first because it is the primary complete PFR-ECR experiment.
    ALL_ROWS.extend(run_d1_lira())

    # D2-D4 use the universal proposal path + their registered comparator families.
    for dataset in ("uc3m_tire", "deep_dynamics_iac", "io_vnbd"):
        time_guard(f"dataset {dataset}", 300)
        try:
            ALL_ROWS.extend(run_d2d4_dataset(dataset))
        except TimeBudgetReached:
            raise
        except Exception as exc:
            # Preserve completed datasets/models and continue to the next dataset.
            mark_error(f"dataset_run:{dataset}", exc)
            print(f"[{dataset}] ERROR: {type(exc).__name__}: {exc}")

except TimeBudgetReached as exc:
    print("\n[TIME GUARD]", exc)
    STATE["stopped_for_time_limit"] = True
    STATE["stop_reason"] = str(exc)
    save_state()
    make_checkpoint_zip()

finally:
    # Rebuild a global summary from all persisted per-model metric files, so a
    # resumed run does not depend on in-memory rows from earlier sessions.
    try:
        import pandas as pd
        frames = []
        for p in RESULTS_ROOT.rglob("*metrics.csv"):
            try:
                df = pd.read_csv(p)
                if not df.empty: frames.append(df)
            except Exception:
                pass
        # LiRA proposal/baseline metrics are JSON per model; also include them.
        for p in (RESULTS_ROOT / "lira_cd").glob("*_metrics.json") if (RESULTS_ROOT / "lira_cd").exists() else []:
            try: frames.append(pd.DataFrame([json.loads(p.read_text())]))
            except Exception: pass
        if frames:
            summary = pd.concat(frames, ignore_index=True, sort=False).drop_duplicates()
            summary.to_csv(RUN_ROOT / "all_metrics.csv", index=False)
            print("\n=== CURRENT METRICS ===")
            print(summary.to_string(index=False))
    except Exception as exc:
        print("[summary warning]", exc)

    STATE["elapsed_hours_this_session"] = round((time.time() - START_TIME)/3600.0, 4)
    save_state()
    make_checkpoint_zip()

    # Final result archive is always generated, even if the 11-hour guard fired.
    if FINAL_ZIP.exists(): FINAL_ZIP.unlink()
    shutil.make_archive(str(FINAL_ZIP.with_suffix("")), "zip", root_dir=RUN_ROOT.parent, base_dir=RUN_ROOT.name)

    print("\n" + "=" * 100)
    print("RUN FINISHED / CHECKPOINTED")
    print("Results folder :", RUN_ROOT)
    print("Checkpoint ZIP:", CHECKPOINT_ZIP)
    print("Result ZIP    :", FINAL_ZIP)
    print("State         :", STATE_PATH)
    print("Completed     :", len(STATE.get("completed", [])), "stages")
    if STATE.get("skipped"):
        print("Skipped       :", json.dumps(STATE["skipped"], indent=2))
    if STATE.get("errors"):
        print("Errors        :", json.dumps(STATE["errors"], indent=2))
    print("\nTo resume in a NEW Kaggle session: save the checkpoint ZIP as notebook output, attach it as input, then rerun this same cell.")
