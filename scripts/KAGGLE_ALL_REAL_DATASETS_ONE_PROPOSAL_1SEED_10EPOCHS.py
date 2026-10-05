# ================================================================
# SafeGrip — ONE PROPOSAL / ALL REAL DATASETS / PAPER BASELINES ONLY
# Kaggle single-cell-compatible driver (paste this whole file in one cell).
#
# Proposal on D1-D4: UniversalSafeGrip (same architecture + training method).
# Dataset-specific adapters/queries are measurement interfaces, not new models.
#
# D1 LiRA-CD baselines: Du2023, Todorovic2022, Lampe2023, Levenberg2023.
# D2 UC3M baselines: Mendoza-Petit2019, Yunta2018.
# D3 real IAC baselines: Chrosniak2024 Deep Dynamics, Fang/Yu2025 FTHD.
# D4 IO-VNBD baselines: Onyekpe2021 QGRU, Onyekpe2021 WhONet.
#
# REAL-DATA CONTRACT:
#   * no synthetic/simulated fallback is generated;
#   * D3 explicitly excludes Bayesrace simulator NPZ files;
#   * D4 records are built from synchronized real vehicle CSVs and use
#     GPS geodesic displacement as ground truth;
#   * if a required real target/source cannot be resolved, the stage is
#     recorded as SKIPPED_REAL_DATA_UNAVAILABLE, never filled synthetically.
#
# Resume:
#   * checkpoint after completed stages and every 2 neural epochs;
#   * stop before Kaggle's 11 h wall time and zip everything completed;
#   * attach the checkpoint ZIP as a Kaggle input and rerun this same cell.
# ================================================================
from __future__ import annotations

import gc
import json
import math
import os
import pickle
import random
import shutil
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

# ------------------------------- SETTINGS ---------------------------------
REPO_URL = os.environ.get("SAFEGRIP_REPO", "https://github.com/tydeptrai21042004/quynh-method.git")
BRANCH = os.environ.get("SAFEGRIP_BRANCH", "main")
SEED = int(os.environ.get("SAFEGRIP_SEED", "3101"))
EPOCHS = int(os.environ.get("SAFEGRIP_EPOCHS", "10"))
MAX_WALL_HOURS = float(os.environ.get("SAFEGRIP_MAX_HOURS", "11"))
SAVE_BUFFER_MIN = int(os.environ.get("SAFEGRIP_SAVE_BUFFER_MIN", "12"))
# 0 = all real records. Set e.g. 20000 for a deliberately bounded real-data run.
MAX_RECORDS_PER_DATASET = int(os.environ.get("SAFEGRIP_MAX_RECORDS", "0"))
UNIVERSAL_BATCH = int(os.environ.get("SAFEGRIP_BATCH", "32"))
BASELINE_BATCH = int(os.environ.get("SAFEGRIP_BASELINE_BATCH", "128"))
RUN_TESTS = os.environ.get("SAFEGRIP_RUN_TESTS", "1") not in {"0", "false", "False"}

DATASETS = ("lira_cd", "uc3m_tire", "deep_dynamics_iac", "io_vnbd")
_requested_datasets = tuple(
    x.strip().lower() for x in os.environ.get("SAFEGRIP_DATASETS", ",".join(DATASETS)).split(",") if x.strip()
)
if not _requested_datasets or any(x not in DATASETS for x in _requested_datasets):
    raise ValueError(f"SAFEGRIP_DATASETS must be a non-empty subset of {DATASETS}")
RUN_DATASETS = _requested_datasets
PROPOSAL_ABLATION = os.environ.get("SAFEGRIP_ABLATION", "npi_v3").strip().lower()
PROPOSAL_BASE_NAME = "universal_safegrip"
PROPOSAL_NAME = (
    PROPOSAL_BASE_NAME if PROPOSAL_ABLATION == "npi_v3"
    else f"{PROPOSAL_BASE_NAME}__ablation_{PROPOSAL_ABLATION}"
)
PROPOSAL_REVISION = "normalized_physical_innovation_v3"
RUN_BASELINES = os.environ.get("SAFEGRIP_RUN_BASELINES", "1") not in {"0", "false", "False"}
D2D4_INPUT_DIMS = {"uc3m_tire": 3, "deep_dynamics_iac": 5, "io_vnbd": 4}
REAL_IAC_FILES = (
    "LVMS_23_01_04_A.csv",
    "LVMS_23_01_04_B.csv",
    "Putnam_park2023_run2_1.csv",
    "Putnam_park2023_run4_1.csv",
    "Putnam_park2023_run4_2.csv",
)

WORK = Path("/kaggle/working")
REPO = WORK / "quynh-method"
RUN_NAME = os.environ.get("SAFEGRIP_RUN_NAME", "safegrip_one_proposal_all_real_1seed_10epochs")
RUN_ROOT = WORK / RUN_NAME
RESULTS_ROOT = RUN_ROOT / "results"
STATE_PATH = RUN_ROOT / "state.json"
CONFIG_PATH = RUN_ROOT / "kaggle_one_proposal_real.yaml"
CHECKPOINT_ZIP = WORK / f"{RUN_NAME}_checkpoint.zip"
FINAL_ZIP = WORK / f"{RUN_NAME}_results.zip"

START_TIME = time.time()
HARD_DEADLINE = START_TIME + MAX_WALL_HOURS * 3600.0
SAFE_DEADLINE = HARD_DEADLINE - SAVE_BUFFER_MIN * 60.0


class TimeBudgetReached(RuntimeError):
    pass


class RealDataUnavailable(RuntimeError):
    pass


def sh(cmd, *, cwd=None, check=True, timeout=None):
    printable = " ".join(map(str, cmd)) if isinstance(cmd, (list, tuple)) else str(cmd)
    print("\n" + "=" * 100)
    print("RUN:", printable, flush=True)
    print("=" * 100, flush=True)
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        check=check,
        timeout=timeout,
        shell=isinstance(cmd, str),
        executable="/bin/bash" if isinstance(cmd, str) else None,
    )


def remaining_seconds() -> float:
    return max(0.0, SAFE_DEADLINE - time.time())


def time_guard(label="next stage", minimum_seconds=180.0):
    left = remaining_seconds()
    if left < minimum_seconds:
        raise TimeBudgetReached(
            f"Stopping before 11h limit: {left/60:.1f} safe minutes remain before {label}."
        )


def atomic_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str), encoding="utf-8")
    tmp.replace(path)


def restore_previous_checkpoint():
    if STATE_PATH.exists():
        return
    candidates = sorted(Path("/kaggle/input").rglob(CHECKPOINT_ZIP.name))
    if candidates:
        print(f"[resume] restoring {candidates[-1]}")
        shutil.unpack_archive(str(candidates[-1]), str(WORK))


def load_state():
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    return {
        "version": 6,
        "proposal": PROPOSAL_NAME,
        "proposal_revision": PROPOSAL_REVISION,
        "real_data_only": True,
        "seed": SEED,
        "epochs": EPOCHS,
        "repo_url": REPO_URL,
        "git_commit": None,
        "completed": [],
        "progress": {},
        "errors": {},
        "skipped": {},
        "data_audit": {},
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def save_state():
    STATE["updated_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    atomic_json(STATE_PATH, STATE)


def make_checkpoint_zip():
    save_state()
    if CHECKPOINT_ZIP.exists():
        CHECKPOINT_ZIP.unlink()
    shutil.make_archive(
        str(CHECKPOINT_ZIP.with_suffix("")), "zip",
        root_dir=RUN_ROOT.parent, base_dir=RUN_ROOT.name,
    )
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


def mark_skipped(stage: str, reason: str):
    STATE.setdefault("skipped", {})[stage] = reason
    save_state()
    make_checkpoint_zip()


def seed_all(seed: int):
    random.seed(seed)
    import numpy as np
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def metric_dict(y, pred):
    import numpy as np
    y = np.asarray(y, dtype=float).reshape(-1)
    pred = np.asarray(pred, dtype=float).reshape(-1)
    m = np.isfinite(y) & np.isfinite(pred)
    y, pred = y[m], pred[m]
    if not len(y):
        return {"n": 0, "rmse": float("nan"), "mae": float("nan"), "r2": float("nan")}
    rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
    mae = float(np.mean(np.abs(pred - y)))
    den = float(np.sum((y - np.mean(y)) ** 2))
    r2 = float(1.0 - np.sum((pred - y) ** 2) / den) if den > 1e-12 else float("nan")
    return {"n": int(len(y)), "rmse": rmse, "mae": mae, "r2": r2}


def cap_records(records):
    if MAX_RECORDS_PER_DATASET <= 0 or len(records) <= MAX_RECORDS_PER_DATASET:
        return records
    # Deterministic even coverage, never synthetic sampling/replacement.
    import numpy as np
    idx = np.linspace(0, len(records) - 1, MAX_RECORDS_PER_DATASET, dtype=int)
    return [records[int(i)] for i in idx]


# ----------------------------- RESTORE / CLONE -----------------------------
restore_previous_checkpoint()
RUN_ROOT.mkdir(parents=True, exist_ok=True)
RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
STATE = load_state()

# Proposal revisions change the meaning of the shared decoder.  Real-data
# caches and completed literature baselines remain valid, but proposal weights
# and proposal metrics must be retrained under the current reference semantics.
def _invalidate_incompatible_proposal_state():
    completed = [
        stage for stage in STATE.get("completed", [])
        if not (stage.startswith("model:") and stage.endswith(f":{PROPOSAL_NAME}"))
    ]
    STATE["completed"] = completed
    STATE["progress"] = {
        k: v for k, v in STATE.get("progress", {}).items()
        if not (k.startswith("model:") and k.endswith(f":{PROPOSAL_NAME}"))
    }
    for dataset in RUN_DATASETS:
        out = RESULTS_ROOT / dataset
        for name in (
            f"{PROPOSAL_NAME}_metrics.csv",
            f"{PROPOSAL_NAME}_predictions.csv",
            f"{PROPOSAL_NAME}_model.pt",
            f"{PROPOSAL_NAME}_train_state.pt",
            f"{PROPOSAL_NAME}_training_log.csv",
        ):
            q = out / name
            if q.exists():
                q.unlink()


if STATE.get("proposal") == PROPOSAL_NAME and (
    STATE.get("version") != 6 or STATE.get("proposal_revision") != PROPOSAL_REVISION
):
    old_revision = STATE.get("proposal_revision", "legacy")
    print(f"[resume] migrating {old_revision} -> {PROPOSAL_REVISION}; proposal stages will retrain")
    _invalidate_incompatible_proposal_state()
    STATE["version"] = 6
    STATE["proposal_revision"] = PROPOSAL_REVISION
    STATE["migration_note"] = (
        f"proposal outputs from {old_revision} invalidated; real-data caches and "
        "publication-backed baseline stages preserved"
    )
    save_state()

if (
    STATE.get("version") != 6
    or STATE.get("proposal") != PROPOSAL_NAME
    or STATE.get("proposal_revision") != PROPOSAL_REVISION
):
    raise RuntimeError("Checkpoint belongs to a different proposal family; use a fresh output directory.")

if STATE.get("seed") != SEED or STATE.get("epochs") != EPOCHS:
    raise RuntimeError(
        f"Checkpoint seed/epochs={STATE.get('seed')}/{STATE.get('epochs')} but requested {SEED}/{EPOCHS}."
    )

if not REPO.exists():
    sh(["git", "clone", "--depth", "1", "--branch", BRANCH, REPO_URL, str(REPO)])
else:
    print(f"[repo] reusing {REPO}")

if STATE.get("git_commit"):
    wanted = STATE["git_commit"]
    current = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    if current != wanted:
        sh(["git", "fetch", "--depth", "1", "origin", wanted], cwd=REPO, check=False)
        sh(["git", "checkout", wanted], cwd=REPO)
else:
    STATE["git_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    save_state()

sh([sys.executable, "-m", "pip", "install", "-q", "-e", ".[paper,dev]"], cwd=REPO)
sys.path.insert(0, str(REPO / "src"))

if RUN_TESTS and not is_done("repo_tests"):
    time_guard("repository tests", 300)
    sh([
        sys.executable, "-m", "pytest", "-q",
        "tests/test_registry.py", "tests/test_d2_d4_baselines.py",
        "tests/test_universal_training.py", "tests/test_sensor_io.py",
    ], cwd=REPO, timeout=max(300, int(remaining_seconds() - 60)))
    mark_done("repo_tests")

# ------------------------------- CONFIG -----------------------------------
import yaml
cfg = yaml.safe_load((REPO / "configs/default.yaml").read_text(encoding="utf-8"))
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
CONFIG_PATH.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

from safegrip.config import load_config
from safegrip.datasets import DATASET_REGISTRY, PAPER_DATASETS
from safegrip.literature import LITERATURE_BASELINES, baselines_for_dataset, validate_paper_baselines
from safegrip.universal.schema import SensorChannel, SensorMeta, SensorRecord, TargetValue

if tuple(PAPER_DATASETS) != DATASETS:
    raise RuntimeError(f"Repository paper datasets changed: {PAPER_DATASETS}")


def enforce_paper_baseline_contract(dataset: str):
    names = tuple(baselines_for_dataset(dataset))
    validate_paper_baselines(names, dataset=dataset, require_runnable=True)
    bad_words = ("dummy", "synthetic", "placeholder", "random_baseline")
    for name in names:
        meta = LITERATURE_BASELINES[name]
        if not meta.get("doi") or meta.get("paper_verified") is not True:
            raise RuntimeError(f"{dataset}/{name} is not verified as a publication-backed comparator")
        hay = " ".join([name, str(meta.get("display_name", "")), str(meta.get("title", ""))]).lower()
        if any(w in hay for w in bad_words):
            raise RuntimeError(f"Non-paper/dummy baseline rejected: {dataset}/{name}")
    return names


PAPER_BASELINE_AUDIT = {
    ds: [
        {
            "name": name,
            "doi": LITERATURE_BASELINES[name]["doi"],
            "title": LITERATURE_BASELINES[name]["title"],
            "paper_verified": LITERATURE_BASELINES[name].get("paper_verified", False),
            "implementation_kind": LITERATURE_BASELINES[name].get("implementation_kind", "paper_common_input_adaptation"),
        }
        for name in enforce_paper_baseline_contract(ds)
    ]
    for ds in DATASETS
}
atomic_json(RUN_ROOT / "paper_baseline_audit.json", PAPER_BASELINE_AUDIT)

# --------------------------- DOWNLOAD / PREPARE ----------------------------
def source_manifest(dataset: str) -> dict:
    p = REPO / "data/raw" / dataset / "SOURCE.json"
    if not p.exists():
        raise RealDataUnavailable(f"Missing real-source manifest: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def audit_real_download(dataset: str):
    raw = REPO / "data/raw" / dataset
    manifest = source_manifest(dataset)
    files = [p for p in raw.rglob("*") if p.is_file()]
    if not files:
        raise RealDataUnavailable(f"{dataset}: no downloaded real files")
    # Strong rejection: the experiment never consumes files whose path itself
    # advertises synthetic/simulated data. D3 is handled by an explicit allow-list.
    suspicious = [str(p.relative_to(raw)) for p in files if "synthetic" in str(p).lower()]
    if suspicious:
        raise RealDataUnavailable(f"{dataset}: synthetic path(s) found in selected source: {suspicious[:3]}")
    audit = {
        "dataset": dataset,
        "dataset_doi": DATASET_REGISTRY[dataset].get("dataset_doi"),
        "download_kind": DATASET_REGISTRY[dataset].get("download_kind"),
        "source_manifest": manifest,
        "downloaded_file_count": len(files),
        "real_data_only": True,
    }
    STATE.setdefault("data_audit", {})[dataset] = audit
    save_state()
    return audit


def ensure_dataset(dataset: str):
    stage = f"data:{dataset}"
    raw = REPO / "data/raw" / dataset
    if not (raw / ".complete").exists():
        time_guard(f"download {dataset}", 300)
        sh([
            sys.executable, "-m", "safegrip.cli", "--config", str(CONFIG_PATH),
            "download", "--datasets", dataset,
        ], cwd=REPO, timeout=max(300, int(remaining_seconds() - 60)))
    audit_real_download(dataset)
    # Preparation is required for LiRA and useful as a source check for D2-D4.
    proc = REPO / "data/processed" / dataset
    if not proc.exists() or not any(proc.rglob("*")):
        time_guard(f"prepare {dataset}", 300)
        try:
            sh([
                sys.executable, "-m", "safegrip.cli", "--config", str(CONFIG_PATH),
                "prepare", "--dataset", dataset,
            ], cwd=REPO, timeout=max(300, int(remaining_seconds() - 60)))
        except subprocess.CalledProcessError as exc:
            # Compatibility for an older checkout where LiRA group-holdout cannot
            # be constructed from the public bundle. This is a REAL-data split
            # fallback, not a data/model fallback.
            if dataset != "lira_cd":
                raise
            local = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
            local.setdefault("split", {})["mode"] = "spatial_within_trajectory"
            local["split"]["insufficient_group_policy"] = "error"
            CONFIG_PATH.write_text(yaml.safe_dump(local, sort_keys=False), encoding="utf-8")
            STATE.setdefault("data_audit", {}).setdefault(dataset, {})["split_fallback"] = (
                "group_holdout unavailable in downloaded real LiRA bundle; retried spatial_within_trajectory"
            )
            save_state()
            sh([
                sys.executable, "-m", "safegrip.cli", "--config", str(CONFIG_PATH),
                "prepare", "--dataset", dataset,
            ], cwd=REPO, timeout=max(300, int(remaining_seconds() - 60)))
    if not is_done(stage):
        mark_done(stage)


# ------------------------- REAL RECORD BUILDERS ----------------------------
def _record_cache_path(dataset: str):
    return RUN_ROOT / "record_cache" / f"{dataset}.pkl"


def save_record_cache(dataset: str, payload):
    p = _record_cache_path(dataset)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_record_cache(dataset: str):
    p = _record_cache_path(dataset)
    if not p.exists():
        return None
    with open(p, "rb") as f:
        return pickle.load(f)


def build_lira_records():
    import pandas as pd
    from safegrip.sensor_io.lira_cd import lira_cd_frame_to_record

    cached = load_record_cache("lira_cd")
    if cached is not None:
        return cached
    csv_path = REPO / "data/processed/lira_cd/lira_aligned.csv"
    if not csv_path.exists():
        raise RealDataUnavailable("LiRA real prepared file lira_aligned.csv is missing")
    df = pd.read_csv(csv_path)
    need = {"mu_ref", "split"}
    if not need.issubset(df.columns):
        raise RealDataUnavailable(f"LiRA prepared data missing columns: {sorted(need - set(df.columns))}")
    group_col = "segment_id" if "segment_id" in df else ("trip_id" if "trip_id" in df else None)
    if group_col is None:
        raise RealDataUnavailable("LiRA prepared data lacks segment/trip identifier")
    win, stride = 64, 4
    buckets = {"train": [], "calibration": [], "validation": [], "test": []}
    for (group, split), g in df.groupby([group_col, "split"], sort=False):
        split = str(split)
        if split not in buckets:
            continue
        g = g.reset_index(drop=True)
        for start in range(0, max(0, len(g) - win + 1), stride):
            chunk = g.iloc[start:start + win].copy()
            endpoint = float(pd.to_numeric(chunk["mu_ref"], errors="coerce").iloc[-1])
            if not math.isfinite(endpoint):
                continue
            try:
                rec = lira_cd_frame_to_record(chunk, sequence_id=f"{group}#{start+win-1}")
                rec = replace(rec, targets={"friction": TargetValue("friction", endpoint)})
                buckets[split].append(rec)
            except Exception:
                continue
    for k in buckets:
        buckets[k] = cap_records(buckets[k])
    if not buckets["train"] or not buckets["test"]:
        raise RealDataUnavailable("LiRA real windows produced empty train/test records")
    split_effective = str(df["split_mode_effective"].iloc[0]) if "split_mode_effective" in df else "prepared_split"
    audit = {
        "source": "DTU LiRA-CD real vehicle + VIAFRIK friction reference",
        "target_source": "real_mu_ref_endpoint",
        "split_policy": split_effective,
        "counts": {k: len(v) for k, v in buckets.items()},
    }
    STATE["data_audit"]["lira_cd"].update(audit)
    save_state()
    payload = (buckets["train"], buckets["validation"] or buckets["calibration"], buckets["test"], split_effective)
    save_record_cache("lira_cd", payload)
    return payload


def _infer_slip_deg(name: str):
    low = name.lower()
    if "13" in low and "slip" in low:
        return 13.0
    if "6" in low and "slip" in low:
        return 6.0
    if "long" in low or "0" in low:
        return 0.0
    return None


def build_uc3m_records():
    import pandas as pd
    from safegrip.data import read_table
    from safegrip.sensor_io.uc3m_tire import uc3m_tire_frame_to_record

    cached = load_record_cache("uc3m_tire")
    if cached is not None:
        return cached
    raw = REPO / "data/raw/uc3m_tire"
    files = sorted(list(raw.rglob("*.xlsx")) + list(raw.rglob("*.xls")) + list(raw.rglob("*.csv")))
    if not files:
        raise RealDataUnavailable("UC3M real Dataverse deposit contains no readable tables")
    records = []
    for path in files:
        time_guard("building real UC3M records", 180)
        try:
            tables = pd.read_excel(path, sheet_name=None) if path.suffix.lower() in {".xlsx", ".xls"} else {"csv": read_table(path)}
        except Exception:
            continue
        for sheet, df in tables.items():
            if df is None or len(df) < 4:
                continue
            win, stride = min(128, len(df)), 64
            starts = [0] if len(df) <= win else range(0, len(df) - win + 1, stride)
            for j, st in enumerate(starts):
                try:
                    rec = uc3m_tire_frame_to_record(
                        df.iloc[st:st+win].copy(),
                        sequence_id=f"{path.relative_to(raw)}::{sheet}#{j}",
                        slip_angle_deg=_infer_slip_deg(path.name),
                    )
                except Exception:
                    continue
                if rec.targets and any(tv.mask for tv in rec.targets.values()):
                    records.append(rec)
                    if MAX_RECORDS_PER_DATASET > 0 and len(records) >= MAX_RECORDS_PER_DATASET:
                        break
            if MAX_RECORDS_PER_DATASET > 0 and len(records) >= MAX_RECORDS_PER_DATASET:
                break
        if MAX_RECORDS_PER_DATASET > 0 and len(records) >= MAX_RECORDS_PER_DATASET:
            break
    if not records:
        raise RealDataUnavailable("UC3M adapter found no records with explicit real tire targets")
    STATE["data_audit"]["uc3m_tire"].update({
        "source": "U6ICRX real tire-test Dataverse deposit",
        "target_source": "real U6ICRX experiment slip-angle condition (0/6/13 deg); no force labels synthesized",
        "record_count": len(records),
    })
    save_state()
    save_record_cache("uc3m_tire", records)
    return records


def build_iac_records():
    from safegrip.data import read_table
    from safegrip.sensor_io.deep_dynamics_iac import deep_dynamics_iac_frame_to_record

    cached = load_record_cache("deep_dynamics_iac")
    if cached is not None:
        return cached
    raw = REPO / "data/raw/deep_dynamics_iac"
    files = []
    missing = []
    for name in REAL_IAC_FILES:
        found = list(raw.rglob(name))
        if found:
            files.append(found[0])
        else:
            missing.append(name)
    if not files:
        raise RealDataUnavailable(f"No real IAC racecar CSVs found; missing {missing}")
    # Hard guarantee: no Bayesrace .npz is ever consumed here.
    assert all(p.suffix.lower() == ".csv" and p.name in REAL_IAC_FILES for p in files)
    records = []
    for path in files:
        df = read_table(path)
        if len(df) < 65:
            continue
        win, stride = 65, 16
        for j, st in enumerate(range(0, len(df) - win + 1, stride)):
            try:
                rec = deep_dynamics_iac_frame_to_record(df.iloc[st:st+win].copy(), sequence_id=f"{path.name}#{j}")
            except Exception:
                continue
            if len(rec.targets) == 3:
                records.append(rec)
                if MAX_RECORDS_PER_DATASET > 0 and len(records) >= MAX_RECORDS_PER_DATASET:
                    break
        if MAX_RECORDS_PER_DATASET > 0 and len(records) >= MAX_RECORDS_PER_DATASET:
            break
    if not records:
        raise RealDataUnavailable("Real IAC CSVs yielded no labeled next-state records")
    STATE["data_audit"]["deep_dynamics_iac"].update({
        "source": "five AV-21 Indy Autonomous Challenge real-racecar CSVs only",
        "selected_files": [p.name for p in files],
        "excluded_simulator": "Bayesrace DYN-PP-ETHZ*.npz explicitly not consumed",
        "target_source": "next measured real vehicle state",
        "record_count": len(records),
    })
    save_state()
    save_record_cache("deep_dynamics_iac", records)
    return records


def haversine_m(lat1, lon1, lat2, lon2):
    # Ground-truth displacement from real GPS coordinates. No synthetic labels.
    r = 6371008.8
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2*r*math.asin(min(1.0, math.sqrt(max(0.0, a))))


def build_io_vnbd_records():
    import pandas as pd
    from safegrip.sensor_io.io_vnbd import (
        io_vnbd_frame_to_record, normalize_io_vnbd_vehicle_frame,
    )

    cached = load_record_cache("io_vnbd")
    if cached is not None:
        return cached
    raw = REPO / "data/raw/io_vnbd"
    sync_files = sorted(
        p for p in raw.rglob("V-*.csv")
        if "unsynchron" not in str(p.parent).lower()
    )
    if not sync_files:
        raise RealDataUnavailable(
            "No synchronized real IO-VNBD vehicle CSVs found after resolving the Git-LFS payload"
        )

    records = []
    used_files = []
    for path in sync_files:
        time_guard("building real IO-VNBD records", 180)
        try:
            raw_frame = pd.read_csv(path, header=None, sep=None, engine="python")
            numeric = normalize_io_vnbd_vehicle_frame(raw_frame)
        except Exception:
            continue
        if len(numeric) < 11:
            continue
        used_this_file = 0
        # The public synchronized vehicle stream is nominally 10 Hz.  Each
        # record contains one second of measured history; GPS is used only to
        # form the real displacement target at the following endpoint.
        for st in range(0, len(numeric) - 10, 10):
            sub = numeric.iloc[st:st+11].reset_index(drop=True)
            lat1, lon1 = float(sub.loc[0, "latitude_deg"]), float(sub.loc[0, "longitude_deg"])
            lat2, lon2 = float(sub.loc[10, "latitude_deg"]), float(sub.loc[10, "longitude_deg"])
            if not all(map(math.isfinite, (lat1, lon1, lat2, lon2))):
                continue
            if not (-90 <= lat1 <= 90 and -90 <= lat2 <= 90 and -180 <= lon1 <= 180 and -180 <= lon2 <= 180):
                continue
            disp = haversine_m(lat1, lon1, lat2, lon2)
            if not math.isfinite(disp) or not (0.0 <= disp <= 100.0):
                continue
            try:
                rec = io_vnbd_frame_to_record(
                    sub.copy(),
                    sequence_id=f"{path.name}#{st//10}",
                    sample_rate_hz=10.0,
                    displacement_m=disp,
                )
            except Exception:
                continue
            if "displacement" not in rec.targets:
                continue
            records.append(rec)
            used_this_file += 1
            if MAX_RECORDS_PER_DATASET > 0 and len(records) >= MAX_RECORDS_PER_DATASET:
                break
        if used_this_file:
            used_files.append(path.name)
        if MAX_RECORDS_PER_DATASET > 0 and len(records) >= MAX_RECORDS_PER_DATASET:
            break
    if not records:
        raise RealDataUnavailable(
            "IO-VNBD synchronized real GPS/vehicle measurements produced no valid displacement windows"
        )
    STATE["data_audit"]["io_vnbd"].update({
        "source": "synchronized IO-VNBD research-vehicle public-road CSVs",
        "selected_file_count": len(used_files),
        "selected_files_preview": used_files[:20],
        "target_source": "real GPS latitude/longitude one-second geodesic displacement",
        "input_source": "measured wheel speeds + indicated speed + yaw rate + accelerations + steering",
        "record_count": len(records),
        "synthetic_targets": False,
    })
    save_state()
    save_record_cache("io_vnbd", records)
    return records


def group_key(record):
    return str(record.sequence_id).split("#", 1)[0]


def split_records(records):
    import numpy as np
    groups = sorted({group_key(r) for r in records})
    rng = np.random.default_rng(SEED)
    if len(groups) >= 3:
        perm = list(np.asarray(groups)[rng.permutation(len(groups))])
        n = len(perm)
        ntr = max(1, int(round(0.6*n)))
        nva = max(1, int(round(0.2*n)))
        if ntr + nva >= n:
            ntr, nva = max(1, n-2), 1
        gtr = set(perm[:ntr]); gva = set(perm[ntr:ntr+nva]); gte = set(perm[ntr+nva:])
        tr = [r for r in records if group_key(r) in gtr]
        va = [r for r in records if group_key(r) in gva]
        te = [r for r in records if group_key(r) in gte]
        policy = "group_holdout_by_real_source_file"
    else:
        # Still real data, but disclose weaker split if archive exposes too few groups.
        idx = rng.permutation(len(records))
        a = max(1, int(0.6*len(idx))); b = max(a+1, int(0.8*len(idx)))
        b = min(b, len(idx)-1) if len(idx) > 2 else b
        tr = [records[i] for i in idx[:a]]; va = [records[i] for i in idx[a:b]]; te = [records[i] for i in idx[b:]]
        policy = "sample_split_real_data_too_few_groups"
    if not tr or not te:
        raise RealDataUnavailable(f"Cannot form non-empty real train/test split from {len(records)} records")
    return tr, va, te, policy


def _split_real_builder(builder):
    def build():
        return split_records(builder())
    return build


REAL_RECORD_BUNDLES = {
    "lira_cd": build_lira_records,
    "uc3m_tire": _split_real_builder(build_uc3m_records),
    "deep_dynamics_iac": _split_real_builder(build_iac_records),
    "io_vnbd": _split_real_builder(build_io_vnbd_records),
}


def real_records_for(dataset: str):
    ensure_dataset(dataset)
    try:
        build = REAL_RECORD_BUNDLES[dataset]
    except KeyError as exc:
        raise ValueError(f"no real-record builder registered for {dataset}") from exc
    tr, va, te, policy = build()
    STATE["data_audit"][dataset]["split_policy"] = policy
    STATE["data_audit"][dataset]["split_counts"] = {
        "train": len(tr), "validation": len(va), "test": len(te),
    }
    save_state()
    return tr, va, te, policy


# -------------------------- ONE SHARED PROPOSAL ----------------------------
def make_research_batches(records, tokenizer, dataset, batch_size):
    from safegrip.universal.dataset import collate_paper_records
    return [
        collate_paper_records(records[s:s+batch_size], tokenizer, dataset)
        for s in range(0, len(records), batch_size)
    ]


def train_universal_proposal(dataset, train_records, val_records, test_records, split_policy):
    import numpy as np
    import pandas as pd
    import torch
    from safegrip.ablations import get_ablation
    from safegrip.model.safegrip_universal import UniversalSafeGrip
    from safegrip.training.trainer import (
        ResearchLossConfig, estimate_target_scales, estimate_innovation_normalizer,
        train_step, research_losses,
    )
    from safegrip.universal.tokenizer import UniversalSensorTokenizer, TokenizerConfig
    from safegrip.universal.queries import queries_for_dataset

    stage = f"model:{dataset}:{PROPOSAL_NAME}"
    out = RESULTS_ROOT / dataset
    out.mkdir(parents=True, exist_ok=True)
    metric_file = out / f"{PROPOSAL_NAME}_metrics.csv"
    if is_done(stage) and metric_file.exists():
        return pd.read_csv(metric_file).to_dict("records")

    seed_all(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ab = get_ablation(PROPOSAL_ABLATION)
    if PROPOSAL_ABLATION == "npi_v3" and ab.dataset_id_conditioning:
        raise RuntimeError("NPI-v3 proposal unexpectedly enables dataset-ID conditioning")
    tokenizer = UniversalSensorTokenizer(TokenizerConfig(use_spectrum=ab.spectrum))
    print(f"[{dataset}] tokenizing REAL records for the shared proposal ...")
    train_batches = make_research_batches(train_records, tokenizer, dataset, UNIVERSAL_BATCH)
    val_batches = make_research_batches(val_records, tokenizer, dataset, UNIVERSAL_BATCH) if val_records else []
    test_batches = make_research_batches(test_records, tokenizer, dataset, UNIVERSAL_BATCH)
    # Fit NPI residual/direct coordinates on TRAINING data only. These are
    # deterministic semantic buffers, not learned heads or dataset-ID parameters.
    innovation_normalizer = estimate_innovation_normalizer(
        train_batches,
        mode=ab.innovation_mode,
        localization_reference=ab.localization_reference,
    )
    scales = estimate_target_scales(train_batches)

    # SAME constructor and optimizer on every dataset.
    model = UniversalSafeGrip(
        tokenizer.feature_dim, ablation=ab, innovation_normalizer=innovation_normalizer
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_cfg = ResearchLossConfig(sensor_dropout=ab.sensor_dropout)
    ckpt = out / f"{PROPOSAL_NAME}_train_state.pt"
    start_epoch = 0
    best_epoch = 0
    best_selection_loss = float("inf")
    best_model_state = None
    if ckpt.exists() and not is_done(stage):
        try:
            payload = torch.load(ckpt, map_location=device, weights_only=False)
        except TypeError:
            payload = torch.load(ckpt, map_location=device)
        if payload.get("proposal_revision") != PROPOSAL_REVISION:
            print(f"[{dataset}] ignoring incompatible proposal train-state checkpoint")
            ckpt.unlink()
        else:
            model.load_state_dict(payload["model"]); opt.load_state_dict(payload["optimizer"])
            start_epoch = int(payload.get("epoch", 0))
            best_epoch = int(payload.get("best_epoch", 0))
            best_selection_loss = float(payload.get("best_selection_loss", float("inf")))
            best_model_state = payload.get("best_model_state")
        print(f"[{dataset}] resume {PROPOSAL_NAME}: epoch {start_epoch}/{EPOCHS}, best={best_epoch}")

    log_file = out / f"{PROPOSAL_NAME}_training_log.csv"
    old_rows = pd.read_csv(log_file).to_dict("records") if log_file.exists() else []
    if ab.innovation_mode == "reference_only":
        # This control contains no learned correction by definition; evaluating
        # it directly avoids wasting epochs while keeping exactly the same split.
        print(f"[{dataset}] reference-only ablation: no optimization required")
        best_epoch = 0
        best_selection_loss = float("nan")
    else:
        for epoch in range(start_epoch, EPOCHS):
            time_guard(f"{stage} epoch {epoch+1}", 180)
            model.train()
            order = list(range(len(train_batches)))
            random.Random(SEED + epoch).shuffle(order)
            losses = []
            for bi in order:
                ls = train_step(model, train_batches[bi].to(device), opt, scales, loss_cfg)
                losses.append(float(ls.total.detach().cpu()))
            val_loss = float("nan")
            if val_batches:
                model.eval(); vals = []
                with torch.no_grad():
                    for b in val_batches:
                        vals.append(float(research_losses(model, b.to(device), scales, loss_cfg, training=False).total.detach().cpu()))
                val_loss = float(np.mean(vals)) if vals else float("nan")
            train_loss = float(np.mean(losses))
            selection_loss = val_loss if np.isfinite(val_loss) else train_loss
            if selection_loss < best_selection_loss:
                best_selection_loss = float(selection_loss)
                best_epoch = epoch + 1
                # Clone to CPU so later optimizer steps cannot mutate the selected
                # state through shared tensor storage.
                best_model_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            row = {
                "epoch": epoch+1, "train_loss": train_loss, "val_loss": val_loss,
                "selection_loss": selection_loss, "is_best": int(best_epoch == epoch + 1),
            }
            old_rows = [r for r in old_rows if int(r.get("epoch", -1)) != epoch+1] + [row]
            pd.DataFrame(old_rows).sort_values("epoch").to_csv(log_file, index=False)
            torch.save({
                "epoch": epoch+1, "model": model.state_dict(), "optimizer": opt.state_dict(),
                "best_epoch": best_epoch, "best_selection_loss": best_selection_loss,
                "best_model_state": best_model_state,
                "proposal_revision": PROPOSAL_REVISION,
                "ablation": PROPOSAL_ABLATION,
            }, ckpt)
            STATE.setdefault("progress", {})[stage] = {"epoch": epoch+1, "of": EPOCHS}
            save_state()
            if (epoch + 1) % 2 == 0:
                make_checkpoint_zip()

    # Evaluate the validation-selected checkpoint, never an arbitrary final
    # epoch. This is particularly important for the seed-sensitive LiRA run.
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"[{dataset}] restored best validation checkpoint: epoch {best_epoch}, loss={best_selection_loss:.6g}")
    model.eval()
    queries = queries_for_dataset(dataset)
    names = [q.name or q.quantity for q in queries]
    all_y = [[] for _ in names]; all_p = [[] for _ in names]; pred_rows = []; cursor = 0
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
        if not all_y[j]:
            continue
        metrics.append({
            "dataset": dataset, "model": PROPOSAL_NAME, "proposal_method": PROPOSAL_NAME,
            "proposal_revision": PROPOSAL_REVISION,
            "ablation": PROPOSAL_ABLATION,
            "innovation_mode": ab.innovation_mode,
            "localization_reference": ab.localization_reference,
            "target": name, "seed": SEED, "epochs": EPOCHS, "best_epoch": best_epoch,
            "split_policy": split_policy,
            "data_kind": "real", **metric_dict(all_y[j], all_p[j]),
        })
    pd.DataFrame(metrics).to_csv(metric_file, index=False)
    pd.DataFrame(pred_rows).to_csv(out / f"{PROPOSAL_NAME}_predictions.csv", index=False)
    torch.save({
        "state_dict": model.state_dict(), "seed": SEED, "epochs": EPOCHS,
        "best_epoch": best_epoch, "best_selection_loss": best_selection_loss,
        "proposal_revision": PROPOSAL_REVISION,
        "ablation": PROPOSAL_ABLATION,
    }, out / f"{PROPOSAL_NAME}_model.pt")
    if ckpt.exists(): ckpt.unlink()
    mark_done(stage)
    del model, opt, train_batches, val_batches, test_batches
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return metrics


# ------------------------------ D1 BASELINES -------------------------------
def train_lira_paper_baselines():
    import pandas as pd
    from safegrip.benchmark import make_bundle, fit_literature, predict_literature, regression_metrics
    from safegrip.pfr_benchmark import _effective_baseline_hparams, _common_eval_start_pfr, _pfr_bundle_cfg, _pfr_hparams
    from safegrip.utils import seed_everything

    dataset = "lira_cd"
    local_cfg = load_config(str(CONFIG_PATH))
    hp = _pfr_hparams(local_cfg)
    names = list(enforce_paper_baseline_contract(dataset))
    eval_start = _common_eval_start_pfr(local_cfg, hp, names, {}, "controlled", "paper")
    common_cfg = _pfr_bundle_cfg(local_cfg, hp)
    csv_path = REPO / "data/processed/lira_cd/lira_aligned.csv"
    out = RESULTS_ROOT / dataset; out.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in names:
        stage = f"model:{dataset}:{name}"
        metric_file = out / f"{name}_metrics.csv"
        if is_done(stage) and metric_file.exists():
            rows.extend(pd.read_csv(metric_file).to_dict("records")); continue
        time_guard(stage, 600)
        bhp = _effective_baseline_hparams(name, local_cfg, {}, "controlled", "paper")
        bhp["epochs"] = EPOCHS; bhp["patience"] = EPOCHS
        b = make_bundle(
            csv_path, common_cfg, sequence_length=int(bhp["sequence_length"]),
            scaler_kind=str(bhp["scaler"]), eval_start=eval_start, feature_mode="raw",
        )
        seed_everything(SEED)
        t0 = time.time()
        model = fit_literature(name, b, local_cfg, epochs=EPOCHS, preset="paper", hp_overrides=bhp)
        pred, sigma = predict_literature(model, name, b.Xt)
        row = {
            "dataset": dataset, "model": name, "target": "friction", "seed": SEED, "epochs": EPOCHS,
            "data_kind": "real", "paper_doi": LITERATURE_BASELINES[name]["doi"],
            **regression_metrics(b.yt, pred, sigma=sigma), "seconds": float(time.time()-t0),
        }
        pd.DataFrame([row]).to_csv(metric_file, index=False)
        pd.DataFrame({"endpoint_id": b.idt, "y_true": b.yt, "prediction": pred}).to_csv(out / f"{name}_predictions.csv", index=False)
        rows.append(row); mark_done(stage)
        del model, b; gc.collect()
    return rows


# ---------------------------- D2-D4 BASELINES ------------------------------
def record_to_matrix(record, input_dim, steps=64):
    import numpy as np
    if not record.channels:
        return None
    t0, t1 = record.start_time, record.end_time
    if not np.isfinite(t0) or not np.isfinite(t1):
        return None
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
        x[:, j] = float(vv[0]) if len(uu) == 1 else np.interp(grid, uu, vv).astype(np.float32)
    return x


def target_vector(record, target_names):
    import numpy as np
    vals = []
    for name in target_names:
        tv = record.targets.get(name)
        if tv is None or not tv.mask:
            return None
        arr = np.asarray(tv.value, dtype=float).reshape(-1)
        arr = arr[np.isfinite(arr)]
        if not len(arr): return None
        vals.append(float(np.mean(arr)))
    return np.asarray(vals, dtype=np.float32)


def build_baseline_xy(records, target_names, input_dim):
    import numpy as np
    xs, ys = [], []
    for r in records:
        y = target_vector(r, target_names)
        x = record_to_matrix(r, input_dim)
        if y is None or x is None or not np.isfinite(x).all():
            continue
        xs.append(x); ys.append(y)
    return (None, None) if not xs else (np.stack(xs), np.stack(ys))


def train_one_d2d4_baseline(dataset, name, train_records, test_records, split_policy):
    import numpy as np
    import pandas as pd
    import torch
    from safegrip.paper_baselines import make_paper_baseline
    from safegrip.baseline_training import baseline_loss

    meta = LITERATURE_BASELINES[name]
    if not meta.get("doi") or meta.get("paper_verified") is not True:
        raise RuntimeError(f"Refusing non-paper comparator {name}")
    stage = f"model:{dataset}:{name}"
    out = RESULTS_ROOT / dataset; out.mkdir(parents=True, exist_ok=True)
    metric_file = out / f"{name}_metrics.csv"
    if is_done(stage) and metric_file.exists():
        return pd.read_csv(metric_file).to_dict("records")

    input_dim = D2D4_INPUT_DIMS[dataset]
    built = make_paper_baseline(
        name, input_dim, debug_scale=False,
        target_names=tuple(DATASET_REGISTRY[dataset]["targets"]),
    )
    xtr, ytr = build_baseline_xy(train_records, built.target_names, input_dim)
    xte, yte = build_baseline_xy(test_records, built.target_names, input_dim)
    if xtr is None or xte is None or len(xtr) < 2 or len(xte) < 1:
        reason = f"SKIPPED_REAL_TARGET_UNAVAILABLE: {name} train={0 if xtr is None else len(xtr)}, test={0 if xte is None else len(xte)}"
        mark_skipped(stage, reason); print("[skip]", reason); return []

    seed_all(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = built.model.to(device)
    baseline_lr = {"onyekpe2021_whonet": 7e-4}.get(name, 1e-3)
    lr = baseline_lr
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    ckpt = out / f"{name}_train_state.pt"
    start_epoch = 0
    if ckpt.exists() and not is_done(stage):
        try: payload = torch.load(ckpt, map_location=device, weights_only=False)
        except TypeError: payload = torch.load(ckpt, map_location=device)
        if payload.get("proposal_revision") != PROPOSAL_REVISION:
            print(f"[{dataset}] ignoring incompatible proposal train-state checkpoint")
            ckpt.unlink()
        else:
            model.load_state_dict(payload["model"]); opt.load_state_dict(payload["optimizer"])
            start_epoch = int(payload.get("epoch", 0))

    X, Y = torch.from_numpy(xtr), torch.from_numpy(ytr)
    n = len(X)
    for epoch in range(start_epoch, EPOCHS):
        time_guard(f"{stage} epoch {epoch+1}", 180)
        model.train(); losses = []
        order = torch.randperm(n, generator=torch.Generator().manual_seed(SEED + epoch))
        for s in range(0, n, BASELINE_BATCH):
            idx = order[s:s+BASELINE_BATCH]
            xb, yb = X[idx].to(device), Y[idx].to(device)
            opt.zero_grad(set_to_none=True)
            loss = baseline_loss(model, xb, yb)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss in {dataset}/{name}")
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
            losses.append(float(loss.detach().cpu()))
        print(f"[{dataset}/{name}] epoch {epoch+1:02d}/{EPOCHS} loss={np.mean(losses):.6g}")
        torch.save({
            "epoch": epoch+1, "model": model.state_dict(), "optimizer": opt.state_dict(),
            "proposal_revision": PROPOSAL_REVISION,
        }, ckpt)
        STATE.setdefault("progress", {})[stage] = {"epoch": epoch+1, "of": EPOCHS}; save_state()
        if (epoch + 1) % 2 == 0: make_checkpoint_zip()

    model.eval(); preds = []
    with torch.no_grad():
        for s in range(0, len(xte), BASELINE_BATCH):
            preds.append(model(torch.from_numpy(xte[s:s+BASELINE_BATCH]).to(device)).detach().cpu().numpy())
    pred = np.concatenate(preds, axis=0)
    rows = []
    for j, target in enumerate(built.target_names):
        rows.append({
            "dataset": dataset, "model": name, "target": target, "seed": SEED, "epochs": EPOCHS,
            "split_policy": split_policy, "data_kind": "real", "paper_doi": meta["doi"],
            "implementation_kind": meta.get("implementation_kind", "paper_structured_local_reproduction"),
            **metric_dict(yte[:,j], pred[:,j]),
        })
    pd.DataFrame(rows).to_csv(metric_file, index=False)
    pred_df = {}
    for j, target in enumerate(built.target_names):
        pred_df[f"true_{target}"] = yte[:,j]; pred_df[f"pred_{target}"] = pred[:,j]
    pd.DataFrame(pred_df).to_csv(out / f"{name}_predictions.csv", index=False)
    torch.save({"state_dict": model.state_dict(), "seed": SEED, "epochs": EPOCHS}, out / f"{name}_model.pt")
    if ckpt.exists(): ckpt.unlink()
    mark_done(stage)
    del model, opt, X, Y; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return rows


# ------------------------------ MAIN ---------------------------------------
def _run_lira_baseline_set(dataset, train_records, test_records, policy):
    return train_lira_paper_baselines()


def _run_registered_baseline_set(dataset, train_records, test_records, policy):
    rows = []
    for name in enforce_paper_baseline_contract(dataset):
        time_guard(f"baseline {dataset}/{name}", 300)
        rows.extend(train_one_d2d4_baseline(dataset, name, train_records, test_records, policy))
    return rows


BASELINE_SET_RUNNERS = {
    "lira_cd": _run_lira_baseline_set,
    "uc3m_tire": _run_registered_baseline_set,
    "deep_dynamics_iac": _run_registered_baseline_set,
    "io_vnbd": _run_registered_baseline_set,
}


def run_dataset(dataset: str):
    import pandas as pd
    train_records, val_records, test_records, policy = real_records_for(dataset)
    out = RESULTS_ROOT / dataset; out.mkdir(parents=True, exist_ok=True)
    atomic_json(out / "real_data_audit.json", STATE["data_audit"][dataset])
    rows = []
    # SAME proposal for D1, D2, D3, D4.
    rows.extend(train_universal_proposal(dataset, train_records, val_records, test_records, policy))
    if RUN_BASELINES:
        rows.extend(BASELINE_SET_RUNNERS[dataset](dataset, train_records, test_records, policy))
    if rows:
        pd.DataFrame(rows).to_csv(out / "metrics.csv", index=False)
    return rows


ALL_ROWS = []
try:
    seed_all(SEED)
    print("\nSafeGrip: ONE proposal across ALL REAL datasets")
    print("proposal:", PROPOSAL_BASE_NAME, "revision:", PROPOSAL_REVISION, "ablation:", PROPOSAL_ABLATION)
    print("run model label:", PROPOSAL_NAME, "commit:", STATE.get("git_commit"))
    print("seed:", SEED, "epochs:", EPOCHS, "max_records:", MAX_RECORDS_PER_DATASET or "ALL")
    for dataset in RUN_DATASETS:
        time_guard(f"dataset {dataset}", 300)
        try:
            ALL_ROWS.extend(run_dataset(dataset))
        except TimeBudgetReached:
            raise
        except RealDataUnavailable as exc:
            reason = f"SKIPPED_REAL_DATA_UNAVAILABLE: {exc}"
            mark_skipped(f"dataset:{dataset}", reason)
            print(f"[{dataset}] {reason}")
        except Exception as exc:
            mark_error(f"dataset:{dataset}", exc)
            print(f"[{dataset}] ERROR: {type(exc).__name__}: {exc}")

except TimeBudgetReached as exc:
    print("\n[TIME GUARD]", exc)
    STATE["stopped_for_time_limit"] = True
    STATE["stop_reason"] = str(exc)
    save_state(); make_checkpoint_zip()

finally:
    try:
        import pandas as pd
        frames = []
        for p in RESULTS_ROOT.rglob("*metrics.csv"):
            try:
                df = pd.read_csv(p)
                if not df.empty: frames.append(df)
            except Exception:
                pass
        if frames:
            summary = pd.concat(frames, ignore_index=True, sort=False).drop_duplicates()
            summary.to_csv(RUN_ROOT / "all_metrics.csv", index=False)
            print("\n=== CURRENT REAL-DATA METRICS ===")
            print(summary.to_string(index=False))
    except Exception as exc:
        print("[summary warning]", exc)

    atomic_json(RUN_ROOT / "real_data_audit.json", STATE.get("data_audit", {}))
    atomic_json(RUN_ROOT / "paper_baseline_audit.json", PAPER_BASELINE_AUDIT)
    STATE["elapsed_hours_this_session"] = round((time.time() - START_TIME)/3600.0, 4)
    save_state(); make_checkpoint_zip()
    if FINAL_ZIP.exists(): FINAL_ZIP.unlink()
    shutil.make_archive(str(FINAL_ZIP.with_suffix("")), "zip", root_dir=RUN_ROOT.parent, base_dir=RUN_ROOT.name)

    print("\n" + "=" * 100)
    print("RUN FINISHED / CHECKPOINTED")
    print("One proposal    :", PROPOSAL_BASE_NAME, "NPI-v3")
    print("Ablation        :", PROPOSAL_ABLATION)
    print("Datasets        :", ", ".join(RUN_DATASETS))
    print("Synthetic data  : DISALLOWED")
    print("Results folder  :", RUN_ROOT)
    print("Checkpoint ZIP :", CHECKPOINT_ZIP)
    print("Result ZIP     :", FINAL_ZIP)
    print("Completed       :", len(STATE.get("completed", [])), "stages")
    if STATE.get("skipped"):
        print("Skipped:", json.dumps(STATE["skipped"], indent=2))
    if STATE.get("errors"):
        print("Errors:", json.dumps(STATE["errors"], indent=2))
    print("To resume in a new Kaggle session: attach the checkpoint ZIP as input and run this same cell again.")
