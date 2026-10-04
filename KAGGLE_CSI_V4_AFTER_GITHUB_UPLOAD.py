# ================================================================
# Universal SafeGrip CSI-v4 — Kaggle fresh GitHub run
# ONE proposal method across ALL REAL datasets
# Proposal-only quick scientific run: unchanged literature baselines skipped
# ================================================================
from pathlib import Path
import os
import re
import runpy
import shutil
import subprocess
import sys
import pandas as pd

# ------------------------------- SETTINGS -------------------------------
REPO_URL = "https://github.com/tydeptrai21042004/quynh-method.git"
BRANCH = "main"
SEED = 3101
EPOCHS = 10
BATCH = 32
MAX_RECORDS = 0              # 0 = ALL real records
MAX_HOURS = 10.5
SAVE_BUFFER_MIN = 10
FRESH_RUN = True             # True = do not reuse an old result directory
RUN_RELEVANT_TESTS = True

WORK = Path("/kaggle/working")
REPO = WORK / "quynh-method"


def sh(cmd, cwd=None, check=True, capture=False):
    cmd = [str(x) for x in cmd]
    print("\n" + "=" * 100)
    print("RUN:", " ".join(cmd), flush=True)
    print("=" * 100, flush=True)
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        check=check,
        text=True,
        capture_output=capture,
    )


# --------------------------- 1. GPU / FRESH CLONE ----------------------
print("Python:", sys.version)
sh(["nvidia-smi"], check=False)

# Always clone the current GitHub state so Kaggle cannot silently reuse
# an older checkout from a previous notebook execution.
if REPO.exists():
    shutil.rmtree(REPO)

sh([
    "git", "clone", "--depth", "1", "--branch", BRANCH,
    REPO_URL, REPO,
])

commit = subprocess.check_output(
    ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
).strip()
print("\nGit commit:", commit)

RUN_NAME = f"safegrip_csi_v4_{commit[:8]}_seed{SEED}_{EPOCHS}ep"
RUN_ROOT = WORK / RUN_NAME
CHECKPOINT = WORK / f"{RUN_NAME}_checkpoint.zip"
RESULT_ZIP = WORK / f"{RUN_NAME}_results.zip"

RUNNER = REPO / "scripts" / "KAGGLE_ALL_REAL_DATASETS_ONE_PROPOSAL_1SEED_10EPOCHS.py"
QUICK_RUNNER = WORK / f"{RUN_NAME}_proposal_only.py"


# --------------------------- 2. VERIFY CSI-v4 --------------------------
required = [
    RUNNER,
    REPO / "src/safegrip/universal/feature_normalization.py",
    REPO / "src/safegrip/universal/innovation.py",
    REPO / "src/safegrip/model/safegrip_universal.py",
    REPO / "src/safegrip/training/trainer.py",
    REPO / "tests/test_calibrated_semantic_innovation.py",
]
missing = [p for p in required if not p.exists()]
if missing:
    raise FileNotFoundError(
        "CSI-v4 files are missing from GitHub:\n" + "\n".join(map(str, missing))
    )

runner_text = RUNNER.read_text(encoding="utf-8")
if 'PROPOSAL_REVISION = "calibrated_semantic_innovation_v4"' not in runner_text:
    raise RuntimeError(
        "GitHub does not appear to contain CSI-v4 yet. "
        "Push the changed files first, then rerun this cell."
    )

print("[OK] CSI-v4 revision found in GitHub checkout.")


# ------------------------------- 3. INSTALL -----------------------------
sh([sys.executable, "-m", "pip", "install", "-q", "-e", ".[paper,dev]"], cwd=REPO)


# --------------------------- 4. RELEVANT TESTS -------------------------
if RUN_RELEVANT_TESTS:
    tests = [
        "tests/test_calibrated_semantic_innovation.py",
        "tests/test_normalized_physical_innovation.py",
        "tests/test_universal_model.py",
        "tests/test_universal_training.py",
    ]
    sh([sys.executable, "-m", "pytest", "-q", *tests], cwd=REPO)


# ---------------------- 5. PROPOSAL-ONLY TEMP RUNNER -------------------
# Baselines are unchanged by CSI-v4, so skip retraining them for this test.
# This edits only a temporary Kaggle copy, never the cloned GitHub repo.
pattern = re.compile(
    r"^(?P<indent>[ \t]*)rows\.extend\(BASELINE_SET_RUNNERS\[dataset\]"
    r"\(dataset, train_records, test_records, policy\)\)\s*$",
    flags=re.MULTILINE,
)


def _skip_baselines(match):
    indent = match.group("indent")
    return indent + 'print(f"[{dataset}] CSI-v4 QUICK MODE: unchanged literature baselines skipped")'

quick_text, n_patches = pattern.subn(_skip_baselines, runner_text, count=1)
if n_patches != 1:
    raise RuntimeError(
        "Could not safely locate the baseline call in the GitHub runner. "
        "Stopped instead of patching the wrong code."
    )
QUICK_RUNNER.write_text(quick_text, encoding="utf-8")
print("[OK] Proposal-only temporary runner:", QUICK_RUNNER)


# ----------------------------- 6. FRESH OUTPUT --------------------------
if FRESH_RUN:
    if RUN_ROOT.exists():
        shutil.rmtree(RUN_ROOT)
    for p in (CHECKPOINT, RESULT_ZIP):
        if p.exists():
            p.unlink()
    print("[OK] Fresh CSI-v4 output requested.")


# ----------------------------- 7. ENVIRONMENT ---------------------------
os.environ["SAFEGRIP_REPO"] = REPO_URL
os.environ["SAFEGRIP_BRANCH"] = BRANCH
os.environ["SAFEGRIP_SEED"] = str(SEED)
os.environ["SAFEGRIP_SEEDS"] = str(SEED)
os.environ["SAFEGRIP_EPOCHS"] = str(EPOCHS)
os.environ["SAFEGRIP_BATCH"] = str(BATCH)
os.environ["SAFEGRIP_MAX_RECORDS"] = str(MAX_RECORDS)
os.environ["SAFEGRIP_MAX_HOURS"] = str(MAX_HOURS)
os.environ["SAFEGRIP_SAVE_BUFFER_MIN"] = str(SAVE_BUFFER_MIN)
os.environ["SAFEGRIP_RUN_TESTS"] = "0"   # already tested above
os.environ["SAFEGRIP_RUN_NAME"] = RUN_NAME
os.environ["PYTHONUNBUFFERED"] = "1"

print(f"""
================================================================================
UNIVERSAL SAFEGRIP — CALIBRATED SEMANTIC INNOVATION v4
Git commit : {commit}
Proposal   : universal_safegrip / calibrated_semantic_innovation_v4
Seed       : {SEED}
Epochs     : {EPOCHS}
Batch      : {BATCH}
Records    : {'ALL REAL RECORDS' if MAX_RECORDS == 0 else MAX_RECORDS}
Datasets   : LiRA-CD, UC3M tire, Deep Dynamics IAC, IO-VNBD
Baselines  : SKIPPED in this quick proposal-only rerun (unchanged code)
Synthetic  : DISALLOWED by repository runner
Run root   : {RUN_ROOT}
================================================================================
""")


# ------------------------------- 8. RUN ---------------------------------
runpy.run_path(str(QUICK_RUNNER), run_name="__main__")


# -------------------------- 9. VERIFY / DISPLAY -------------------------
ALL_METRICS = RUN_ROOT / "all_metrics.csv"
if not ALL_METRICS.exists():
    raise FileNotFoundError(
        f"Run completed without expected metrics file:\n{ALL_METRICS}\n"
        f"Inspect checkpoint/state under {RUN_ROOT}."
    )

df = pd.read_csv(ALL_METRICS)
proposal = df[
    df["model"].astype(str).str.lower().eq("universal_safegrip")
].copy()

if "proposal_revision" in proposal.columns:
    bad = proposal[
        proposal["proposal_revision"].astype(str) != "calibrated_semantic_innovation_v4"
    ]
    if not bad.empty:
        raise RuntimeError("Non-CSI-v4 proposal rows found in the fresh output.")

show_cols = [c for c in [
    "dataset", "target", "seed", "epochs", "best_epoch", "n",
    "mae", "rmse", "r2", "proposal_revision", "split_policy",
] if c in proposal.columns]

print("\n" + "=" * 100)
print("CSI-v4 PROPOSAL RESULTS")
print("=" * 100)
try:
    display(proposal[show_cols].sort_values(["dataset", "target"]).reset_index(drop=True))
except NameError:
    print(proposal[show_cols].sort_values(["dataset", "target"]).to_string(index=False))

print("\nArtifacts:")
print("  metrics    :", ALL_METRICS)
print("  checkpoint :", CHECKPOINT)
print("  result zip :", RESULT_ZIP)
print("  commit     :", commit)
