#!/usr/bin/env bash
set -euo pipefail
PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHONPATH="$(pwd)/src${PYTHONPATH:+:${PYTHONPATH}}"
${PYTHON_BIN} -m pip install -q -e ".[dev]"
SG="${PYTHON_BIN} -m safegrip.cli"
CONFIG="${CONFIG:-configs/kaggle_small.yaml}"

${PYTHON_BIN} -m pytest -q
$SG --config "$CONFIG" download --datasets lira
$SG --config "$CONFIG" prepare --dataset lira

# One-seed / three-epoch development run.  This is a pipeline check, not a paper result.
$SG --config "$CONFIG" benchmark \
  --dataset lira \
  --preset quick \
  --proposal pfr \
  --protocol controlled \
  --models todorovic2022_cnn,lampe2023_gru

$SG --config "$CONFIG" statistics \
  --results results/lira_quick \
  --proposal safegrip_pfr \
  --bootstrap 500

echo "SafeGrip-PFR small LiRA run complete."
echo "Main metrics:      results/lira_quick/metrics.csv"
echo "PFR ablation:      results/lira_quick/pfr_component_ablation.csv"
echo "Projection theorem: results/lira_quick/pfr_theorem_audit.json"
