#!/usr/bin/env bash
set -euo pipefail
PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHONPATH="$(pwd)/src${PYTHONPATH:+:${PYTHONPATH}}"
${PYTHON_BIN} -m pip install -q -e ".[dev]"
SG="${PYTHON_BIN} -m safegrip.cli"
CONFIG="${CONFIG:-configs/kaggle_small.yaml}"
MODELS="du2023_inceptiontime,todorovic2022_cnn,lampe2023_gru,levenberg2023_stft"

${PYTHON_BIN} -m pytest -q
$SG --config "$CONFIG" download --datasets lira_cd
$SG --config "$CONFIG" prepare --dataset lira_cd

# Development smoke run only; the PFR benchmark itself writes component and
# safety ablations, so there is no separate legacy `ablation` CLI command.
$SG --config "$CONFIG" benchmark \
  --dataset lira_cd \
  --preset quick \
  --protocol controlled \
  --models "$MODELS"

$SG --config "$CONFIG" statistics \
  --results results/lira_cd_quick \
  --proposal safegrip_pfr \
  --bootstrap 500

echo "SafeGrip-PFR-ECR small LiRA run complete."
echo "Main metrics:      results/lira_cd_quick/metrics.csv"
echo "PFR ablation:      results/lira_cd_quick/pfr_component_ablation.csv"
echo "Safety theorem:    results/lira_cd_quick/pfr_theorem_audit.json"
