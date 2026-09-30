#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
CFG="${CFG:-configs/kaggle_trust.yaml}"
export PYTHONPATH="$(pwd)/src${PYTHONPATH:+:${PYTHONPATH}}"
${PYTHON_BIN} -m pip install -q -e ".[dev]"
SG="${PYTHON_BIN} -m safegrip.cli"
MODELS="du2023_inceptiontime,todorovic2022_cnn,lampe2023_gru,levenberg2023_stft"

$SG --config "$CFG" download --datasets lira_cd
$SG --config "$CFG" prepare --dataset lira_cd
$SG --config "$CFG" benchmark --dataset lira_cd --preset trust --models "$MODELS" --protocol controlled
$SG --config "$CFG" statistics --results results/lira_cd_trust --proposal safegrip_pfr --bootstrap 2000

echo "SafeGrip-PFR-ECR trust run complete."
echo "Main metrics:      results/lira_cd_trust/metrics.csv"
echo "PFR ablation:      results/lira_cd_trust/pfr_component_ablation.csv"
echo "Safety theorem:    results/lira_cd_trust/pfr_theorem_audit.json"
