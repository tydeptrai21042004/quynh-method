#!/usr/bin/env bash
set -euo pipefail
PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHONPATH="$(pwd)/src${PYTHONPATH:+:${PYTHONPATH}}"
${PYTHON_BIN} -m pip install -q -e ".[dev]"
SG="${PYTHON_BIN} -m safegrip.cli"
# Small REAL-DATA development run. This verifies the complete LiRA -> SafeGrip
# -> two literature baselines -> ablation pipeline. It is not a final paper run.
CONFIG="${CONFIG:-configs/kaggle_small.yaml}"

${PYTHON_BIN} -m pytest -q
$SG --config "$CONFIG" download --datasets lira
$SG --config "$CONFIG" prepare --dataset lira

# Proposal + two lightweight literature-backed comparator families.
$SG --config "$CONFIG" benchmark \
  --dataset lira \
  --preset quick \
  --models todorovic2022_cnn,lampe2023_gru

# Full component ablation in one-seed / three-epoch development mode.
$SG --config "$CONFIG" ablation \
  --dataset lira \
  --preset quick \
  --variants safegrip_data_only,safegrip_static_only,safegrip_no_gate,safegrip_no_bound,safegrip_no_uq,safegrip_no_calibration,safegrip

echo "Small LiRA run complete."
echo "Main metrics: results/lira_quick/metrics.csv"
echo "Ablation:     results/lira_ablation_quick/ablation_metrics.csv"
echo "Diagnostics:  data/processed/lira/lira_friction_schema_report.csv"
echo "              data/processed/lira/lira_stream_assembly_report.json"
echo "              data/processed/lira/lira_alignment_report.csv"
echo "              data/processed/lira/lira_preprocessing_report.json"
