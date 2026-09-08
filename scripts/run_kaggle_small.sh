#!/usr/bin/env bash
set -euo pipefail

# Small REAL-DATA development run. This verifies the complete LiRA -> SafeGrip
# -> two literature baselines -> ablation pipeline. It is not a final paper run.
CONFIG="${CONFIG:-configs/kaggle_small.yaml}"

python -m pytest -q
safegrip --config "$CONFIG" download --datasets lira
safegrip --config "$CONFIG" prepare --dataset lira

# Proposal + two lightweight literature-backed comparator families.
safegrip --config "$CONFIG" benchmark \
  --dataset lira \
  --preset quick \
  --models todorovic2022_cnn,lampe2023_gru

# Full component ablation in one-seed / three-epoch development mode.
safegrip --config "$CONFIG" ablation \
  --dataset lira \
  --preset quick \
  --variants safegrip_data_only,safegrip_no_projection,safegrip_no_uq,safegrip_no_physics_loss,safegrip_no_calibration,safegrip_no_temporal,safegrip

echo "Small LiRA run complete."
echo "Main metrics: results/lira_quick/metrics.csv"
echo "Ablation:     results/lira_ablation_quick/ablation_metrics.csv"
echo "Diagnostics:  data/processed/lira/lira_friction_schema_report.csv"
echo "              data/processed/lira/lira_stream_assembly_report.json"
echo "              data/processed/lira/lira_alignment_report.csv"
echo "              data/processed/lira/lira_preprocessing_report.json"
