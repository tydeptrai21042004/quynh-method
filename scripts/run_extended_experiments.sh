#!/usr/bin/env bash
set -euo pipefail
python -m pip install -q -e '.[paper]'
DATASET="${DATASET:-lira}"
PRESET="${PRESET:-paper}"
RESULTS="${RESULTS:-results/${DATASET}_${PRESET}}"
HP="${HP:-results/${DATASET}_tuning/best_hparams.yaml}"

safegrip experiment --dataset "$DATASET" --study excitation --results "$RESULTS"
safegrip experiment --dataset "$DATASET" --study robustness --results "$RESULTS"

HP_ARGS=()
if [[ -f "$HP" ]]; then HP_ARGS=(--proposal-hparams "$HP"); fi
safegrip experiment --dataset "$DATASET" --study scarcity --preset "$PRESET" "${HP_ARGS[@]}"

# Cross-route is only scientifically valid when explicit route IDs are present.
if ! safegrip experiment --dataset "$DATASET" --study cross-route --preset "$PRESET" "${HP_ARGS[@]}"; then
  echo "[info] cross-route experiment skipped: fewer than two explicit route IDs or insufficient route splits."
fi
