#!/usr/bin/env bash
set -euo pipefail
python -m pip install -q -e '.[paper]'
DATASET="${DATASET:-lira}"
TRIALS="${TRIALS:-30}"
safegrip tune --dataset "$DATASET" --trials "$TRIALS"
safegrip tune-baselines --dataset "$DATASET" --trials "$TRIALS"
