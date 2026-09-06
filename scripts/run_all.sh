#!/usr/bin/env bash
set -euo pipefail
python -m pip install -q -e '.[paper]'
DATASET="${DATASET:-lira}"
if [[ "$DATASET" != "synthetic" ]]; then safegrip download --datasets "$DATASET"; fi
safegrip prepare --dataset "$DATASET"
safegrip benchmark --dataset "$DATASET" --preset quick
printf '\nDone: results/%s_quick\n' "$DATASET"
