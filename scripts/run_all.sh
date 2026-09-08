#!/usr/bin/env bash
set -euo pipefail
PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHONPATH="$(pwd)/src${PYTHONPATH:+:${PYTHONPATH}}"
${PYTHON_BIN} -m pip install -q -e .
SG="${PYTHON_BIN} -m safegrip.cli"
DATASET="${DATASET:-lira}"
if [[ "$DATASET" != "synthetic" ]]; then $SG download --datasets "$DATASET"; fi
$SG prepare --dataset "$DATASET"
$SG benchmark --dataset "$DATASET" --preset quick
printf '\nDone: results/%s_quick\n' "$DATASET"
