#!/usr/bin/env bash
set -euo pipefail
PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHONPATH="$(pwd)/src${PYTHONPATH:+:${PYTHONPATH}}"
${PYTHON_BIN} -m pip install -q -e .
SG="${PYTHON_BIN} -m safegrip.cli"
DATASET="${DATASET:-lira}"
$SG download --datasets "$DATASET"
$SG prepare --dataset "$DATASET"
$SG benchmark --dataset "$DATASET" --preset quick --proposal pfr --protocol controlled
printf '\nDone: results/%s_quick\n' "$DATASET"
