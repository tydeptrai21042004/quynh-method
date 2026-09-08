#!/usr/bin/env bash
set -euo pipefail
PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHONPATH="$(pwd)/src${PYTHONPATH:+:${PYTHONPATH}}"
${PYTHON_BIN} -m pip install -q -e .
SG="${PYTHON_BIN} -m safegrip.cli"
DATASET="${DATASET:-lira}"
TRIALS="${TRIALS:-30}"
$SG tune --dataset "$DATASET" --trials "$TRIALS"
$SG tune-baselines --dataset "$DATASET" --trials "$TRIALS"
