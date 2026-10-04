#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHONPATH="$(pwd)/src${PYTHONPATH:+:${PYTHONPATH}}"
${PYTHON_BIN} -m pip install -q -e '.[paper,dev]'
${PYTHON_BIN} -m pytest -q

# The paper evidence is the frozen NPI-v3 all-real-data repeated-seed protocol.
SAFEGRIP_SEEDS="${SAFEGRIP_SEEDS:-3101,3102,3103}" \
SAFEGRIP_EPOCHS="${SAFEGRIP_EPOCHS:-10}" \
${PYTHON_BIN} scripts/KAGGLE_ALL_REAL_DATASETS_MULTI_SEED.py
