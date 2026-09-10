#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
CFG="${CFG:-configs/kaggle_trust.yaml}"

export PYTHONPATH="$(pwd)/src${PYTHONPATH:+:${PYTHONPATH}}"
${PYTHON_BIN} -m pip install -q -e ".[dev]"

${PYTHON_BIN} -m safegrip.cli --config "$CFG" download --datasets lira
${PYTHON_BIN} -m safegrip.cli --config "$CFG" prepare --dataset lira
${PYTHON_BIN} -m safegrip.cli --config "$CFG" benchmark --dataset lira --preset trust --models todorovic2022_cnn,lampe2023_gru
${PYTHON_BIN} -m safegrip.cli --config "$CFG" statistics --results results/lira_trust --bootstrap 2000
${PYTHON_BIN} -m safegrip.cli --config "$CFG" ablation --dataset lira --preset trust --variants safegrip_backbone_raw,safegrip_persistent,safegrip_neural_innovation,safegrip_no_identifiability,safegrip_excitation_proxy,safegrip_no_acceptance,safegrip_no_cf_agreement,safegrip_no_innovation_supervision,safegrip_no_bound,safegrip_no_uq,safegrip

${PYTHON_BIN} - <<'PY2'
import json
from pathlib import Path
p=Path('results/lira_trust/result_health.json')
health=json.loads(p.read_text())
print('\n=== SCIENTIFIC HEALTH ===')
print(json.dumps(health, indent=2))
if health['status'] != 'PASS':
    print('\nRESULT STATUS = REVIEW. The run completed, but automatic scientific sanity gates did not all pass.')
else:
    print('\nRESULT STATUS = PASS. Automatic degeneracy/sanity gates passed; still report multi-seed uncertainty and limitations.')
if health['status'] != 'PASS':
    raise SystemExit(2)
PY2
