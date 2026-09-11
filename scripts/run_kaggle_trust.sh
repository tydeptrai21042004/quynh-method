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
${PYTHON_BIN} -m safegrip.cli --config "$CFG" ablation --dataset lira --preset trust --variants safegrip_base_temporal,safegrip_no_dynamic_loss,safegrip_no_safety_loss,safegrip_no_physics_residual,safegrip_no_utility_gate,safegrip_no_identifiability,safegrip_no_bound,safegrip_no_regime_head,safegrip_no_heteroscedastic,safegrip_no_uq,safegrip

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
