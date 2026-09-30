#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHONPATH="$(pwd)/src${PYTHONPATH:+:${PYTHONPATH}}"
${PYTHON_BIN} -m pip install -q -e '.[paper,dev]'
SG="${PYTHON_BIN} -m safegrip.cli"

DATASET="lira_cd"
MODELS="du2023_inceptiontime,todorovic2022_cnn,lampe2023_gru,levenberg2023_stft"
MAIN="results/${DATASET}_paper"

# Re-prepare intentionally: the current paper protocol uses trajectory-level
# group holdout, so stale prepared files from the older within-trajectory split
# must not be silently reused.
$SG download --datasets "$DATASET"
$SG prepare --dataset "$DATASET"
$SG benchmark --dataset "$DATASET" --preset paper --models "$MODELS" --protocol controlled
$SG statistics --results "$MAIN" --proposal safegrip_pfr --bootstrap 5000

# Only Du and Lampe expose sufficiently specified source settings in this
# repository. The full four-model paper comparison remains the controlled run.
if [[ "${RUN_SOURCE_FAITHFUL:-0}" == "1" ]]; then
  $SG benchmark --dataset "$DATASET" --preset paper \
    --models du2023_inceptiontime,lampe2023_gru --protocol source-faithful
fi

${PYTHON_BIN} scripts/check_paper_readiness.py
${PYTHON_BIN} scripts/package_paper_results.py
printf '\nSafeGrip-PFR-ECR paper run complete.\n'
printf '  main table:       %s/metrics.csv\n' "$MAIN"
printf '  component table:  %s/pfr_component_ablation.csv\n' "$MAIN"
printf '  safety table:     %s/pfr_safety_metrics.csv\n' "$MAIN"
printf '  theorem audit:    %s/pfr_theorem_audit.json\n' "$MAIN"
