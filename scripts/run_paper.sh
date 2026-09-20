#!/usr/bin/env bash
set -euo pipefail
PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHONPATH="$(pwd)/src${PYTHONPATH:+:${PYTHONPATH}}"
${PYTHON_BIN} -m pip install -q -e '.[paper,dev]'
SG="${PYTHON_BIN} -m safegrip.cli"
TRIALS="${TRIALS:-20}"
BASELINE_TRIALS="${BASELINE_TRIALS:-30}"
TUNE_EPOCHS="${TUNE_EPOCHS:-}"

$SG download --datasets lira
$SG prepare --dataset lira

HP_ARGS=()
BASELINE_HP_ARGS=()
if [[ "${SKIP_TUNING:-0}" != "1" ]]; then
  TUNE_EPOCH_ARGS=()
  if [[ -n "$TUNE_EPOCHS" ]]; then TUNE_EPOCH_ARGS=(--epochs "$TUNE_EPOCHS"); fi

  # Only neural approximation/optimization parameters are tuned.  The physical
  # interval, alpha, conformal rule, and projection theorem are fixed.
  $SG tune --method pfr --dataset lira --trials "$TRIALS" "${TUNE_EPOCH_ARGS[@]}"
  $SG tune-baselines --dataset lira --protocol controlled --trials "$BASELINE_TRIALS" "${TUNE_EPOCH_ARGS[@]}"
  HP_ARGS=(--proposal-hparams results/lira_pfr_tuning/best_hparams.yaml)
  BASELINE_HP_ARGS=(--baseline-hparams results/lira_baseline_tuning/best_hparams.yaml)
fi

$SG benchmark --dataset lira --preset paper --proposal pfr --protocol controlled \
  "${HP_ARGS[@]}" "${BASELINE_HP_ARGS[@]}"
$SG statistics --results results/lira_paper --proposal safegrip_pfr --bootstrap 5000

if [[ "${RUN_SOURCE_FAITHFUL:-0}" == "1" ]]; then
  $SG benchmark --dataset lira --preset paper --proposal pfr --protocol source-faithful \
    "${HP_ARGS[@]}" "${BASELINE_HP_ARGS[@]}"
fi

${PYTHON_BIN} scripts/check_paper_readiness.py
${PYTHON_BIN} scripts/package_paper_results.py
printf '\nSafeGrip-PFR paper run complete.\n'
printf '  main table:       results/lira_paper/metrics.csv\n'
printf '  component table:  results/lira_paper/pfr_component_ablation.csv\n'
printf '  theorem audit:    results/lira_paper/pfr_theorem_audit.json\n'
printf '  PFR tuning:       results/lira_pfr_tuning/\n'
printf '  baseline tuning:  results/lira_baseline_tuning/\n'
