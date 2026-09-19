#!/usr/bin/env bash
set -euo pipefail
PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHONPATH="$(pwd)/src${PYTHONPATH:+:${PYTHONPATH}}"
${PYTHON_BIN} -m pip install -q -e '.[paper]'
SG="${PYTHON_BIN} -m safegrip.cli"
TRIALS="${TRIALS:-30}"
BASELINE_TRIALS="${BASELINE_TRIALS:-$TRIALS}"
TUNE_EPOCHS="${TUNE_EPOCHS:-}"

$SG download --datasets lira
$SG prepare --dataset lira

HP_ARGS=()
BASELINE_HP_ARGS=()
if [[ "${SKIP_TUNING:-0}" != "1" ]]; then
  TUNE_EPOCH_ARGS=()
  if [[ -n "$TUNE_EPOCHS" ]]; then TUNE_EPOCH_ARGS=(--epochs "$TUNE_EPOCHS"); fi

  # FRC tunes only approximation/optimization parameters. Mathematical
  # certificate quantities remain fixed by configs/default.yaml.
  $SG tune --method frc --dataset lira --trials "$TRIALS" "${TUNE_EPOCH_ARGS[@]}"
  $SG tune-baselines --dataset lira --protocol controlled --trials "$BASELINE_TRIALS" "${TUNE_EPOCH_ARGS[@]}"
  HP_ARGS=(--proposal-hparams results/lira_frc_tuning/best_hparams.yaml)
  BASELINE_HP_ARGS=(--baseline-hparams results/lira_baseline_tuning/best_hparams.yaml)
fi

$SG benchmark --dataset lira --preset paper --proposal frc --protocol controlled \
  "${HP_ARGS[@]}" "${BASELINE_HP_ARGS[@]}"
$SG statistics --results results/lira_paper --proposal safegrip_frc --bootstrap 5000

if [[ "${RUN_SOURCE_FAITHFUL:-0}" == "1" ]]; then
  # Separate table: preserves configured source-style training budgets and is
  # deliberately not described as compute-matched.
  $SG benchmark --dataset lira --preset paper --proposal frc --protocol source-faithful \
    "${HP_ARGS[@]}" "${BASELINE_HP_ARGS[@]}"
fi

${PYTHON_BIN} scripts/check_paper_readiness.py
${PYTHON_BIN} scripts/package_paper_results.py
printf '\nSafeGrip-FRC paper run complete.\n'
printf '  main table:       results/lira_paper/metrics.csv\n'
printf '  theorem audit:    results/lira_paper/theorem_audit.json\n'
printf '  certificates:     results/lira_paper/frc_resolution_certificates.csv\n'
printf '  fixed-H ablation: results/lira_paper/fixed_horizon_ablation.csv\n'
printf '  FRC tuning:       results/lira_frc_tuning/\n'
printf '  baseline tuning:  results/lira_baseline_tuning/\n'
