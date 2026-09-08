#!/usr/bin/env bash
set -euo pipefail
PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHONPATH="$(pwd)/src${PYTHONPATH:+:${PYTHONPATH}}"
${PYTHON_BIN} -m pip install -q -e '.[paper]'
SG="${PYTHON_BIN} -m safegrip.cli"
TRIALS="${TRIALS:-30}"
BASELINE_TRIALS="${BASELINE_TRIALS:-$TRIALS}"
TUNE_EPOCHS="${TUNE_EPOCHS:-}"

# Primary reproducible benchmark.
$SG download --datasets lira
$SG prepare --dataset lira

HP_ARGS=()
BASELINE_HP_ARGS=()
if [[ "${SKIP_TUNING:-0}" != "1" ]]; then
  TUNE_EPOCH_ARGS=()
  if [[ -n "$TUNE_EPOCHS" ]]; then TUNE_EPOCH_ARGS=(--epochs "$TUNE_EPOCHS"); fi

  # Proposal and every literature comparator receive validation-only tuning.
  # Test remains untouched until the final benchmark.
  $SG tune --dataset lira --trials "$TRIALS" "${TUNE_EPOCH_ARGS[@]}" --no-test
  $SG tune-baselines --dataset lira --trials "$BASELINE_TRIALS" "${TUNE_EPOCH_ARGS[@]}"
  HP_ARGS=(--proposal-hparams results/lira_tuning/best_hparams.yaml)
  BASELINE_HP_ARGS=(--baseline-hparams results/lira_baseline_tuning/best_hparams.yaml)
fi

# Paper preset uses five independent seeds by default and evaluates every method
# on the same validation/test endpoints even when temporal context differs.
$SG benchmark --dataset lira --preset paper "${HP_ARGS[@]}" "${BASELINE_HP_ARGS[@]}"
# Reuse exactly the selected proposal hyperparameters for every ablation.
$SG ablation --dataset lira --preset paper "${HP_ARGS[@]}"

# Low-cost reviewer analyses reuse the frozen final predictions.
$SG experiment --dataset lira --study excitation --results results/lira_paper
$SG experiment --dataset lira --study robustness --results results/lira_paper

# Training-heavy scarcity/cross-route experiments are kept opt-in.
if [[ "${RUN_EXTENDED:-0}" == "1" ]]; then
  bash scripts/run_extended_experiments.sh
fi

if [[ "${DOWNLOAD_AUX:-0}" == "1" ]]; then
  bash scripts/download_all_datasets.sh
fi

${PYTHON_BIN} scripts/check_paper_readiness.py
${PYTHON_BIN} scripts/package_paper_results.py
printf '\nPaper run complete and readiness gate passed.\n'
printf '  main table:       results/lira_paper/metrics.csv\n'
printf '  per-seed table:   results/lira_paper/metrics_by_seed.csv\n'
printf '  parity controls:  results/lira_paper/projection_control_metrics.csv\n'
printf '  baseline tuning:  results/lira_baseline_tuning/\n'
printf '  proposal tuning:  results/lira_tuning/\n'
printf '  ablation:         results/lira_ablation_paper/ablation_metrics.csv\n'
printf '  excitation:       results/lira_excitation/excitation_metrics.csv\n'
printf '  robustness:       results/lira_robustness/physics_robustness_metrics.csv\n'
printf '  extended:         RUN_EXTENDED=1 bash scripts/run_paper.sh\n' 
