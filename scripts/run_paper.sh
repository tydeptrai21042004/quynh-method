#!/usr/bin/env bash
set -euo pipefail
python -m pip install -q -e '.[paper]'
TRIALS="${TRIALS:-30}"
EPOCHS="${TUNE_EPOCHS:-40}"

# Primary reproducible benchmark.
safegrip download --datasets lira
safegrip prepare --dataset lira

# Tune proposal on train/calibration/validation only. Test stays untouched here.
if [[ "${SKIP_TUNING:-0}" != "1" ]]; then
  safegrip tune --dataset lira --trials "$TRIALS" --epochs "$EPOCHS" --no-test
  HP_ARGS=(--proposal-hparams results/lira_tuning/best_hparams.yaml)
else
  HP_ARGS=()
fi

# Every direct baseline below is backed by a published tire/friction paper.
safegrip benchmark --dataset lira --preset paper "${HP_ARGS[@]}"
# Reuse exactly the selected proposal hyperparameters for every ablation.
safegrip ablation --dataset lira --preset paper "${HP_ARGS[@]}"

# Optional auxiliary open-data download/normalization. These datasets have
# different targets and are deliberately NOT mixed into the direct LiRA table.
if [[ "${DOWNLOAD_AUX:-0}" == "1" ]]; then
  bash scripts/download_all_datasets.sh
fi

printf '\nPaper run complete.\n'
printf '  main table: results/lira_paper/metrics.csv\n'
printf '  ablation:   results/lira_ablation_paper/ablation_metrics.csv\n'
printf '  tuning:     results/lira_tuning/\n'
