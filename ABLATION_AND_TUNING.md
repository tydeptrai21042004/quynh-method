# SafeGrip ablation and hyperparameter protocol

## Controlled ablation

All variants use the same split, seed, feature list and selected hyperparameters unless a component being removed makes a parameter irrelevant.

| Variant | Temporal TCN | UQ head | soft physics loss | calibrated lower set | hard projection |
|---|---:|---:|---:|---:|---:|
| `safegrip_data_only` | yes | yes | no | reported only | no |
| `safegrip_no_projection` | yes | yes | yes | yes | no |
| `safegrip_no_uq` | yes | no | yes | yes | yes |
| `safegrip_no_physics_loss` | yes | yes | no | yes | yes |
| `safegrip_no_calibration` | yes | yes | yes | raw mechanics bound | yes |
| `safegrip_no_temporal` | no (last-state MLP) | yes | yes | yes | yes |
| `safegrip` | yes | yes | yes | yes | yes |

This design isolates:

- the complete data-only backbone (`safegrip_data_only`);
- soft-vs-hard physics contributions;
- uncertainty modeling;
- external-reference calibration;
- temporal modeling.

Run:

```bash
safegrip ablation --dataset lira --preset paper
```

After tuning, reuse exactly the same selected proposal hyperparameters:

```bash
safegrip ablation --dataset lira --preset paper \
  --proposal-hparams results/lira_tuning/best_hparams.yaml
```

## Hyperparameter search

The main benchmark uses **validation RMSE for every method** so model selection is aligned with the primary point-estimation comparison. Test data are never queried by either proposal or baseline tuning.

SafeGrip search space (`configs/default.yaml`) includes:

| Parameter | Search |
|---|---|
| `sequence_length` | {16, 32, 64, 100, 128} |
| `hidden` | {32, 64, 96, 128} |
| `tcn_blocks` | integer 2–5 |
| `kernel_size` | {2, 3, 5} |
| `dropout` | 0–0.30 |
| `lr` | log-uniform 1e-4–3e-3 |
| `weight_decay` | log-uniform 1e-6–1e-3 |
| `batch_size` | {128, 256, 512} |
| `lambda_mse` | log-uniform 0.05–0.50 |
| `lambda_physics` | log-uniform 1e-3–0.30 |

Literature baselines receive the **same number of Optuna trials**. Recoverable source architecture/preprocessing remains fixed; model-specific LiRA adaptation parameters are selected on validation. For the explicitly adapted Transformer and SV-DKL comparators, unknown architecture details are also validation-selected rather than presented as source parameters.

```bash
safegrip tune --dataset lira --trials 30 --no-test
safegrip tune-baselines --dataset lira --trials 30
```

### Intentionally fixed

`mu_upper` and `alpha` are **not tuned**. They define physical/safety/calibration assumptions; tuning them to validation error would weaken their interpretation.

## No-test-leakage sequence

```text
train + calibration + validation
        -> proposal tuning + equal-budget baseline tuning
        -> freeze one configuration per method
        -> five independent final seeds
        -> common held-out test endpoints
        -> report mean +/- std
```

Methods may use different temporal context lengths. `benchmark.common_warmup_samples` forces every validation/test prediction to correspond to the same endpoint set, so history length does not silently change the evaluation population.

## Outputs

```text
results/lira_tuning/
  optuna.sqlite3
  trials.csv
  best_hparams.yaml
  param_importance.json
  tuning_summary.json

results/lira_baseline_tuning/
  best_hparams.yaml
  tuning_summary.json
  <baseline-id>/trials.csv
  <baseline-id>/best_hparams.yaml
  <baseline-id>/optuna.sqlite3
```

Both tuning summaries record that test data are not used during search.
