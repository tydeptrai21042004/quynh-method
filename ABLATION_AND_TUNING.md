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

SafeGrip uses Optuna TPE. The objective is validation Gaussian NLL, so both point prediction and predicted scale are evaluated. The held-out test partition is not queried by the objective.

Default search space (`configs/default.yaml`):

| Parameter | Search |
|---|---|
| `sequence_length` | {16, 32, 64, 128} |
| `hidden` | {32, 64, 96, 128} |
| `tcn_blocks` | integer 2–5 |
| `kernel_size` | {2, 3, 5} |
| `dropout` | 0–0.30 |
| `lr` | log-uniform 1e-4–3e-3 |
| `weight_decay` | log-uniform 1e-6–1e-3 |
| `batch_size` | {128, 256, 512} |
| `lambda_mse` | log-uniform 0.05–0.50 |
| `lambda_physics` | log-uniform 1e-3–0.30 |

### Intentionally fixed

`mu_upper` and `alpha` are **not tuned**. They define physical/safety/calibration assumptions; tuning them to validation error would weaken the interpretation of the guarantee.

## No-test-leakage sequence

Paper script:

```text
train + calibration + validation
        -> Optuna selection (--no-test)
        -> best_hparams.yaml
        -> freeze configuration
        -> common final baseline/proposal test
        -> ablation with same selected hyperparameters
```

The selected sequence length is applied as the common evidence-window length for the direct benchmark so all methods are evaluated on the same test endpoints.

## Outputs

```text
results/lira_tuning/
  optuna.sqlite3
  trials.csv
  best_hparams.yaml
  param_importance.json
  tuning_summary.json
```

`tuning_summary.json` explicitly records `test_used_during_search: false` and the fixed safety parameters.
