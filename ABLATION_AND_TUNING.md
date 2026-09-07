# SafeGrip ablation and hyperparameter protocol

## Controlled ablation

All variants use the same split, feature list, selected proposal hyperparameters and final seed list unless the removed component makes a parameter irrelevant.

| Variant | Temporal TCN | UQ head | soft physics loss | calibrated lower set | hard projection |
|---|---:|---:|---:|---:|---:|
| `safegrip_data_only` | yes | yes | no | reported only | no |
| `safegrip_no_projection` | yes | yes | yes | yes | no |
| `safegrip_no_uq` | yes | no | yes | yes | yes |
| `safegrip_no_physics_loss` | yes | yes | no | yes | yes |
| `safegrip_no_calibration` | yes | yes | yes | raw mechanics bound | yes |
| `safegrip_no_temporal` | no (last-state MLP) | yes | yes | yes | yes |
| `safegrip` | yes | yes | yes | yes | yes |

The deterministic `safegrip_no_uq` objective is

`MSE + lambda_physics * L_physics`,

not `(1 + lambda_mse) * MSE`. This keeps removal of the heteroscedastic head controlled.

Run:

```bash
safegrip ablation --dataset lira --preset paper
```

After tuning, reuse exactly the same selected proposal hyperparameters:

```bash
safegrip ablation --dataset lira --preset paper \
  --proposal-hparams results/lira_tuning/best_hparams.yaml
```

Paper mode uses the same five seeds as the main table and exports:

```text
ablation_metrics.csv          # mean/std
ablation_metrics_by_seed.csv  # individual runs
ablation_predictions.csv
ablation_design.json
```

## Hyperparameter search

The main benchmark uses **validation RMSE for every method**. Test data are never queried by proposal or baseline tuning.

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

Literature baselines receive the **same number of Optuna trials**. Recoverable source architecture/preprocessing remains fixed; unknown details of explicitly adapted methods are selected on validation and disclosed as adaptations.

```bash
safegrip tune --dataset lira --trials 30 --no-test
safegrip tune-baselines --dataset lira --trials 30
```

### Intentionally fixed protocol assumptions

The following are not optimized against validation RMSE:

- `mu_upper`;
- conformal `alpha`;
- LiRA GPS match tolerance;
- heading tolerance;
- physical uncertainty margins.

Sensitivity to important physical assumptions is evaluated separately by the robustness experiment.
