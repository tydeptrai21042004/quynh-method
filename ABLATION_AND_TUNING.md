# SafeGrip v2 ablation and hyperparameter protocol

## Controlled ablation

All variants use the same data splits, endpoint identities, seed list and selected hyperparameters unless a removed component makes a parameter irrelevant.

| Variant | Temporal GRU | Excitation gate | sample-specific bound | post-hoc UQ | lower-bound relaxation |
|---|---:|---:|---:|---:|---:|
| `safegrip_data_only` | yes | no | no | yes | reported only |
| `safegrip_static_only` | no | no | yes | yes | yes |
| `safegrip_no_gate` | yes | no (fixed 50/50 fusion) | yes | yes | yes |
| `safegrip_no_bound` | yes | yes | no; global `[0,mu_upper]` only | yes | n/a |
| `safegrip_no_uq` | yes | yes | yes | no | yes |
| `safegrip_no_calibration` | yes | yes | raw mechanics lower endpoint | yes | no |
| `safegrip` | yes | yes | yes, by parameterization | yes | yes |

The full point estimator is

```text
mu_hat = lower + (mu_upper - lower) * sigmoid(z)
```

and is trained with Huber/SmoothL1 loss. There is no soft physics-loss weight and no post-hoc point projection in the full method.

Predictive UQ is fitted only after the point network is frozen. Validation residuals train a positive scale head; a disjoint calibration subset determines the block-max conformal multiplier.

## Hyperparameter search

The proposal search is selected only by validation RMSE. The test partition is locked during search.

| Parameter | Search |
|---|---|
| `sequence_length` | {8, 16, 24, 32, 64} |
| `hidden` | {32, 64, 96} |
| `gru_hidden` | {16, 32, 64} |
| `dropout` | 0–0.30 |
| `lr` | log-uniform 1e-4–3e-3 |
| `weight_decay` | log-uniform 1e-6–1e-3 |
| `batch_size` | {128, 256, 512} |
| `huber_beta` | {0.03, 0.05, 0.10} |
| `excitation_beta` | {0.5, 1.0, 2.0} |

The point hyperparameter search runs the `safegrip_no_uq` variant because UQ is post-hoc and must not influence selection of the point estimator. The final selected point configuration is then refit with the full UQ pipeline.

**Not tuned against validation score:** `mu_upper`, conformal `alpha`, LiRA matching tolerances, physical uncertainty margins, and the fixed label-free excitation-feature reference scales.

Literature comparators receive the same Optuna trial budget while preserving their recoverable source architecture/preprocessing constraints.
