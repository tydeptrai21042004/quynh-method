# SafeGrip v3 ablation and hyperparameter protocol

## Controlled ablation

All variants use the same data splits, endpoint identities, seed list and selected hyperparameters unless a removed component makes a parameter irrelevant.

| Variant | Temporal prior/evidence | Excitation reliability | sample-specific bound | post-hoc UQ | lower-bound relaxation |
|---|---:|---:|---:|---:|---:|
| `safegrip_data_only` | yes | fixed 0.5 | no | yes | reported only |
| `safegrip_static_only` | no | inactive | yes | yes | yes |
| `safegrip_no_gate` | yes | fixed 0.5 | yes | yes | yes |
| `safegrip_no_bound` | yes | monotone | no; global `[0,U]` only | yes | n/a |
| `safegrip_no_uq` | yes | monotone | yes | no | yes |
| `safegrip_no_calibration` | yes | monotone | raw mechanics lower endpoint | yes | no |
| `safegrip` | yes | monotone | yes, by parameterization | yes | yes |

The full point estimator is

```text
q = q_prior + reliability(E) * evidence_delta
mu_hat = lower + (mu_upper-lower) * sigmoid(q).
```

There is no soft physics penalty and no post-hoc point projection in the full method.

## Point objective

The primary objective is Huber/SmoothL1. Three low-weight temporal regularizers are available:

- relative-change loss between same-segment endpoint pairs;
- ranking loss for changes larger than `rank_min_delta`;
- weak-excitation smoothness weighted by `(1-E)`.

The test partition and calibration labels never enter point-model gradient training.

## UQ

Predictive UQ is fitted only after point-model selection. Validation residuals train a positive scale head initialized near the observed residual scale. A disjoint calibration subset determines the block-max conformal multiplier. Excitation inflation is monotone because `excitation_beta >= 0`.

## Hyperparameter search

The proposal search is selected only by validation RMSE. The test partition is locked during search.

| Parameter | Search/default |
|---|---|
| `sequence_length` | {8, 16, 24, 32, 64} |
| `hidden` | {32, 64, 96} |
| `gru_hidden` | {16, 32, 64} |
| `dropout` | 0–0.30 |
| `lr` | log-uniform 1e-4–3e-3 |
| `weight_decay` | log-uniform 1e-6–1e-3 |
| `batch_size` | {128, 256, 512} |
| `huber_beta` | {0.03, 0.05, 0.10} |
| `evidence_window` | {4, 8, 12} |
| `delta_scale` | {1, 2, 3} |
| `delta_loss_weight` | {0.05, 0.15, 0.35} |
| `rank_loss_weight` | {0, 0.03, 0.05} |
| `excitation_beta` | {0.5, 1, 2} |

Structural defaults not expanded in the normal 30-trial search include monotone-gate initialization, pair lag, weak-excitation smoothness weight and physical/calibration assumptions.

**Not tuned against validation score:** `mu_upper`, conformal `alpha`, LiRA matching tolerances, physical uncertainty margins and fixed label-free excitation reference scales.

Literature comparators receive the same Optuna trial budget while preserving their recoverable source architecture/preprocessing constraints.
