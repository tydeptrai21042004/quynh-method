# Semantic physical-reference validation

This note records **diagnostic checks**, not final multi-seed paper results.
The purpose is to verify that the minimal proposal revision changes a real
failure mode without adding dataset-specific parameters.

## Unit/regression tests

Repository test result after the revision:

```text
164 passed, 14 warnings
```

The tests cover:

- exact state-reference semantics and token-order invariance;
- horizontal specific-force friction reference;
- integrated-speed displacement reference;
- documented UC3M U6ICRX five-column workbook layout and real slip condition;
- documented IO-VNBD 29-column synchronized vehicle layout;
- D2 paper-baseline output projection to the real public slip-angle target;
- Git-LFS pointer rejection for the IO-VNBD synchronized archive.

## LiRA real-checkpoint diagnostic

Using the previously saved real LiRA test records (467 windows) and the **same
frozen decoder weights**, the old prediction was nearly constant:

| output coordinate | MAE | RMSE | R2 | prediction std |
|---|---:|---:|---:|---:|
| old decoder / zero physical reference | 0.03162 | 0.04104 | -0.5621 | 0.0000009 |
| semantic reference + same frozen decoder | 0.02421 | 0.03194 | 0.0538 | 0.01130 |

The physical reference in this diagnostic is the dimensionless horizontal
specific-force demand `sqrt(ax_rms^2 + ay_rms^2) / g`.  It is **not** asserted
to equal road friction and is not used as a deterministic lower bound.

This diagnostic shows that the reference operator restores physically varying
signal to the collapsed LiRA output without changing network parameters.  It
does **not** replace the required retrained repeated-seed experiment: the final
claim must come from the new multi-seed runner.

## No dataset-name branch in the proposal

The proposal still uses one equation for every query:

```text
y_hat(q) = P_q(X) + R_theta(X, q)
```

`P_q` is resolved through physical target semantics (`state_component`,
`road_friction`, `localization`) and canonical sensor metadata. Dataset names
are not used by the reference operator.
