# SafeGrip v2 method (v0.6.0)

SafeGrip v2 replaces the earlier TCN + jointly trained Gaussian head + post-hoc projection design.

## Point estimator

Only the SafeGrip proposal receives label-free engineered dynamics features. Literature baselines keep their original common-sensor inputs.

The proposal uses:

1. a short temporal GRU branch;
2. a window-statistics MLP branch (last / mean / standard deviation);
3. a label-free excitation score built from acceleration utilization, relative wheel-speed spread, torque demand and jerk;
4. excitation-aware gated fusion;
5. an identified-set output parameterization

```text
mu_hat = lower + (mu_upper - lower) * sigmoid(z)
```

so the full proposal is feasible by construction and does not rely on a post-hoc clipping step.

The point estimator is optimized with Huber/SmoothL1 loss and selected by validation RMSE.

## Calibration roles

The calibration split has two disjoint contiguous roles:

- the first configured fraction (`calibration.lower_fraction`, default 0.25) calibrates only the conservative mechanics lower-endpoint relaxation;
- the remaining calibration endpoints are reserved for predictive conformal UQ.

Neither calibration subset participates in gradient training.

## Predictive uncertainty

After the point estimator is frozen, a separate residual-scale head is fitted on validation residuals. Its scale is enlarged under weak excitation:

```text
scale_eff = scale * (1 + excitation_beta * (1 - excitation))
```

The disjoint UQ calibration subset supplies standardized absolute residual scores. Because benchmark windows overlap, calibration uses one maximum score per non-overlapping block before taking the finite-sample conformal quantile. This is a conservative dependence-mitigation device; the repository does **not** claim arbitrary-dependence finite-sample conformal validity.

The reported interval is the conformal data interval intersected with the configured physical support/identified set.

## Default temporal context

The default proposal context is 16 samples at 10 Hz (about 1.6 s). The benchmark warm-up is determined by methods actually compared, not by unused candidates in the tuning search space.

## Controlled ablations

- `safegrip_data_only`: no sample-specific lower bound and no excitation gate.
- `safegrip_static_only`: no GRU temporal branch.
- `safegrip_no_gate`: fixed 50/50 static-temporal fusion.
- `safegrip_no_bound`: global `[0, mu_upper]` support only.
- `safegrip_no_uq`: full point estimator, no residual-scale/conformal interval.
- `safegrip_no_calibration`: raw mechanics lower endpoint without its statistical relaxation.
- `safegrip`: full v2 method.

The automatic health gate still refuses to call a run trustworthy when SafeGrip fails simple train-only sanity baselines, has negative R2, lacks enough test endpoints, or its predictive interval materially undercovers.
