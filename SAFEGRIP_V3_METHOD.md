# SafeGrip v3 method (v0.7.0)

SafeGrip v3 is an excitation-aware, physics-constrained estimator for road-friction reference under limited vehicle excitation. It replaces the v2 hidden-feature fusion gate with an explicit **slow prior + short dynamic-evidence update**.

## 1. Label-free excitation / observability proxy

The proposal-only feature layer derives acceleration utilization, jerk, relative wheel-speed spread, pressure spread and torque utilization without using `mu_ref`. Their bounded instantaneous composite is `sg_excitation_instant`.

Because an informative maneuver remains useful for a short time after its peak, v3 uses a causal peak-memory score

```text
E_t = max(E_inst,t, decay * E_{t-1})
```

with reset at every `(segment_id, split)` boundary. `sg_excitation_score` therefore never looks forward and never crosses train/calibration/validation/test or trajectory boundaries. The score is preserved in physical `[0,1]` scale after all other features are standardized.

## 2. Slow prior

For the complete context window, SafeGrip forms summary statistics and a GRU representation. They are fused into `h_prior`, from which the network predicts an identified-set logit `q_prior`.

With lower endpoint `L_t` and global upper support `U`, the prior prediction is

```text
mu_prior = L_t + (U-L_t) * sigmoid(q_prior).
```

The prior is intended to represent persistent road-friction state rather than a one-sample maneuver response.

## 3. Short dynamic evidence

The final `evidence_window` samples are processed by a separate statistics encoder and GRU. This branch predicts a bounded correction

```text
delta_t = delta_scale * tanh(f_evidence(h_evidence)).
```

The evidence branch does not directly predict absolute friction.

## 4. Monotone excitation reliability

Dynamic evidence enters through

```text
r(E) = sigmoid(softplus(a) * (E - sigmoid(tau))).
```

Because `softplus(a) > 0`, reliability is monotone non-decreasing in excitation for every learned parameter value:

```text
dr(E)/dE >= 0.
```

The final identified-set logit is

```text
q_t = q_prior + r(E_t) * delta_t
```

and the point estimate is

```text
mu_hat = L_t + (U-L_t) * sigmoid(q_t).
```

Thus `L_t <= mu_hat <= U` by construction. Under weak excitation, the update shrinks toward the persistent prior; stronger excitation can admit a larger dynamic correction.

## 5. Point-training objective

The primary loss remains Huber/SmoothL1 on absolute friction. v3 adds same-segment temporal regularizers using only training labels:

```text
L = L_point
  + lambda_delta * L_change
  + lambda_rank  * L_rank
  + lambda_smooth * L_weak-excitation-smooth.
```

`L_change` compares predicted and true friction changes between paired endpoints from the same trajectory segment. `L_rank` penalizes incorrect ordering when the true change exceeds `rank_min_delta`. The smoothness term is weighted by `(1-E_t)`, so the model is discouraged from making unsupported jumps only when excitation is weak.

Pairs are constructed from stable endpoint IDs and never cross a segment or split boundary.

## 6. Physics lower endpoint

The vehicle-level mechanics layer now uses a vector force balance. Under the configured level-road/forward-motion approximation,

```text
F_x,tire ~= m*a_x + F_drag + F_rr
F_y,tire ~= m*a_y.
```

After subtracting bounded acceleration-vector and unmodelled-force uncertainty, the norm provides a conditional lower bound on utilized friction. The result is not hard-clipped inside the physics function; preprocessing audits values against `mu_upper` so unit/schema failures remain visible.

The paper claim remains conditional on the configured uncertainty margins and `mu* <= mu_upper`.

## 7. Uncertainty quantification

Point-model selection is completed before UQ fitting. A small positive residual-scale head is trained on validation residual magnitude. Its final bias is initialized from the median validation residual scale, avoiding the v2 `softplus(0) ~= 0.69` mismatch.

Weak excitation monotonically inflates uncertainty:

```text
s_t = s_base,t * (1 + beta * (1-E_t)),   beta >= 0.
```

A disjoint calibration subset computes a block-max split-conformal multiplier. The raw predictive interval is then intersected with the physical support. Block maxima mitigate overlap dependence but are not claimed to provide arbitrary-dependence conformal validity.

## 8. Auditable outputs

`predictions.csv` includes, in addition to the final estimate:

- `safegrip_prior`;
- `safegrip_evidence_delta`;
- `safegrip_reliability`;
- `safegrip_excitation`;
- final/prior latent coordinates;
- post-hoc UQ scale and calibrated intervals when enabled.

These fields make it possible to verify whether the dynamic update is active, whether reliability follows excitation, and whether improvement comes from the prior or evidence branch.
