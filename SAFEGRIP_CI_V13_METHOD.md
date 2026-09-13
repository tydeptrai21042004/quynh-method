# SafeGrip-CI v1.3 — Counterfactual Energy-Guided Selective Physics Correction

## 1. Scope

SafeGrip-CI v1.3 is a hybrid road-friction estimator. A causal Conv1D+GRU temporal network remains responsible for the primary friction prediction. Physics is not treated as a second oracle. Instead, a separately trained friction-conditioned dynamics model evaluates whether nearby friction hypotheses are consistent with the observed dynamics, and a learned selector applies a correction only when that evidence is useful.

The redesign directly addresses the v1.2 LiRA evidence: the single Gauss--Newton correction was often harmful, the utility gate was nearly constant, and Jacobian-based identifiability had little measurable effect.

## 2. Temporal base estimator

For a causal sensor window X_t, the temporal encoder produces h_t and a direct estimate

    mu_B = F_theta(X_t).

The base estimate is supervised explicitly even in the full model. This prevents the hybrid path from hiding a weak temporal estimator behind projection or correction.

## 3. Counterfactual friction-energy landscape

Let G_phi(c_t, mu) be a separately trained friction-conditioned dynamics predictor. Around the detached base estimate, v1.3 constructs K odd, symmetric local friction hypotheses

    mu_k = clip(mu_B + delta_k, 0, mu_max),

with delta_k spanning [-r, r]. Each hypothesis receives an observed-dynamics energy

    E_k = mean((G_phi(c_t, mu_k) - y_dyn,t)^2).

Per-sample energy centering/scaling makes the posterior robust to arbitrary sensor scale. The energy posterior is

    w_k = softmax(-(E_k - min(E)) / (T * scale(E))).

The physics candidate is the posterior mean

    mu_P = sum_k w_k mu_k,

and the raw correction is

    Delta_mu = clip(mu_P - mu_B, -Delta_max, Delta_max).

This replaces the v1.2 finite-difference Jacobian and one-step Gauss--Newton update.

## 4. Entropy identifiability and physics evidence

The normalized posterior entropy is

    H_n = -sum_k w_k log(w_k) / log(K),

and local identifiability is

    I = clip(1 - H_n, 0, 1).

A flat energy landscape therefore produces low identifiability automatically. The model also exports:

- normalized energy improvement from mu_B to mu_P;
- local energy curvature around the center hypothesis;
- base/candidate dynamics residuals;
- signed and absolute candidate correction;
- aleatoric scale and optional regime-change evidence.

All physics evidence entering the point-estimation selector is detached. Point-estimation gradients cannot make the dynamics model appear artificially informative.

## 5. Two-stage selective correction

v1.2 used one scalar gate for both "should physics help?" and "how much should be applied?". v1.3 separates them:

    p_help = H_help(h_t, evidence),
    alpha = H_step(h_t, evidence),
    g = p_help * alpha.

The corrected raw estimate is

    mu_raw = mu_B + g * physics_correction_scale * Delta_mu.

With the utility gate ablated, the whole physics correction is applied. With the magnitude head ablated, alpha=1 and only help probability controls selection.

## 6. Supervision of the selector

On training labels only, the benefit target is

    help* = 1[ |mu_P - mu_true| + m_help < |mu_B - mu_true| ].

The benefit head uses binary cross entropy. The magnitude target is the clipped least-squares fraction of the candidate step that would move the base toward the target,

    alpha* = clip(((mu_true-mu_B) Delta_mu) / (Delta_mu^2 + eps), 0, 1),

and is trained only where the candidate is useful and the correction is non-degenerate. These oracle targets never enter inference.

## 7. Friction-discriminative dynamics training

The dynamics model is pretrained before selector training. In addition to endpoint reconstruction, v1.3 uses a multi-negative counterfactual objective. True friction is the positive hypothesis and several nearby symmetric alternatives are negatives. If E(mu) is dynamics error, the contrastive objective is equivalent to cross entropy over -E/T and forces G_phi to use friction rather than explaining the endpoint solely through temporal continuity.

The default full model freezes dynamics after this pretraining stage. This stabilizes selector evidence and prevents point-loss leakage into physics evidence.

## 8. Accuracy and safety objectives

The full point path includes:

- Huber loss on the final estimate;
- direct Huber supervision on mu_B;
- benefit-head loss;
- useful-sample correction-fraction loss;
- a smooth unsafe-overestimation loss aligned with the reported threshold mu_hat - mu_true > 0.05;
- a do-no-harm penalty when the corrected prediction is worse than the base by more than a small margin;
- optional heteroscedastic residual training.

The previous change/regime/smoothness losses default to zero in v1.3 because the v1.2 controlled ablations did not show that they earned their complexity.

## 9. Training stages

1. **Temporal pretraining**: train the base estimator.
2. **Dynamics pretraining**: train reconstruction plus counterfactual friction discrimination.
3. **Selective-correction training**: train the point estimator and two selector heads; dynamics is frozen by default.
4. **UQ calibration**: fit the residual-scale head and block-max split-conformal factor using the disjoint UQ calibration role.

## 10. Physical support and UQ

When enabled, the final point estimate is projected to the global/conditional feasible interval. Projection is a safety constraint and is evaluated with a no-bound ablation; it is not automatically credited as an accuracy contribution.

Predictive UQ remains a separate calibration layer. `safegrip_no_uq` has the same point-estimation path as `safegrip`.

## 11. Primary v1.3 ablations

The controlled set is:

- `safegrip_base_temporal`
- `safegrip_no_safety_loss`
- `safegrip_no_physics_residual`
- `safegrip_no_utility_gate`
- `safegrip_no_identifiability`
- `safegrip_no_magnitude_head`
- `safegrip_no_energy_improvement`
- `safegrip_no_contrastive_dynamics`
- `safegrip_no_do_no_harm`
- `safegrip_no_bound`
- `safegrip_no_heteroscedastic`
- `safegrip_no_uq`
- `safegrip`

The paper should treat the matched temporal base and matched GRU comparisons as central. v1.3 is supported only if the selective correction improves the measured accuracy/safety trade-off on locked real-data evaluation; implementation novelty alone is not an empirical superiority result.
