# SafeGrip-CI method (v0.9.0)

## Core contribution

SafeGrip-CI v0.9 keeps the strongest part of v0.8 — persistent friction state plus counterfactual identifiability — but changes how candidate innovations are trained and authorized.

The estimator follows one principle:

> learn a friction correction independently, estimate whether friction is locally observable from the current dynamics, and only veto the correction when counterfactual dynamics provide affirmative evidence that the candidate is harmful.

The full point estimator uses raw vehicle-sensor channels. Handcrafted excitation remains only as a controlled comparator.

## 1. Persistent bounded friction state

At a segment start, a causal context encoder predicts a bounded prior. At later endpoints the previous estimate is blended with the context prior,

\[
\mu_t^- = \rho\hat\mu_{t-1} + (1-\rho)\mu_t^{ctx}.
\]

For mechanics lower endpoint \(L_t\) and global upper support \(U\),

\[
q_t^- = \operatorname{logit}\left(\frac{\mu_t^- - L_t}{U-L_t}\right).
\]

The persistent state is reset at every trajectory-segment boundary.

## 2. Independently supervised neural innovation

A short causal raw-sensor encoder predicts

\[
\nu_t = \Delta_{max}\tanh f_\theta(X_{t-H:t}).
\]

The candidate is

\[
q_t^{cand}=q_t^-+\nu_t,\qquad
\mu_t^{cand}=L_t+(U-L_t)\sigma(q_t^{cand}).
\]

Unlike v0.8, the candidate is supervised *before* authority is applied. If

\[
q_t^\star=\operatorname{logit}\left(\frac{\mu_t^\star-L_t}{U-L_t}\right),
\]

the direct innovation target is

\[
\nu_t^\star=q_t^\star-q_t^-.
\]

This separates candidate quality from update authority; a low authority can no longer hide a poor candidate during innovation training.

## 3. Friction-conditioned counterfactual dynamics

A causal dynamics model predicts selected standardized endpoint responses from the window prefix and a friction hypothesis,

\[
\hat y_t^{dyn}=G_\phi(X_{t-H:t-1},\mu).
\]

The endpoint response is not supplied to the dynamics encoder, so the residual comparison cannot be solved by copying the observed endpoint.

## 4. Counterfactual identifiability

Nearby friction hypotheses are evaluated around the persistent prior,

\[
J_t \approx
\frac{G_\phi(X,\mu_t^-+\delta)-G_\phi(X,\mu_t^- -\delta)}{2\delta}.
\]

The local information score is

\[
I_t^{raw}=\frac{1}{d_y}\|J_t\|_2^2,
\qquad
I_t=\frac{I_t^{raw}}{I_t^{raw}+\lambda_I}.
\]

This is a model-based local observability signal: it asks whether nearby friction hypotheses produce distinguishable vehicle responses under the current maneuver.

## 5. Normalized asymmetric counterfactual veto

Let

\[
R_t^- = \|y_t^{dyn}-G_\phi(X,\mu_t^-)\|_2^2,
\qquad
R_t^{cand} = \|y_t^{dyn}-G_\phi(X,\mu_t^{cand})\|_2^2.
\]

v0.8 used a symmetric sigmoid of the raw residual difference. On LiRA that residual difference was extremely small, so the factor collapsed near 0.5 and unnecessarily halved almost every innovation.

v0.9 first forms the scale-free score

\[
z_t=\frac{R_t^- - R_t^{cand}}
{R_t^- + R_t^{cand}+\varepsilon},\qquad -1\le z_t\le1.
\]

The veto probability is

\[
p_t^{veto}=\sigma\{-\gamma(z_t+\tau)\},
\]

and the trust multiplier is

\[
V_t=1-\rho_v p_t^{veto}.
\]

Therefore neutral evidence leaves \(V_t\) close to one; only a candidate that is materially worse under the counterfactual dynamics model loses substantial authority. This is intentionally asymmetric.

## 6. Local inverse-dynamics agreement

The same finite-difference sensitivity gives a damped local inverse-dynamics step. Let

\[
e_t=y_t^{dyn}-G_\phi(X,\mu_t^-).
\]

Then

\[
\Delta\mu_t^{cf}
=\operatorname{clip}\left(
\frac{J_t^\top e_t}{J_t^\top J_t+\lambda_{GN}},
-r_{max},r_{max}
\right).
\]

This quantity uses observed dynamics and the learned dynamics model, not the friction label. During training it is detached and used as a weak auxiliary target for the learned candidate correction. The model also exports a candidate/counterfactual agreement score for diagnostics and ablation.

This creates two independent views of the desired update:

1. a supervised neural friction innovation;
2. a local inverse-dynamics correction derived from the current dynamics residual.

The auxiliary agreement term encourages consistent direction without turning the inverse-dynamics approximation into a hard estimator.

## 7. Final authority and bounded update

The final authority is

\[
K_t=I_tV_t,
\qquad 0\le K_t\le1.
\]

The updated latent state and friction estimate are

\[
q_t=q_t^-+K_t\nu_t,
\]

\[
\boxed{\hat\mu_t=L_t+(U-L_t)\sigma(q_t)}.
\]

Thus \(L_t\le\hat\mu_t\le U\) by construction. If friction is locally unidentifiable, \(I_t\) suppresses the update. If the candidate is clearly counterfactually harmful, \(V_t\) attenuates it. Neutral dynamics evidence no longer imposes an arbitrary 0.5 penalty.

## 8. Training objective

Point training uses only training labels. Lower-bound calibration and predictive-UQ calibration remain disjoint from gradient training.

\[
\mathcal L =
\mathcal L_{point}
+\lambda_{innov}\mathcal L_{innov}
+\lambda_{cand}\mathcal L_{cand}
+\lambda_{dir}\mathcal L_{dir}
+\lambda_{agree}\mathcal L_{agree}
+\lambda_{dyn}\mathcal L_{dyn}
+\lambda_{cf}\mathcal L_{cf}
+\lambda_{harm}\mathcal L_{harm}.
\]

- `point`: Huber loss on the final bounded estimate;
- `innov`: direct supervision of raw \(\nu_t\) toward \(q_t^\star-q_t^-\);
- `cand`: Huber loss on \(\mu_t^{cand}\), independent of authority;
- `dir`: penalizes innovation direction opposite to the required latent correction;
- `agree`: identifiability-weighted agreement with the detached local inverse-dynamics correction;
- `dyn`: friction-conditioned dynamics reconstruction;
- `cf`: counterfactual ranking that requires the true training friction hypothesis to explain the dynamics better than displaced hypotheses;
- `harm`: penalizes final accepted updates that increase friction error relative to the prior.

The dynamics model is warm-started before joint training so its friction sensitivity is not random when it first influences authority.

## 9. Uncertainty

The residual-scale head is post-hoc, so UQ fitting cannot improve coverage by degrading the selected point estimator. Scale is inflated under low identifiability and calibrated on the disjoint UQ role using the repository's dependence-aware block procedure.

Conformal calibration is an evaluation/calibration layer, not part of the proposal novelty claim.

## 10. Primary ablations

1. `safegrip_backbone_raw` — raw temporal regressor only;
2. `safegrip_persistent` — persistent bounded state without innovation;
3. `safegrip_neural_innovation` — candidate innovation with unconditional authority;
4. `safegrip_no_identifiability` — asymmetric veto without counterfactual identifiability;
5. `safegrip_excitation_proxy` — handcrafted excitation in place of identifiability;
6. `safegrip_no_acceptance` — identifiability-only authority, isolating the asymmetric veto;
7. `safegrip_no_cf_agreement` — removes the local inverse-dynamics agreement objective;
8. `safegrip_no_innovation_supervision` — removes direct candidate supervision;
9. `safegrip_no_bound` — removes the sample-specific mechanics lower endpoint;
10. `safegrip_no_uq` — full point estimator without post-hoc UQ;
11. `safegrip` — full v0.9 estimator.

## 11. Statistical contract

v0.9 exports both `predictions.csv` and `predictions_by_seed.csv`.

- `metrics.csv` is the mean/std of independently trained seed metrics;
- `predictions.csv` is an explicitly labeled seed ensemble for diagnostics;
- statistical inference uses matched per-seed predictions and resamples trajectory segments, not overlapping endpoints as independent observations;
- diagnostic columns such as identifiability, veto probability, candidate state and persistence flags are excluded from the comparator registry.

## Claim boundary

The implementation is designed to be more distinctive and to directly address the v0.8 LiRA failure mode. The included synthetic sanity benchmark shows that the revised implementation can learn a non-degenerate positive-R2 estimator, but this is not evidence that LiRA performance has improved. The revised real-data trust run must be executed before making any new empirical claim.
