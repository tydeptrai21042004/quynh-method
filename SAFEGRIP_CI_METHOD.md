# SafeGrip-CI method (v1.0.0)

## Core contribution

SafeGrip-CI v1.0 is a **persistent friction-state estimator with multi-scale counterfactual observability and an asymmetric trust-region update**. A learned innovation proposes how the friction state should change; a friction-conditioned dynamics model then asks whether nearby friction hypotheses are distinguishable, whether that sensitivity is locally consistent across perturbation scales, and whether the learned update agrees with a local inverse-dynamics correction. The mechanics lower endpoint remains a support constraint rather than a substitute label.

The central distinction is that **candidate estimation and update authority are separate problems**. The candidate is directly supervised, while its authority is determined by label-free counterfactual evidence at inference.

## 1. Persistent bounded friction state

At a segment start, a causal context encoder predicts a bounded context prior. At later endpoints, the previous estimate is blended with that prior,

\[
\mu_t^- = \rho\hat\mu_{t-1} + (1-\rho)\mu_t^{ctx}.
\]

For mechanics lower endpoint \(L_t\) and global upper support \(U\),

\[
q_t^- = \operatorname{logit}\!\left(\frac{\mu_t^- - L_t}{U-L_t}\right).
\]

The persistent state is reset at every trajectory-segment boundary.

## 2. Independently supervised neural innovation

A short causal raw-sensor encoder predicts

\[
\nu_t = \Delta_{max}\tanh f_\theta(X_{t-H:t}).
\]

The unconstrained candidate is

\[
q_t^{cand}=q_t^-+\nu_t,
\qquad
\mu_t^{cand}=L_t+(U-L_t)\sigma(q_t^{cand}).
\]

With training target

\[
q_t^\star=\operatorname{logit}\!\left(\frac{\mu_t^\star-L_t}{U-L_t}\right),
\qquad
\nu_t^\star=q_t^\star-q_t^-,
\]

the raw innovation is supervised before authority is applied. This prevents a low gate value from hiding an inaccurate candidate.

## 3. Friction-conditioned counterfactual dynamics

A causal dynamics model predicts selected standardized endpoint responses from the window prefix and a friction hypothesis,

\[
\widehat y_t^{dyn}=G_\phi(X_{t-H:t-1},\mu).
\]

The current endpoint response is excluded from the dynamics encoder, so the counterfactual residual cannot be solved by directly copying the quantity it is asked to explain.

## 4. Multi-scale counterfactual observability

A single finite-difference displacement can produce a spuriously high sensitivity because of local curvature or model artifacts. v1.0 therefore evaluates three symmetric friction perturbation scales,

\[
\delta_s\in\left\{\frac{\delta}{c},\delta,c\delta\right\}, \qquad c\ge 1.
\]

At each scale,

\[
J_t^{(s)} \approx
\frac{G_\phi(X,\mu_t^-+\delta_s)-G_\phi(X,\mu_t^- -\delta_s)}
{(\mu_t^-+\delta_s)-(\mu_t^- -\delta_s)}.
\]

Define the scale-specific information

\[
I_t^{(s)}=\frac{1}{d_y}\left\|J_t^{(s)}\right\|_2^2.
\]

The robust local information is the median across scales,

\[
\widetilde I_t=\operatorname{median}_s I_t^{(s)}.
\]

To detect sensitivity that changes strongly with perturbation scale, compute

\[
CV_t=\frac{\operatorname{std}_s I_t^{(s)}}
{\operatorname{mean}_s I_t^{(s)}+\varepsilon},
\qquad
C_t=\frac{1}{1+\lambda_C CV_t^2}.
\]

The final observability/identifiability score is

\[
I_t=\frac{\widetilde I_t}{\widetilde I_t+\lambda_I}\,C_t,
\qquad 0\le I_t\le1.
\]

This makes authority depend not only on the magnitude of a learned friction sensitivity, but also on whether that sensitivity is stable in the local counterfactual neighborhood.

## 5. Normalized asymmetric residual veto

Let

\[
R_t^- = \|y_t^{dyn}-G_\phi(X,\mu_t^-)\|_2^2,
\qquad
R_t^{cand} = \|y_t^{dyn}-G_\phi(X,\mu_t^{cand})\|_2^2.
\]

Use the scale-free improvement

\[
z_t=\frac{R_t^- - R_t^{cand}}
{R_t^- + R_t^{cand}+\varepsilon},\qquad -1\le z_t\le1.
\]

The residual veto probability is

\[
p_t^{res}=\sigma\{-\gamma_r(z_t+\tau_r)\},
\]

and the corresponding multiplier is

\[
V_t^{res}=1-\rho_r p_t^{res}.
\]

The mechanism is intentionally asymmetric: neutral evidence does not halve every update; substantial attenuation requires affirmative evidence that the candidate explains the observed dynamics worse than the prior.

## 6. Local inverse-dynamics correction and agreement

Using the mean multi-scale sensitivity

\[
\bar J_t=\frac{1}{S}\sum_sJ_t^{(s)},
\]

and prior residual vector

\[
e_t=y_t^{dyn}-G_\phi(X,\mu_t^-),
\]

a damped one-step inverse-dynamics correction is

\[
\Delta\mu_t^{cf}=
\operatorname{clip}\!\left(
\frac{\bar J_t^\top e_t}{\bar J_t^\top\bar J_t+\lambda_{GN}},
-r_{max},r_{max}
\right).
\]

The learned candidate correction is

\[
\Delta\mu_t^{nn}=\mu_t^{cand}-\mu_t^-.
\]

v1.0 uses a genuine \([0,1]\) agreement score,

\[
A_t=\operatorname{clip}\!\left(
1-\frac{|\Delta\mu_t^{nn}-\Delta\mu_t^{cf}|}
{|\Delta\mu_t^{nn}|+|\Delta\mu_t^{cf}|+\varepsilon},0,1
\right).
\]

Thus matching corrections approach 1 and opposite-direction corrections approach 0. The previous implementation unintentionally compressed disagreement into the upper half of the range; this is corrected in v1.0.

## 7. Agreement-aware asymmetric trust region

Strong candidate/inverse-dynamics disagreement provides a second veto,

\[
p_t^{agr}=\sigma\{\gamma_a(\tau_a-A_t)\}.
\]

Because the inverse step is itself unreliable when friction is unobservable, its veto is weighted by the same counterfactual observability,

\[
V_t^{agr}=1-\rho_a\,p_t^{agr}I_t.
\]

The combined update authority is

\[
\boxed{K_t=I_tV_t^{res}V_t^{agr}},
\qquad 0\le K_t\le1.
\]

This yields a dual-evidence trust region: a correction is suppressed when friction is locally unobservable, when the candidate worsens the dynamics residual, or when an observable local inverse-dynamics direction strongly contradicts the neural candidate.

## 8. Final bounded update

The final state is

\[
q_t=q_t^-+K_t\nu_t,
\]

\[
\boxed{\hat\mu_t=L_t+(U-L_t)\sigma(q_t)}.
\]

Therefore \(L_t\le\hat\mu_t\le U\) by construction.

## 9. Training objective and curriculum

Point training uses only the training-label role. Lower-bound calibration and predictive-UQ calibration remain disjoint from gradient training.

\[
\mathcal L =
\mathcal L_{point}
+\lambda_{innov}\mathcal L_{innov}
+\lambda_{update}\mathcal L_{update}
+\lambda_{cand}\mathcal L_{cand}
+\lambda_{dir}\mathcal L_{dir}
+\lambda_{agree}\mathcal L_{agree}
+\lambda_{dyn}\mathcal L_{dyn}
+\lambda_{rank}\mathcal L_{rank}
+\lambda_{harm}\mathcal L_{harm}.
\]

- `point`: Huber loss on the final bounded state;
- `innov`: direct supervision of the raw latent innovation;
- `update`: supervision of the authority-weighted innovation;
- `cand`: candidate-point Huber loss before authority;
- `dir`: sign/direction regularizer on meaningful latent corrections;
- `agree`: identifiability-weighted agreement with the detached inverse-dynamics correction;
- `dyn`: friction-conditioned dynamics reconstruction;
- `rank`: true-friction dynamics must outperform displaced friction hypotheses by a margin;
- `harm`: penalizes accepted updates that increase friction error relative to the persistent prior.

The dynamics branch is warm-started before joint training. This avoids letting random friction sensitivity control the estimator at the beginning of optimization.

## 10. Uncertainty

The residual-scale head is fitted after point-model selection. Scale can be inflated under low counterfactual identifiability and then calibrated on the disjoint UQ role with the dependence-aware block procedure. Conformal calibration is an evaluation/calibration layer and is not part of the core novelty claim.

## 11. Controlled ablations

### Primary mechanism ablations

1. `safegrip_backbone_raw` — raw temporal regression only;
2. `safegrip_persistent` — persistent bounded state without innovation;
3. `safegrip_neural_innovation` — neural innovation with unconditional authority;
4. `safegrip_no_identifiability` — no counterfactual observability score;
5. `safegrip_excitation_proxy` — handcrafted excitation replaces counterfactual observability;
6. `safegrip_no_acceptance` — removes residual-based veto;
7. `safegrip_no_cf_agreement` — removes inverse-dynamics agreement training and inference use;
8. `safegrip_single_scale_cf` — returns to one finite-difference scale;
9. `safegrip_no_linearity_consistency` — retains multi-scale sensitivity but removes the cross-scale consistency discount;
10. `safegrip_no_agreement_veto` — retains agreement supervision but removes its inference-time veto;
11. `safegrip_no_counterfactual_ranking` — removes dynamics counterfactual ranking;
12. `safegrip_no_innovation_supervision` — removes direct candidate supervision;
13. `safegrip_no_bound` — removes the sample-specific mechanics lower support;
14. `safegrip_no_uq` — full point estimator without post-hoc UQ;
15. `safegrip` — full v1.0 estimator.

### Supplementary optimization ablations

- `safegrip_no_state_update_loss`;
- `safegrip_no_direction_loss`;
- `safegrip_no_dynamics_pretrain`.

These are kept separate from the core mechanism table because they test optimization design rather than the estimator's inference structure.

## 12. Hyperparameter stability contract

The repository now supports two distinct experiments:

1. **Optuna selection** (`safegrip tune`) chooses point-model hyperparameters using validation RMSE only.
2. **One-factor sensitivity** (`safegrip sensitivity`) freezes the selected model settings and sweeps scientifically important proposal parameters over their predeclared grid on the same locked validation endpoints.

The second experiment is not used to select a better test result. It is evidence about stability around the selected configuration.

## Claim boundary

v1.0 creates a sharper methodological distinction from generic physics-informed regression, adaptive observer gains, handcrafted excitation factors, and uncertainty-only friction estimators. However, the repository does **not** claim that this exact combination is the first ever without a formal exhaustive literature/patent search. It also does not claim improved LiRA accuracy until the full leakage-safe real-data tuning and multi-seed benchmark have been run.
