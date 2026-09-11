# SafeGrip-CI v1.2 — Risk-Aware Selective Physics Correction

## 1. Design principle

SafeGrip-CI v1.2 deliberately removes the recursive friction-state and multi-expert arbitration used by v1.1. The point estimate is owned by a strong raw-sensor temporal network. Physics is allowed to make only a bounded residual correction, and a learned utility gate decides how much of that correction to apply from evidence that is available at inference time.

The core pipeline is

`raw temporal sensors -> base friction estimate -> local dynamics evidence -> bounded inverse correction -> utility gate -> physical projection -> conformal UQ`.

No handcrafted excitation score is an input to the full proposal.

## 2. Temporal base estimator

For a causal window `X[t-L:t]`, a Conv1D front-end extracts short-range motion patterns and a multi-layer GRU carries longer temporal context. Endpoint/mean/std statistics are encoded in parallel and fused with the GRU representation:

\[
h_t = F_\theta(X_{t-L:t}).
\]

The primary estimate is

\[
\mu_t^N = \mu_{\max}\,\sigma(H_\mu(h_t)).
\]

This direct estimate is trained explicitly and does not depend on the previous predicted friction value.

## 3. Friction-conditioned dynamics model

A separate causal-prefix dynamics encoder predicts selected endpoint dynamics channels from context and a friction hypothesis:

\[
\widehat y_t^d = G_\phi(c_t,\mu).
\]

The dynamics branch is trained by its own supervised objective and a counterfactual ranking objective. Physics evidence is detached before entering the point-correction path; therefore the point loss cannot improve by distorting the dynamics model.

## 4. Counterfactual observability

Around the neural estimate, local friction sensitivity is approximated by a symmetric finite difference:

\[
J_t \approx
\frac{G_\phi(c_t,\mu_t^N+\delta)-G_\phi(c_t,\mu_t^N-\delta)}{2\delta}.
\]

Its energy is converted into a bounded observability feature

\[
I_t = \frac{\|J_t\|^2}{\|J_t\|^2+\lambda_I}.
\]

`I_t` means **local observability of friction**. It is not interpreted as a probability that either the neural estimate or physics correction is correct.

## 5. Bounded inverse-dynamics residual correction

With residual

\[
r_t = y_t^d-G_\phi(c_t,\mu_t^N),
\]

a damped one-step local inverse correction is

\[
\Delta\mu_t^{ID}
=\frac{J_t^\top r_t}{J_t^\top J_t+\lambda_{ID}}.
\]

The correction is trust-region bounded:

\[
\widetilde{\Delta\mu}_t^{ID}
=\operatorname{clip}(\Delta\mu_t^{ID},-\Delta_{\max},\Delta_{\max}).
\]

The inverse branch is **not** an independent full estimator in v1.2.

## 6. Learned correction-utility gate

A single gate consumes inference-available evidence: temporal representation, observability, dynamics-residual magnitude, normalized inverse step, regime-change probability, aleatoric scale, and sensitivity energy:

\[
g_t=\sigma\!\left(H_g(h_t,I_t,\|r_t\|,\widetilde{\Delta\mu}_t^{ID},p_t^{chg},\sigma_t,\|J_t\|^2)\right).
\]

The unprojected corrected estimate is

\[
\mu_t^{corr}=\mu_t^N + g_t\,s_{ID}\,\widetilde{\Delta\mu}_t^{ID}.
\]

During training, the gate is supervised by the useful fraction of the candidate physics correction. Let `d_t` be the bounded scaled physics correction and `e_t = mu_t - mu_t^N`. The target is

\[
g_t^* = \operatorname{clip}\!\left(\frac{e_t d_t}{d_t^2+\epsilon},0,1\right).
\]

This target uses friction labels only during training. At inference the gate receives no friction label.

## 7. Dynamic-change and regime supervision

The estimator is trained to reproduce friction variation, not only its absolute level:

\[
\mathcal L_\Delta = \operatorname{Huber}
\left((\hat\mu_t-\hat\mu_{t-1}),(\mu_t-\mu_{t-1})\right).
\]

A regime head predicts whether the reference friction changes more than a training threshold:

\[
p_t^{chg}=\sigma(H_c(h_t)).
\]

Conditional temporal smoothing is applied mainly to stable regimes, so abrupt surface changes are not suppressed by a persistent-state prior.

## 8. Risk-aware objective

Dangerous overestimation receives an explicit asymmetric penalty:

\[
\mathcal L_{unsafe}
=\left[\max(0,\hat\mu_t-\mu_t-\tau)\right]^2.
\]

The training objective combines

\[
\mathcal L =
\mathcal L_{point}
+\lambda_b\mathcal L_{base}
+\lambda_\Delta\mathcal L_\Delta
+\lambda_s\mathcal L_{unsafe}
+\lambda_g\mathcal L_{gate}
+\lambda_c\mathcal L_{change}
+\lambda_{sm}\mathcal L_{smooth}
+\lambda_h\mathcal L_{hetero}
+\lambda_d\mathcal L_{dyn}
+\lambda_{cf}\mathcal L_{cf}.
\]

The point and base losses use Smooth-L1/Huber regression.

## 9. Heteroscedastic evidence and uncertainty

The temporal representation also predicts a positive aleatoric scale. A heteroscedastic Gaussian-style residual objective trains this scale. It is then combined with observability/gate evidence in the learned residual-scale head. Final intervals use the repository's disjoint block-max split-conformal calibration; test labels never tune the point estimator or the gate.

## 10. Physical projection

After selective correction, the point estimate is projected into the conditional feasible friction set:

\[
\hat\mu_t=\Pi_{\mathcal C_t}(\mu_t^{corr}).
\]

Projection is treated as a safety constraint, not as evidence that physics improves point accuracy.

## 11. What is intentionally removed from the full v1.2 path

The full proposal no longer uses recursive friction persistence, adaptive persistence, direction/magnitude innovation factorization, prior/neural/inverse three-way arbitration, agreement vetoes, multi-scale local-linearity authority discounts, or chained multiplicative authority terms. Those historical mechanisms remain in the repository only where required to reproduce earlier experiments.

## 12. Primary ablations

The controlled primary variants are:

- `safegrip_base_temporal`
- `safegrip_no_dynamic_loss`
- `safegrip_no_safety_loss`
- `safegrip_no_physics_residual`
- `safegrip_no_utility_gate`
- `safegrip_no_identifiability`
- `safegrip_no_bound`
- `safegrip_no_regime_head`
- `safegrip_no_heteroscedastic`
- `safegrip_no_uq`
- `safegrip`

All use the same split identities and, in the primary frozen-hyperparameter table, the same selected full-model hyperparameters.

## 13. Scientific claim discipline

v1.2 is designed to test whether **selective physics correction improves the accuracy-safety trade-off of a strong temporal estimator**. It must not be claimed to outperform the GRU baseline until a fresh locked TRUST/PAPER run demonstrates that result with matched-seed/trajectory statistics. The previous v1.1 result is historical evidence that motivated this redesign, not a v1.2 performance result.
