# SafeGrip-PNTR: Physics-Neural Trust-Region Friction Estimation

## 1. Why the proposal changed

The previous SafeGrip-FRC estimator solved a global inverse problem over the full friction grid. On the LiRA development run, its causal response model was insufficiently discriminative in friction: the direct GRU control was accurate, while global response inversion could jump to a distant friction candidate. SafeGrip-PNTR changes the estimator so that a good neural prediction is never discarded by an unconstrained global inverse search.

The active proposal is now deliberately centered on **physics + neural estimation**:

1. a GRU predicts the **non-negative friction slack above a mechanics lower bound**;
2. the physical lower bound is therefore embedded directly in the neural output parameterization;
3. a friction-conditioned neural response model supplies local physical evidence;
4. a regularized trust-region inverse step may refine the physics-neural anchor only when the observed dynamics support it.

The previous FRC and CI-v1.4 implementations remain available for reproducibility.

## 2. Physics-residual neural anchor

Let \(\ell_t\) be the calibration-relaxed vehicle-mechanics lower grip bound and let \(\mu_U\) be the declared upper bound. Instead of asking the GRU to learn friction from zero, PNTR asks it to learn only the non-negative **slack above the physical lower bound**,

\[
s_t^*=\max(\mu_t-\ell_t,0).
\]

For an observation window \(X_t\), the GRU predicts a standardized slack variable

\[
\widehat z_t=f_\theta(X_t),
\qquad
\widehat s_t=s_s\widehat z_t+m_s,
\]

where \(m_s\) and \(s_s\) are fitted on the **training split only**. The physics-neural anchor is then

\[
\boxed{
\mu_t^0
=
\min\{\mu_U,\;\ell_t+\max(\widehat s_t,0)\}.
}
\]

Therefore the mechanics constraint is part of the neural parameterization itself:

\[
\ell_t\le\mu_t^0\le\mu_U
\]

for every endpoint, without needing test labels or a post-hoc clipping rule. A same-encoder direct-\(\mu\) GRU is trained as `direct_gru_control`; this isolates whether the physics-residual parameterization contributes beyond ordinary GRU capacity.

## 3. Mechanics admissible interval

The lower bound \(\ell_t\) is computed from measured vehicle dynamics and the repository's nominal force-balance model, then calibration-relaxed using training/calibration information. At inference time it uses observed vehicle signals only. The trust-region refinement in Section 5 is centered on the already admissible physics-neural anchor \(\mu_t^0\).

## 4. Friction-conditioned causal response model

For a causal context \(c_k\) and a hypothetical friction value \(\mu\), the response model predicts the standardized next response innovation

\[
\widehat g_\psi(c_k,\mu).
\]

The observed response innovation is \(r_k\). For a horizon \(H\), define

\[
E_{t,H}(\mu)
=
\frac{1}{Hd_r}
\sum_{k=t-H+1}^{t}
\left\|r_k-\widehat g_\psi(c_k,\mu)\right\|_2^2.
\]

A plain forward MSE can let the network ignore its friction input. PNTR therefore adds a small friction-discriminative training term. For nearby wrong hypotheses \(\mu^+\) and \(\mu^-\),

\[
\mathcal L_{\rm rank}
=
\big[m+e(\mu)-e(\mu^+)\big]_+
+
\big[m+e(\mu)-e(\mu^-)\big]_+,
\]

where \(e(\cdot)\) is the response MSE for one transition. The response-model training objective is

\[
\mathcal L_{\rm resp}
=
\mathcal L_{\rm MSE}
+\lambda_{\rm rank}\mathcal L_{\rm rank}.
\]

This term is applied only with training labels. Test friction labels are never used by the response model or the inverse step.

## 5. Physics-neural trust-region estimator

Let \(\tau>0\) be the maximum allowed physics correction. The local admissible interval is

\[
\mathcal T_t=
[\ell_t,\mu_U]
\cap
[\mu_t^0-\tau,\mu_t^0+\tau].
\]

For each candidate \(\mu\in\mathcal T_t\), PNTR minimizes the regularized objective

\[
F_{t,H}(\mu)
=
E_{t,H}(\mu)
+
\lambda_A
\left(\frac{\mu-\mu_t^0}{\tau}\right)^2.
\]

The exact neural anchor \(\mu_t^0\) is always an admissible fallback. A nonzero physics correction is accepted only when the candidate lowers both the regularized objective and the response energy by the configured minimum amount. If no horizon produces a valid improvement, PNTR returns \(\mu_t^0\).

The selected horizon is the horizon with the largest positive reduction in regularized objective. Because \(E_{t,H}\) is an average per response component, the scores are comparable across horizons.

## 6. Deterministic safeguard

Let \(\widehat\mu_{t,H}\) be the PNTR candidate for horizon \(H\). Because the exact anchor is a candidate with zero anchor penalty,

\[
F_{t,H}(\widehat\mu_{t,H})
\le
F_{t,H}(\mu_t^0)
=
E_{t,H}(\mu_t^0).
\]

Therefore every accepted correction satisfies

\[
E_{t,H}(\widehat\mu_{t,H})
+
\lambda_A
\left(\frac{\widehat\mu_{t,H}-\mu_t^0}{\tau}\right)^2
\le
E_{t,H}(\mu_t^0),
\]

and, by construction,

\[
|\widehat\mu_{t,H}-\mu_t^0|\le \tau.
\]

This is an executable deterministic safeguard. It does **not** claim that true friction error improves on every endpoint; that stronger statement requires assumptions linking response-model energy to friction error.

## 7. Conditional local-recovery result

For a continuous local analysis, suppose the physical response energy \(E(\mu)\) is \(m\)-strongly convex on the trust interval, and suppose the response-model/noise mismatch at the true friction \(\mu^*\) satisfies

\[
|E'(\mu^*)|\le\eta.
\]

For the regularized objective

\[
F(\mu)=E(\mu)+\frac{\lambda}{2}(\mu-\mu^0)^2,
\]

an interior minimizer \(\widehat\mu\) obeys

\[
|\widehat\mu-\mu^*|
\le
\frac{\eta+\lambda|\mu^0-\mu^*|}{m+\lambda}.
\]

Thus the estimator has the intended behavior:

- when local physical information is strong (large \(m\)) and model mismatch is small, the response term can improve the neural anchor;
- when local physics is weak, the anchor regularization dominates and prevents a large inverse-identification jump.

The code reports local finite-difference identifiability as a diagnostic. The strong-convexity assumption is **not** asserted automatically from that diagnostic.

## 8. Primary ablations

Each PNTR benchmark exports the following same-training-run ablations:

- `direct_gru_control`: same GRU encoder predicting friction directly, without the physics-residual parameterization;
- `pntr_physics_residual_anchor`: GRU prediction of non-negative slack above the mechanics lower bound;
- `safegrip_pntr`: physics-residual anchor plus trust-region response refinement;
- `pntr_fixed_h4`, `pntr_fixed_h8`, `pntr_fixed_h16`, `pntr_fixed_h32`: fixed-horizon refinements.

These directly answer whether gains come from the neural estimator, the explicit mechanics bound, the response-based physical refinement, or adaptive horizon selection.

## 9. Claims deliberately not made

SafeGrip-PNTR does not claim that GRUs, target standardization, trust-region regularization, or counterfactual ranking are individually new. The research claim should be restricted to the friction-estimation formulation and the specific physics-neural safeguarded estimator, together with its deterministic objective safeguard and conditional local-recovery analysis.
