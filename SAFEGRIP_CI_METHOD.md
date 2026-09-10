# SafeGrip-CI method (v0.8.0)

## Core contribution

SafeGrip-CI replaces the v0.7 handcrafted-excitation gate with a single estimation principle:

> update the persistent friction state only to the extent that current vehicle dynamics can counterfactually distinguish nearby friction hypotheses and the candidate state explains the observed dynamics better than the prior.

The full proposal uses raw vehicle-sensor channels. Handcrafted excitation is retained only as an explicit comparator ablation.

## 1. Persistent friction state

At the beginning of a trajectory segment, a causal context encoder provides a bounded context prior. At later endpoints the previous predicted friction is blended with the current context prior:

\[
\mu_t^- = \rho\,\hat\mu_{t-1} + (1-\rho)\,\mu_t^{ctx},
\]

where \(\rho\in[0,1]\) is `state_persistence`. The state is reset at each trajectory-segment boundary.

The prior is expressed in identified-set coordinates. For lower endpoint \(L_t\) and global upper support \(U\),

\[
q_t^- = \operatorname{logit}\!\left(\frac{\mu_t^- - L_t}{U-L_t}\right).
\]

## 2. Candidate neural innovation

A short causal raw-sensor encoder predicts a bounded latent innovation

\[
\nu_t = \Delta_{max}\tanh f_\theta(X_{t-H:t}).
\]

The unconstrained candidate state is

\[
q_t^{cand}=q_t^-+\nu_t,
\qquad
\mu_t^{cand}=L_t+(U-L_t)\sigma(q_t^{cand}).
\]

The innovation branch does not receive the handcrafted SafeGrip excitation score in the full method.

## 3. Friction-conditioned counterfactual dynamics model

A dynamics model \(G_\phi\) predicts selected standardized endpoint dynamics from the causal window prefix and a friction hypothesis:

\[
\hat y_t^{dyn}=G_\phi(X_{t-H:t-1},\mu).
\]

The current implementation uses available acceleration and wheel-speed channels when present. Because the endpoint target channels are excluded from the dynamics-model context, the residual test cannot be satisfied by copying the endpoint observation.

## 4. Counterfactual identifiability

Around the current prior, SafeGrip-CI evaluates two nearby friction hypotheses \(\mu_t^-\pm\delta\). A finite-difference sensitivity is

\[
J_t \approx
\frac{G_\phi(X,\mu_t^-+\delta)-G_\phi(X,\mu_t^- -\delta)}{2\delta}.
\]

The local information score is

\[
I_t^{raw}=\frac{1}{d_y}\lVert J_t\rVert_2^2,
\qquad
I_t=\frac{I_t^{raw}}{I_t^{raw}+\lambda_I}.
\]

Unlike an acceleration/torque excitation proxy, \(I_t\) measures whether changing friction changes the modelled vehicle response under the current maneuver.

## 5. Counterfactual acceptance

The same dynamics model evaluates whether the candidate friction actually improves consistency with the observed endpoint dynamics:

\[
R_t^- = \lVert y_t^{dyn}-G_\phi(X,\mu_t^-)\rVert_2^2,
\]

\[
R_t^{cand} = \lVert y_t^{dyn}-G_\phi(X,\mu_t^{cand})\rVert_2^2.
\]

The acceptance factor is

\[
A_t=\sigma\{\gamma[(R_t^- - R_t^{cand})-m_A]\}.
\]

## 6. Innovation authority and bounded update

The final authority is

\[
K_t=A_t I_t,
\qquad 0\le K_t\le1.
\]

The updated state is

\[
q_t=q_t^-+K_t\nu_t,
\]

and the friction estimate is

\[
\boxed{\hat\mu_t=L_t+(U-L_t)\sigma(q_t)}.
\]

Therefore \(L_t\le\hat\mu_t\le U\) by construction. If the learned dynamics are locally insensitive to friction, \(I_t=0\) and the innovation is blocked. If the candidate fails to improve dynamics consistency, \(A_t\) decreases its authority.

## 7. Training objective

Point training uses only training labels. Lower-bound calibration and predictive-UQ calibration remain disjoint from gradient training.

The main objective is

\[
\mathcal L =
\mathcal L_{point}
+\lambda_{innov}\mathcal L_{innov}
+\lambda_{dyn}\mathcal L_{dyn}
+\lambda_{cf}\mathcal L_{cf}
+\lambda_{harm}\mathcal L_{harm}.
\]

- `point`: Huber loss on final friction.
- `innov`: direct latent supervision of the accepted innovation \(K_t\nu_t\) toward the residual required to move the prior to the true friction.
- `dyn`: friction-conditioned dynamics reconstruction.
- `cf`: counterfactual ranking requiring the true friction hypothesis to explain the dynamics better than displaced hypotheses.
- `harm`: penalizes accepted updates that increase absolute friction error relative to the prior.

Before joint training, the friction-conditioned dynamics subnetwork is warm-started with dynamics reconstruction and counterfactual ranking. This prevents a random near-zero sensitivity model from suppressing all innovations at the start of optimization.

## 8. Uncertainty

The residual-scale head remains post-hoc so UQ training cannot trade away point accuracy. Native interval scale is inflated when counterfactual identifiability is low, then calibrated with the existing disjoint block-max split-conformal procedure.

Conformal calibration is a calibration layer, not part of the claimed novelty.

## 9. Primary ablations

The primary v0.8 ablations are:

1. `safegrip_backbone_raw` — raw temporal regression only;
2. `safegrip_persistent` — persistent bounded state without neural innovation;
3. `safegrip_neural_innovation` — persistent state + innovation with unconditional authority;
4. `safegrip_no_identifiability` — counterfactual acceptance without local identifiability;
5. `safegrip_excitation_proxy` — handcrafted excitation proxy instead of counterfactual identifiability;
6. `safegrip_no_acceptance` — identifiability without candidate-consistency acceptance;
7. `safegrip_no_innovation_supervision` — full architecture without direct innovation supervision;
8. `safegrip_no_bound` — remove the sample-specific mechanics lower endpoint;
9. `safegrip_no_uq` — retain the full point estimator but remove post-hoc UQ;
10. `safegrip` — full SafeGrip-CI.

The explicit ablation registry is unit-tested so distinct primary variants cannot silently resolve to identical model semantics.

## Claim boundary

The code implements and tests the structural properties above. It does not claim that counterfactual identifiability guarantees positive held-out R2, nor that the learned dynamics model is a complete tire model. Real LiRA trust results must pass the scientific-health gate before paper-mode results are treated as final evidence.
