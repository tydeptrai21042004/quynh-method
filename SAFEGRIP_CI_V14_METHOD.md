# SafeGrip-CI v1.4 — Strong-Base Innovation-Energy Residual Correction

SafeGrip-CI v1.4 is the active proposal in this repository. It keeps the central v1.3 idea—counterfactual friction hypotheses evaluated by a learned dynamics model—but changes the point-estimation path to address the main failure mode observed in prior LiRA runs: the selective physics module was trying to rescue a weaker primary regressor.

## 1. Strong primary estimator

The primary friction estimate is produced by a two-layer raw-sensor GRU. The TRUST/default configuration uses a 100-sample context and 256 recurrent units, matching the scale of the strongest GRU literature comparator used by this repository.

Training labels are standardized using **training-only** statistics:

\[
z_\mu=(\mu-\bar\mu_{tr})/s_{tr}.
\]

The network predicts `z_mu` with a linear head and maps it back to physical units. There is no `mu_upper * sigmoid(z)` compression in the active v1.4 base head. Physical projection is applied after mapping back to friction units.

Stage A is accuracy-only and minimizes MSE. Safety, UQ, counterfactual, and selector objectives are excluded from this pretraining stage.

## 2. Innovation-conditioned dynamics

The counterfactual dynamics branch predicts endpoint innovation rather than the absolute endpoint:

\[
\Delta x_t=x_t-x_{t-1}.
\]

For a friction hypothesis \(\mu_k\), the energy is

\[
E_k=\|G_\phi(c_t,\mu_k)-\Delta x_t\|_2^2.
\]

This reduces the trivial temporal-persistence shortcut available to an absolute endpoint reconstruction model.

The dynamics branch is pretrained with reconstruction MSE plus a multi-negative friction-contrastive objective. By default it is frozen before point-path training.

## 3. Boundary-aware local hypothesis grid

The local counterfactual grid always contains the zero-offset/base hypothesis. Hypotheses on a physically impossible side are masked rather than duplicated through clipping at 0 or `mu_upper`.

For valid hypotheses, posterior weights are

\[
w_k \propto \exp(-E_k/(T s_E)).
\]

The correction is centered against the uniform distribution over valid hypotheses:

\[
\Delta\mu_{CF}=\sum_k (w_k-u_k)\delta_k.
\]

Therefore a flat energy landscape gives exactly zero correction even near a physical boundary.

## 4. Calibrated identifiability

Posterior entropy alone can make numerically tiny energy differences look informative after per-sample normalization. v1.4 therefore combines posterior concentration with absolute normalized energy-margin strength.

\[
I_H=1-H(w)/\log K_{valid},
\]

\[
M=(E_{base}-E_{min})/s_E,
\]

\[
I=I_H\,\sigma((M-m_I)/\tau_I).
\]

`energy_noise_floor` prevents the local scale from collapsing toward zero.

## 5. Single continuous correction controller

The v1.3 final gate was the product of benefit probability and correction fraction. v1.4 removes that double attenuation from the prediction path.

The active correction is

\[
\hat\mu=\Pi_C\{\mu_B+\alpha\,s_{CF}\,\Delta\mu_{CF}\},
\]

where `alpha = correction_fraction` is the single continuous controller and `s_CF` is the configured physics-correction scale.

`benefit_probability` is retained as an auxiliary interpretable output and for diagnostics, but it is not multiplied into the correction.

The oracle continuous target is

\[
\alpha^*=\operatorname{clip}\left(
\frac{(y-\mu_B)\Delta\mu_{CF}}{\Delta\mu_{CF}^2+\epsilon},0,1
\right).
\]

Magnitude supervision is softly weighted by candidate utility rather than restricted to a hard helpful/not-helpful subset.

## 6. Soft utility supervision

Candidate gain is

\[
g=|\mu_B-y|-|\mu_{CF}-y|.
\]

The auxiliary benefit target is

\[
q^*=\sigma((g-m_h)/\tau_h).
\]

This avoids discontinuous binary labels around tiny error differences.

## 7. Staged optimization

The active training protocol is:

1. **Stage A:** strong base GRU, standardized-target MSE only.
2. **Stage B:** innovation dynamics + contrastive friction discrimination.
3. **Stage C1:** correction-controller warm-up with base LR set to zero.
4. **Stage C2:** joint refinement with the base LR scaled down (default 0.1× controller LR).
5. **Stage D:** post-hoc residual-scale/conformal UQ calibration.

The default TRUST point-training configuration sets the asymmetric safety loss and heteroscedastic point-path loss to zero. `do_no_harm` remains a low-weight optional late regularizer. This deliberately separates point-accuracy learning from later safety/UQ analysis.

## 8. Oracle diagnostics

`selector_metrics` now reports:

- candidate direction accuracy;
- oracle switch RMSE;
- oracle continuous-correction RMSE;
- oracle mean correction fraction.

These diagnostics are evaluation-only. They are intended to distinguish three failure modes:

- weak base estimator;
- weak counterfactual candidate;
- weak selector/controller.

No oracle quantity is used for training or test-time inference.

## Claim boundary

The code changes are motivated by the failure analysis of earlier LiRA runs, but they do **not** establish that v1.4 beats a baseline. A new validation-selected, frozen-hyperparameter multi-seed TRUST/PAPER run is required before making an accuracy-superiority claim.
