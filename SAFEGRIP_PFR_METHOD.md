# SafeGrip-PFR-ECR: Excitation-Aware Conformal Risk-Controlled Residual Estimation

## 1. Design objective

The revised proposal keeps a single recurrent backbone but changes how physics is
used.  Mechanics is no longer only a broad output clipping interval.  Instead,
it has two roles:

1. a mechanics-derived lower-grip signal anchors the residual target and marks
   temporally informative/high-excitation observations;
2. a separately calibrated mechanics lower estimate is fused with a one-sided
   conformal lower estimate from the learned predictor for controller-facing
   safety use.

The method therefore exposes two outputs with different purposes:

- `mu_point`: the accuracy-oriented friction estimate;
- `mu_safe`: a conservative controller-facing lower friction estimate.

The test labels are used only for final evaluation.

## 2. Mechanics anchor and excitation

Let `L0_t` denote the instantaneous mechanics-derived lower-grip signal and let
`W_t` be a trailing, trajectory-local evidence window.  The residual anchor is

\[
L^W_{0,t}=\max_{j\in W_t}L_{0,j}.
\]

The learned signed residual target is

\[
r_t^\star=\mu_t-L^W_{0,t}.
\]

For every recurrent input timestep, the network also receives the instantaneous
mechanics channel and the dimensionless excitation

\[
e_j=\operatorname{clip}(L_{0,j}/U,0,1).
\]

These channels use only observable vehicle signals and configured mechanics
parameters; they do not use friction labels.

## 3. Excitation-aware temporal encoder

A GRU produces hidden states

\[
h_j=\operatorname{GRU}_\theta(x_j).
\]

Physics determines deterministic temporal weights

\[
a_j=\frac{\exp(\gamma e_j)}{\sum_k\exp(\gamma e_k)},
\qquad
h_t^{\rm phys}=\sum_j a_jh_j.
\]

The parameter `gamma` is selected on validation data only.  `gamma=0` gives
uniform temporal pooling and is retained as the decisive no-attention ablation.

The residual head gives a standardized residual estimate, transformed back to
physical units as `r_theta`.  The point estimate is

\[
\boxed{\mu_{\rm point,t}=L^W_{0,t}+r_\theta(h_t^{\rm phys}).}
\]

A second scalar head returns

\[
\sigma_t=\operatorname{softplus}(s_\phi(h_t^{\rm phys}))+\varepsilon>0.
\]

The scale is not interpreted as a parametric Gaussian confidence interval.  It
is used only to normalize the one-sided conformal nonconformity score.

## 4. Training objective

The point head is trained on the training split only using a robust loss in
standardized residual coordinates.  The scale head receives a small Gaussian
negative-log-likelihood auxiliary term so it can adapt to heteroscedastic error.
Model selection uses validation point-estimate MSE only.

No calibration or test friction labels are used in neural optimization.

## 5. Risk allocation

Let the desired total miscoverage be `alpha_total`.  The implementation uses a
fixed split

\[
\beta=\rho\alpha_{\rm total},
\qquad
\alpha_s=(1-\rho)\alpha_{\rm total},
\]

where `rho=risk_split` is a protocol parameter, not a test-tuned value.  The
default is `rho=1/2`.

## 6. Calibrated mechanics lower estimate

On calibration points define

\[
s_i^{\rm phys}=L^W_{0,i}-\mu_i.
\]

Let `q_beta` be the finite-sample higher split-conformal quantile.  The mechanics
lower estimate is

\[
\boxed{L_{\beta,t}=\max\{0,L^W_{0,t}-q_\beta\}.}
\]

Under the usual exchangeability assumption,

\[
\Pr\{L_{\beta,new}\le\mu_{new}\}\ge 1-\beta.
\]

## 7. Normalized one-sided conformal learned lower estimate

For the fitted point/scale network, calibration scores are

\[
z_i=\frac{\mu_{{\rm point},i}-\mu_i}{\sigma_i}.
\]

Let `q_s` be the finite-sample higher quantile at level `1-alpha_s`.  The code
clips `q_s` below by zero, which can only make the resulting lower estimate more
conservative.  Define

\[
\boxed{C_{\alpha_s,t}=\mu_{\rm point,t}-q_s\sigma_t.}
\]

Then, under split-conformal exchangeability,

\[
\Pr\{C_{\alpha_s,new}\le\mu_{new}\}\ge1-\alpha_s.
\]

## 8. Controller-facing safe output

The final safety value is

\[
\boxed{
\mu_{\rm safe,t}
=
\max\{L_{\beta,t},C_{\alpha_s,t}\},
}
\]

followed only by clipping to physical support `[0,U]`.

This is intentionally not used as the primary RMSE estimate.  It is a lower
operational value intended to reduce friction overestimation.

## 9. Joint coverage theorem

Suppose

\[
L_\beta\le\mu
\quad\text{and}\quad
C_{\alpha_s}\le\mu.
\]

Then

\[
\max\{L_\beta,C_{\alpha_s}\}\le\mu.
\]

Therefore, using the union bound,

\[
\Pr\{\mu_{\rm safe}\le\mu\}
\ge
1-\beta-\alpha_s
=
1-\alpha_{\rm total}.
\]

The probability statement requires the usual split-conformal exchangeability
assumption.  Independence between the two component coverage events is not
required for the union bound.

## 10. Decisive ablations

The benchmark now reports a larger controlled ablation set.  The primary
point-estimator chain is:

1. `direct_gru_control`: direct friction regression on raw sensors;
2. `pfr_residual_raw`: residual regression on raw sensors with endpoint GRU
   representation;
3. `pfr_no_physics_channels_uniform`: same PFR tensor shape and uniform pooling,
   but both mechanics-derived input channels are zeroed;
4. `pfr_physics_features_no_attention`: explicit PFR mechanics channels with
   uniform temporal pooling;
5. `safegrip_pfr`: full excitation-aware point estimator.

Three additional component controls isolate otherwise confounded choices:

- `pfr_direct_target_full`: same full PFR architecture but direct friction
  supervision, testing whether the explicit residual anchor contributes;
- `pfr_no_scale_multitask`: full point path with the heteroscedastic NLL
  auxiliary weight set to zero, testing whether scale multi-task learning
  changes point accuracy;
- `pfr_physics_features_no_attention` versus
  `pfr_no_physics_channels_uniform`: isolates physics-input information while
  holding tensor dimensionality and uniform pooling fixed.

The safety path is decomposed without retraining:

- `pfr_safe_mechanics_only`: calibrated mechanics lower estimate only;
- `pfr_safe_statistical_only`: normalized conformal statistical lower estimate
  only;
- `safegrip_pfr_safe`: fused controller-facing lower estimate.

Finally, `pfr_risk_split_sensitivity.csv` reports calibration-only sensitivity
to mechanics/statistical risk allocations of 0.25/0.75, 0.50/0.50, and
0.75/0.25 by default.  Because the learned point estimator is held fixed, this
analysis does not consume additional training budget.

A particularly important interpretation rule is that `pfr_excitation` is a
normalized transform of `pfr_mechanics_lower` in the current implementation.
The zero-both-channels control therefore tests the mechanics-derived information
as a block and avoids over-claiming two independent physical inputs.

## 11. Reporting contract

Primary predictive comparison uses `safegrip_pfr = mu_point` and reports MAE,
RMSE, R2, and overestimation statistics.  Safety reporting is separate and
includes:

- empirical mechanics lower coverage;
- empirical normalized conformal lower coverage;
- empirical fused safe coverage;
- unsafe-overestimation rate before and after safety correction;
- mean point-to-safe gap;
- predicted scale and attention diagnostics.

The benchmark also writes a deterministic fusion audit.  A `PASS` there checks
the algebraic implication "both component lower estimates covered => fused
lower estimate covered".  It is not presented as an empirical proof of the
exchangeability assumption or of population coverage.

## Primary paper-supported comparison

SafeGrip-PFR-ECR is the only active proposal. The primary accuracy table contains exactly four paper-supported road-friction/grip comparators: `du2023_inceptiontime`, `todorovic2022_cnn`, `lampe2023_gru`, and `levenberg2023_stft`. Internal controls such as `direct_gru_control` remain only in the component-ablation output and are not presented as literature baselines.

The comparison uses common locked endpoints and a common continuous target. Du2023 is restricted to the dynamics-only InceptionTime branch so it receives no vision advantage. Levenberg2023 is explicitly a low-rate STFT method-structure adaptation because the repository's 20-Hz common LiRA stream cannot reproduce the paper's high-frequency vibration band. Full provenance is exported to `paper_baseline_provenance.csv`; see `LITERATURE_BASELINES.md`.

