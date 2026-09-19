# SafeGrip-FRC: Finite-Window Friction Resolution Certification

## 1. Scope

SafeGrip-FRC is the active proposal. It intentionally replaces the large SafeGrip-CI v1.4 correction/controller stack with one causal response model and one finite-grid inverse-identification rule. The neural architecture is not claimed as novel; the research contribution is the friction-specific finite-window resolution construction and its executable recovery certificate.

The legacy CI-v1.4 implementation remains in the repository for reproducibility and can be selected with `safegrip benchmark --proposal legacy-ci`.

## 2. Causal response model

Let `c_k` contain only sensor samples strictly before the response ending at sample `k`. For a hypothetical friction coefficient `mu`, the learned model predicts a standardized response innovation

\[
\widehat g_\theta(c_k,\mu)\in\mathbb R^r.
\]

The default response channels are `ax`, `ay`, and the four wheel-speed channels when available. Response normalization is fitted on training data only.

For a finite horizon `H`, stack the observations and model responses:

\[
Y_{t,H}=\begin{bmatrix}\Delta z_{t-H+1}\\\vdots\\\Delta z_t\end{bmatrix},
\qquad
\widehat\Phi_{t,H}(\mu)=\begin{bmatrix}\widehat g_\theta(c_{t-H+1},\mu)\\\vdots\\\widehat g_\theta(c_t,\mu)\end{bmatrix}.
\]

The code constructs each context as `X[j:j+C]` and the corresponding response from the next endpoint. The response endpoint is therefore never included in its own context.

## 3. Grid inverse estimator

Let the declared candidate grid be

\[
\mathcal G=\{\mu_1,\ldots,\mu_M\}.
\]

Define

\[
J_{t,H}(\mu)=\|Y_{t,H}-\widehat\Phi_{t,H}(\mu)\|_2^2,
\]

and estimate

\[
\widehat\mu_{t,H}\in\arg\min_{\mu\in\mathcal G}J_{t,H}(\mu).
\]

No learned gate, correction head, entropy posterior, or physical projection is part of the primary estimator.

## 4. Finite-window friction separation

For a requested friction resolution `delta`, define the finite-grid separation margin

\[
S_{t,H}(\delta)
=
\min_{\substack{\mu_i,\mu_j\in\mathcal G\\|\mu_i-\mu_j|\ge\delta}}
\|\widehat\Phi_{t,H}(\mu_i)-\widehat\Phi_{t,H}(\mu_j)\|_2.
\]

`S(delta)` is nondecreasing in `delta`; this property is unit-tested.

## 5. Finite-grid recovery theorem

Let `q(mu*)` denote the nearest grid point to the true coefficient `mu*`. Suppose

\[
\|Y_{t,H}-\widehat\Phi_{t,H}(q(\mu^*))\|_2\le r_H.
\]

If

\[
S_{t,H}(\delta)>2r_H,
\]

then every grid minimizer satisfies

\[
|\widehat\mu_{t,H}-q(\mu^*)|<\delta.
\]

For a uniform grid with maximum spacing `h_mu`, nearest-grid approximation gives

\[
|\widehat\mu_{t,H}-\mu^*|<\delta+\frac{h_\mu}{2}.
\]

The proof is the triangle inequality plus the residual-minimizer property. The repository does not claim that this proof technique is new mathematics; the contribution is the friction-specific finite-window separation/certificate construction and its use for estimation/horizon selection.

## 6. Calibration residual radius

For each horizon, calibration data are used only to measure the nearest-grid forward residual norm. The default code uses the empirical 0.95 quantile:

\[
r_H=Q_{0.95}\bigl(\|Y_{i,H}-\widehat\Phi_{i,H}(q(\mu_i))\|_2\bigr).
\]

This is called an **empirical calibration residual radius**. It is not advertised as a universal 95% coverage guarantee under arbitrary dependence.

## 7. Resolution certificate and horizon selection

The discrete certificate is

\[
\delta^{\rm cert}_{t,H}
=
\min\{\delta\in\mathcal D:S_{t,H}(\delta)>2r_H\}.
\]

The continuous-error certificate reported by code is

\[
e^{\rm cert}_{t,H}=\delta^{\rm cert}_{t,H}+h_\mu/2.
\]

The adaptive horizon is

\[
H_t^*\in\arg\min_H e^{\rm cert}_{t,H}.
\]

If no candidate horizon is certified, the declared fallback horizon is used and the endpoint is marked `certified=0` rather than inventing a guarantee.

## 8. Primary comparison

The primary table includes:

- `safegrip_frc`;
- `direct_gru_control`, using the same GRU encoder width/depth and closely matched head capacity but direct `mu` regression;
- the registered literature adaptations.

Physical projection is excluded from primary metrics and exported separately as `projection_control.csv`.

## 9. Required outputs

A paper run exports:

- `metrics.csv`, `metrics_by_seed.csv`;
- `predictions_by_seed.csv`;
- `fixed_horizon_ablation.csv`;
- `frc_resolution_certificates.csv`;
- `frc_separation_curves.csv`;
- `frc_certificate_validity.csv`;
- `theorem_audit.json`;
- `fairness_audit.json`;
- `projection_control.csv`;
- `reproducibility_manifest.json`.
