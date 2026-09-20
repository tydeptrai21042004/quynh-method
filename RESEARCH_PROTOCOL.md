# Research protocol — SafeGrip-PFR

## Primary question

Does a mechanics-translated residual hypothesis class improve friction
estimation relative to the same recurrent encoder trained directly on friction,
and does calibrated convex projection preserve or improve that raw residual
estimate under the stated physical-coverage assumptions?

## Data roles

The four prepared partitions have non-overlapping roles:

- **train**: neural parameter fitting and target/input scaling;
- **calibration**: estimation of the single one-sided conformal relaxation
  quantile `q_alpha`;
- **validation**: early stopping and neural hyperparameter selection;
- **test**: final reporting only.

No test friction label may affect training, calibration, model selection,
projection, or hyperparameter search.

## Proposal

\[
r^\star=\mu-L_0(X),
\qquad
\widetilde\mu=L_0(X)+f_\theta(X),
\]

\[
L_\alpha(X)=\max\{0,L_0(X)-q_\alpha\},
\qquad
\widehat\mu=\Pi_{[L_\alpha(X),U]}(\widetilde\mu).
\]

The mathematical quantities `alpha`, the conformal quantile convention, the
mechanics model, and `mu_upper` are protocol quantities and are not selected by
validation RMSE.

## Required controlled comparison

Use identical locked validation/test endpoint IDs for:

- `safegrip_pfr`;
- `direct_gru_control`;
- every registered literature comparator.

The direct GRU and residual GRU use the same architecture and optimizer budget.
Only the target parameterization differs.

## Decisive ablation

The required component table has only:

1. `direct_gru_control`;
2. `pfr_residual_raw`;
3. `safegrip_pfr`.

This is intentionally sufficient.  PFR has no gate, response network, trust
radius, horizon selector, or iterative inverse solver that would require extra
component ablations.

## Required theorem audit

For every seed and test endpoint, save

\[
G=(\widetilde\mu-\mu)^2-(\widehat\mu-\mu)^2
\]

and

\[
D^2=\operatorname{dist}(\widetilde\mu,[L_\alpha,U])^2.
\]

On covered endpoints the implementation must satisfy

\[
G\ge D^2
\]

up to numerical tolerance.  Any covered-point violation is a release-blocking
implementation error.

## Reporting

Report mean ± standard deviation across the declared evaluation seeds for:

- MAE;
- RMSE;
- R²;
- unsafe positive-overestimation mean;
- unsafe-overestimate rate at 0.05;
- raw residual RMSE;
- feasible-interval coverage;
- projection correction rate;
- mean squared error gain from projection.

Also report the number of test endpoints and independent trajectory segments.
Quick mode is a software/development run and must not be presented as the final
paper comparison.
