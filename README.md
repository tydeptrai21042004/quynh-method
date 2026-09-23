# SafeGrip-Open v2.3 — SafeGrip-PFR-ECR

The active proposal is **SafeGrip-PFR-ECR: Excitation-Aware Conformal
Risk-Controlled Residual Estimation**.  Earlier FRC, PNTR, and CI methods remain
for reproducibility.

## Active method

PFR-ECR uses one GRU backbone.  It adds the mechanics lower-grip signal to the
input and uses normalized mechanics excitation to weight the recurrent hidden
states:

\[
a_j=\operatorname{softmax}(\gamma L_{0,j}/U),
\qquad
h^{\rm phys}=\sum_j a_jh_j.
\]

The point estimate is

\[
\mu_{\rm point}=L_0^W+r_\theta(h^{\rm phys}).
\]

A positive scale head produces `sigma`.  Calibration creates a one-sided
statistical lower estimate

\[
C_{\alpha_s}=\mu_{\rm point}-q_s\sigma,
\]

which is fused with an independently calibrated mechanics lower estimate:

\[
\mu_{\rm safe}=\max\{L_\beta,C_{\alpha_s}\}.
\]

`mu_point` is the primary accuracy output.  `mu_safe` is reported separately as
the controller-facing conservative output.  With total risk split as
`beta + alpha_s = alpha_total`, the union-bound coverage statement is

\[
\Pr\{\mu_{\rm safe}\le\mu\}\ge1-\alpha_{\rm total}
\]

under the usual split-conformal exchangeability assumptions.

See [`SAFEGRIP_PFR_METHOD.md`](SAFEGRIP_PFR_METHOD.md) for the exact method,
assumptions, ablations, and theorem.

## Install and test

```bash
python -m pip install -e ".[paper,dev]"
pytest -q
```

## Quick LiRA run

```bash
safegrip download --datasets lira
safegrip prepare --dataset lira
safegrip benchmark --dataset lira --preset quick --proposal pfr --protocol controlled
```

## Paper workflow

```bash
bash scripts/run_paper.sh
```

For a separate source-setting comparator table:

```bash
RUN_SOURCE_FAITHFUL=1 bash scripts/run_paper.sh
```

## PFR tuning

```bash
safegrip tune --method pfr --dataset lira --trials 20
```

Validation-only tuning covers ordinary approximation parameters such as context
length, GRU width, attention strength, optimization settings, and mechanics
window length.  The total risk level, risk split, conformal quantile rule, and
physical support are not tuned against test performance.

## Main comparison and ablations

The main table contains `safegrip_pfr` (the point estimate),
`direct_gru_control`, and the registered literature baselines.

The PFR component table additionally contains:

- `pfr_residual_raw` — residual learning without explicit PFR physics channels;
- `pfr_physics_features_no_attention` — physics channels with uniform pooling;
- `safegrip_pfr` — full excitation-aware point estimator;
- `safegrip_pfr_safe` — controller-facing safety output, not the RMSE objective.

## Primary result files

```text
results/lira_paper/
  metrics.csv
  metrics_by_seed.csv
  predictions_by_seed.csv
  pfr_component_ablation.csv
  pfr_component_ablation_by_seed.csv
  pfr_safety_metrics.csv
  pfr_safety_metrics_by_seed.csv
  pfr_safety_fusion_audit.csv
  pfr_theorem_audit.json
  pfr_method_audit.json
  fairness_audit.json
  reproducibility_manifest.json
```

`pfr_safety_fusion_audit.csv` checks the deterministic fusion logic for every
endpoint/seed.  `pfr_theorem_audit.json` records the risk allocation, empirical
coverage diagnostics, and the union-bound statement.

## Fairness contract

The active benchmark uses:

- train-only input scaling and residual standardization;
- train-only neural optimization;
- validation-only hyperparameter selection;
- calibration-only mechanics and statistical conformal quantiles;
- locked validation/test endpoint IDs across compared methods;
- no test labels for fitting, calibration, or selection;
- no response inversion, friction-grid search, trust radius, or iterative
  projection solver.

## Retained legacy paths

```bash
safegrip benchmark --dataset lira --preset paper --proposal frc
safegrip benchmark --dataset lira --preset paper --proposal legacy-ci
```
