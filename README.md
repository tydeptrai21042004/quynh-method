# SafeGrip-Open v2.2 — SafeGrip-PFR

The active proposal is **SafeGrip-PFR: Physics-Feasible Residual Estimation**.
The previous SafeGrip-FRC, SafeGrip-PNTR, and SafeGrip-CI paths remain in the
repository only for reproducibility.

## Why PFR is the active method

PFR intentionally removes the response-inversion/trust-region stack.  It uses
one GRU and one calibrated projection:

\[
r_t^\star=\mu_t-L_{0,t},
\qquad
\widetilde\mu_t=L_{0,t}+f_\theta(X_t),
\]

\[
L_{\alpha,t}=\max\{0,L_{0,t}-q_\alpha\},
\qquad
\widehat\mu_t=\Pi_{[L_{\alpha,t},U]}(\widetilde\mu_t).
\]

`L0` is the mechanics-derived lower-grip signal.  The GRU learns only the
signed residual above that known structured term.  The scalar `q_alpha` is
estimated from the calibration split with one-sided split conformal
calibration.  The projection contains no learned parameter.

The neural model is trained **only on training labels**.  Calibration labels
are used only for `q_alpha`; test labels are never used for fitting,
calibration, projection, or model selection.

## Main theorem

For a test point whose true friction satisfies

\[
\mu^\star\in[L_\alpha,U],
\]

orthogonal projection onto the feasible interval gives

\[
\boxed{
|\widehat\mu-\mu^\star|^2
\le
|\widetilde\mu-\mu^\star|^2
-
\operatorname{dist}(\widetilde\mu,[L_\alpha,U])^2.
}
\]

Thus the physics projection cannot increase the **actual squared friction
error** on the coverage event, and its guaranteed improvement is at least the
squared distance of the raw estimate to the feasible set.

Under the standard split-conformal exchangeability assumption,

\[
\Pr\{\mu_{new}\ge L_\alpha(X_{new})\}\ge 1-\alpha.
\]

The full interval statement additionally assumes the configured physical upper
bound `U` is valid.

See [`SAFEGRIP_PFR_METHOD.md`](SAFEGRIP_PFR_METHOD.md) for the derivation,
assumptions, theorem, and ablation logic.

## Install and test

```bash
python -m pip install -e ".[paper,dev]"
pytest -q
```

## Quick real-data run

```bash
safegrip download --datasets lira
safegrip prepare --dataset lira
safegrip benchmark --dataset lira --preset quick --proposal pfr --protocol controlled
```

The default proposal is already `pfr`, so `--proposal pfr` may be omitted.

## Paper protocol

```bash
bash scripts/run_paper.sh
```

The paper workflow performs validation-only PFR tuning, baseline tuning, the
five-seed controlled comparison, paired statistics, theorem/fairness audits,
and packaging.

For a separate source-setting comparator table:

```bash
RUN_SOURCE_FAITHFUL=1 bash scripts/run_paper.sh
```

## PFR tuning

Only ordinary neural approximation/optimization parameters are tunable:

```bash
safegrip tune --method pfr --dataset lira --trials 20
```

The following are **not** tuned against validation performance:

- conformal level `alpha`;
- conformal quantile rule;
- mechanics lower-bound formula;
- physical upper bound `mu_upper`;
- projection theorem.

## Main comparison

The primary table contains:

- `safegrip_pfr` — active proposal;
- `direct_gru_control` — same GRU trained directly on friction;
- the registered literature baselines.

The decisive component table contains exactly three estimators:

1. `direct_gru_control`;
2. `pfr_residual_raw` — residual estimator before projection;
3. `safegrip_pfr` — residual estimator after calibrated projection.

This separates two questions cleanly:

- does physics residualization improve learning?
- does the theorem-backed projection improve or preserve the raw residual estimate?

## Primary result files

A PFR benchmark writes:

```text
results/lira_paper/
  metrics.csv
  metrics_by_seed.csv
  predictions_by_seed.csv
  pfr_component_ablation.csv
  pfr_component_ablation_by_seed.csv
  pfr_projection_theorem_audit.csv
  pfr_theorem_audit.json
  pfr_method_audit.json
  fairness_audit.json
  reproducibility_manifest.json
```

`pfr_projection_theorem_audit.csv` evaluates the pointwise theorem identity on
every test endpoint and seed.  `pfr_theorem_audit.json` reports covered-point
violations; this count must be zero up to numerical tolerance.

## Fairness contract

PFR uses:

- train-only input scaling;
- train-only residual target standardization;
- train-only neural optimization;
- calibration-only estimation of the scalar `q_alpha`;
- locked validation/test endpoint IDs across methods;
- the same recurrent architecture for direct-vs-residual comparison;
- no test labels for fitting or calibration;
- no learned response model, search grid, trust radius, or test-time optimizer.

## Retained legacy paths

The following are retained for reproducing earlier experiments, but they are no
longer the active proposal:

```bash
safegrip benchmark --dataset lira --preset paper --proposal frc
safegrip benchmark --dataset lira --preset paper --proposal legacy-ci
```

`src/safegrip/pntr_benchmark.py`, `SAFEGRIP_PNTR_METHOD.md`, and the PNTR unit
tests are also retained as historical material.
