# SafeGrip-Open v2.1 — SafeGrip-PNTR

The active proposal is **SafeGrip-PNTR: Physics-Neural Trust-Region Friction Estimation**. The previous theorem-driven SafeGrip-FRC and SafeGrip-CI v1.4 paths are retained for reproducibility.

## Active method

SafeGrip-PNTR is built around a physics-residual neural estimate plus explicit physical evidence rather than global response inversion. A target-standardized GRU predicts the non-negative friction slack above the vehicle-mechanics lower bound, so the neural anchor is physically admissible by construction; a friction-conditioned causal response model may then make only a bounded local correction. The response model is trained with both response MSE and a friction-discriminative ranking loss so it cannot obtain a low training loss simply by ignoring the hypothetical friction input.

For horizon $H$, PNTR minimizes locally

\[
F_{t,H}(\mu)=E_{t,H}(\mu)+\lambda_A\left(\frac{\mu-\mu_t^0}{\tau}\right)^2,
\]

inside the intersection of the mechanics interval and the neural trust region. The exact neural anchor is always the fallback. Hence an accepted physics correction is bounded by $\tau$ and cannot have a larger regularized objective than the anchor.

See `SAFEGRIP_PNTR_METHOD.md` for the exact estimator, safeguards, conditional local-recovery result, and ablations.

### Retained methods

- `--proposal frc`: previous global finite-window response-inversion/certification method.
- `--proposal legacy-ci`: previous SafeGrip-CI v1.4 method.

## Primary paper run

```bash
bash scripts/run_paper.sh
```

The controlled protocol is the primary comparison. It gives literature comparators a common training budget while preserving their registered architecture/preprocessing family. A source-setting comparison can be run separately:

```bash
RUN_SOURCE_FAITHFUL=1 bash scripts/run_paper.sh
```

Development run:

```bash
TRIALS=5 TUNE_EPOCHS=8 bash scripts/run_paper.sh
```

## Direct commands

Tune only ordinary FRC training parameters; the theorem, candidate grid, certificate deltas, and horizons are not tuned:

```bash
# PNTR currently uses the fixed declared method hyperparameters in configs/default.yaml.
# The retained FRC path can still be tuned with:
safegrip tune --method frc --dataset lira --trials 30
```

Tune literature comparators on the same validation endpoint contract and fixed tuning seeds:

```bash
safegrip tune-baselines --dataset lira --protocol controlled --trials 30
```

Run the primary benchmark:

```bash
safegrip benchmark --dataset lira --preset paper \
  --proposal pntr --protocol controlled \
  --baseline-hparams results/lira_baseline_tuning/best_hparams.yaml
```

Run the retained FRC proposal:

```bash
safegrip benchmark --dataset lira --preset paper --proposal frc
```

Run the retained old CI proposal:

```bash
safegrip benchmark --dataset lira --preset paper --proposal legacy-ci
```

## Primary comparison design

The primary table contains:

- `safegrip_pntr` — active physics-neural trust-region proposal;
- `direct_gru_control` and `pntr_physics_residual_anchor` are exported as same-encoder/component controls rather than literature comparators;
- `todorovic2022_cnn`;
- `lampe2023_lstm`;
- `lampe2023_gru`;
- `schaefke2023_transformer`;
- `chen2025_svdkl`.

The literature implementations are adaptations to the common LiRA task. Their fidelity level is explicitly recorded in `src/safegrip/literature.py`; the code does not claim exact reproduction when sensors/data/protocol differ from the source study.

The mechanics constraint is an explicit part of PNTR, not hidden post-processing. Its effect is isolated by comparing the same-encoder `direct_gru_control` against `pntr_physics_residual_anchor`, which predicts only the non-negative friction slack above the mechanics lower bound before the response-based refinement.

## Fairness safeguards

- train-only input scaling;
- train-only response normalization;
- strictly causal response contexts;
- locked validation/test endpoint IDs across methods;
- fixed common tuning seeds instead of `seed + trial.number`;
- equal trial budget for controlled comparison;
- common controlled training budget;
- test labels never used for tuning or horizon selection;
- the mechanics projection is part of the declared PNTR estimator and is isolated by a dedicated ablation;
- source-setting literature comparison reported separately.

The legacy v1.4 tuner was also corrected so inactive `conv_channels`, inactive `huber_beta`, and fixed zero-valued losses no longer consume Optuna dimensions.

## Split protocols

Default:

```yaml
split:
  mode: spatial_within_trajectory
```

This preserves contiguous per-trajectory train/calibration/validation/test blocks with purge gaps.

For the stronger distribution-shift experiment, prepare data with:

```yaml
split:
  mode: group_holdout
```

Whole trajectories are then assigned to one partition only. At least four trajectory/trip groups are required.

## Main PNTR outputs

A paper benchmark writes:

```text
results/lira_paper/
  metrics.csv
  metrics_by_seed.csv
  predictions_by_seed.csv
  pntr_component_ablation.csv
  pntr_horizon_ablation.csv
  pntr_method_audit.json
  fairness_audit.json
  reproducibility_manifest.json
```

`pntr_method_audit.json` records the trust radius, anchor regularization, exact-fallback safeguard, and the limits of the true-error claim. The retained FRC path still produces its original theorem audit.

## Open-data policy

The existing data registry/download/preparation code is retained. The primary friction benchmark uses LiRA's available road-friction reference under the repository's alignment and leakage safeguards. Auxiliary datasets remain auxiliary unless they provide the exact target required by an experiment.

## Tests

```bash
pytest -q
```

In addition to the existing suite, `tests/test_pntr.py` checks trust-region boundedness, exact neural fallback, regularized-objective safety, local identifiability, and the conditional strong-convexity error bound. The retained FRC theorem tests remain in `tests/test_friction_resolution.py`.

## Legacy SafeGrip-CI

`SAFEGRIP_CI_V14_METHOD.md` and the v1.x files are retained so old experiments remain reproducible. They are not the active proposal and are not run by the new paper script.
