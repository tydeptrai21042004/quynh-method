# SafeGrip-Open v2.0 — SafeGrip-FRC

The active proposal is **SafeGrip-FRC: Finite-Window Friction Resolution Certification**. The repository retains SafeGrip-CI v1.4 only as a legacy reproducibility path.

## Active method

SafeGrip-FRC contains only one learned component: a causal GRU response model

\[
(c_k,\mu)\mapsto \widehat{\Delta z}_k.
\]

Friction is not produced by a direct proposal head. For each finite observation horizon, the code minimizes the response residual over a declared friction grid, computes a finite-resolution separation margin, and reports a theorem-aligned recovery certificate. The adaptive observation horizon is selected by the smallest finite certificate, not by a learned gate.

See `SAFEGRIP_FRC_METHOD.md` and `THEOREM_VALIDATION.md` for the exact definitions and theorem.

### What was removed from the primary proposal

The active FRC path has no benefit classifier, correction-fraction head, entropy posterior, energy temperature, direct friction base network, learned arbitration, regime head, heteroscedastic head, or primary physical clipping. Those mechanisms remain only in the legacy CI-v1.4 code.

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
safegrip tune --method frc --dataset lira --trials 30
```

Tune literature comparators on the same validation endpoint contract and fixed tuning seeds:

```bash
safegrip tune-baselines --dataset lira --protocol controlled --trials 30
```

Run the primary benchmark:

```bash
safegrip benchmark --dataset lira --preset paper \
  --proposal frc --protocol controlled \
  --proposal-hparams results/lira_frc_tuning/best_hparams.yaml \
  --baseline-hparams results/lira_baseline_tuning/best_hparams.yaml
```

Run the retained old proposal:

```bash
safegrip benchmark --dataset lira --preset paper --proposal legacy-ci
```

## Primary comparison design

The primary table contains:

- `safegrip_frc` — theorem-driven response-inversion proposal;
- `direct_gru_control` — same GRU encoder width/depth with a closely matched direct-regression head;
- `todorovic2022_cnn`;
- `lampe2023_lstm`;
- `lampe2023_gru`;
- `schaefke2023_transformer`;
- `chen2025_svdkl`.

The literature implementations are adaptations to the common LiRA task. Their fidelity level is explicitly recorded in `src/safegrip/literature.py`; the code does not claim exact reproduction when sensors/data/protocol differ from the source study.

Physical projection is **not** used in the primary metrics. A common projection control is exported separately so any gain from post-processing is visible rather than attributed to the proposed estimator.

## Fairness safeguards

- train-only input scaling;
- train-only response normalization;
- strictly causal response contexts;
- locked validation/test endpoint IDs across methods;
- fixed common tuning seeds instead of `seed + trial.number`;
- equal trial budget for controlled comparison;
- common controlled training budget;
- test labels never used for tuning or horizon selection;
- primary prediction contains no physical projection;
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

## Main FRC outputs

A paper benchmark writes:

```text
results/lira_paper/
  metrics.csv
  metrics_by_seed.csv
  predictions_by_seed.csv
  fixed_horizon_ablation.csv
  frc_resolution_certificates.csv
  frc_separation_curves.csv
  frc_certificate_validity.csv
  theorem_audit.json
  fairness_audit.json
  projection_control.csv
  reproducibility_manifest.json
```

`theorem_audit.json` is an executable consistency gate: if the implemented grid estimator violates the stated finite-grid recovery implication on any endpoint where the full theorem premise holds, the benchmark fails.

## Open-data policy

The existing data registry/download/preparation code is retained. The primary friction benchmark uses LiRA's available road-friction reference under the repository's alignment and leakage safeguards. Auxiliary datasets remain auxiliary unless they provide the exact target required by an experiment.

## Tests

```bash
pytest -q
```

In addition to the existing suite, `tests/test_friction_resolution.py` checks closed-form separation on a linear response map, monotonicity of `S(delta)`, grid-certificate construction, horizon fallback, exact-grid inversion, and the finite-grid theorem.

## Legacy SafeGrip-CI

`SAFEGRIP_CI_V14_METHOD.md` and the v1.x files are retained so old experiments remain reproducible. They are not the active proposal and are not run by the new paper script.
