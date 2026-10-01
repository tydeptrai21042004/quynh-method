# SafeGrip-PFR-ECR

**SafeGrip-PFR-ECR is the only active proposal in this repository.** The primary benchmark compares its point estimator with four paper-supported road-friction/grip baselines: Du2023 Dynamics-InceptionTime, Todorovic2022 CNN, Lampe2023 GRU, and Levenberg2023 vibration/STFT. Older proposal families are not exposed by the CLI.

The main prediction table contains only the proposal plus those paper baselines. `direct_gru_control` and other component variants are internal ablations. See `SAFEGRIP_PFR_METHOD.md` and `LITERATURE_BASELINES.md`.

The active proposal is **SafeGrip-PFR-ECR: Excitation-Aware Conformal
Risk-Controlled Residual Estimation**. The executable benchmark is real-data-only.

## Active method

PFR-ECR uses one GRU backbone.  It adds the mechanics lower-grip signal to the
input and uses normalized mechanics excitation to weight the recurrent hidden
states:

\[
a_j=\operatorname{softmax}(\gamma L_{0,j}/U),
\qquad
h^{\rm phys}=\sum_j a_jh_j.
\]

Let the trailing mechanics quantity be the anchor

\[
A_0^W=\max_{j\in W}L_{0,j}.
\]

The point estimate is

\[
\mu_{\rm point}=A_0^W+r_\theta(h^{\rm phys}).
\]

`A_0^W` is called an **anchor**, not an unconditional current-friction lower
bound. Interpreting the trailing maximum itself as a deterministic lower bound
requires a local temporal-persistence assumption; PFR-ECR instead calibrates
its violations on the calibration partition.

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

when the relevant calibration/test units satisfy the split-conformal
exchangeability assumption. The paper configuration uses trajectory-level
`group_holdout` and block-max calibration to reduce overlap-induced
pseudo-replication; this is explicitly a dependence-mitigation protocol, not a
claim of validity under arbitrary temporal dependence. If a downloaded public
LiRA artifact exposes fewer than four independent trajectory/trip groups, a
group holdout is impossible; with the explicit
`split.insufficient_group_policy: fallback_spatial_within_trajectory` setting,
preprocessing falls back to the purged within-trajectory split and records both
the requested and effective protocols in its audit metadata. Such a fallback
run must not be described as a group-holdout result.

See [`SAFEGRIP_PFR_METHOD.md`](SAFEGRIP_PFR_METHOD.md) for the exact method,
assumptions, ablations, and theorem.

## Install and test

```bash
python -m pip install -e ".[paper,dev]"
pytest -q
```

## Quick LiRA run

```bash
safegrip download --datasets lira_cd
safegrip prepare --dataset lira_cd
safegrip benchmark --dataset lira_cd --preset quick --protocol controlled
```

## Paper workflow

```bash
bash scripts/run_paper.sh
```

For a separate source-setting comparator table:

```bash
RUN_SOURCE_FAITHFUL=1 bash scripts/run_paper.sh
```

## Main comparison and ablations

The main table contains `safegrip_pfr` (the point estimate) and the four registered paper-supported literature baselines. `direct_gru_control` is retained only as an internal ablation/control.

The PFR component table additionally contains:

- `pfr_residual_raw` — residual learning without explicit PFR physics channels;
- `pfr_physics_features_no_attention` — physics channels with uniform pooling;
- `safegrip_pfr` — full excitation-aware point estimator;
- `safegrip_pfr_safe` — controller-facing safety output, not the RMSE objective.

## Primary result files

```text
results/lira_cd_paper/
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

## Closed real-data registry

The public CLI exposes the four final-protocol real datasets only: `lira_cd`,
`uc3m_tire`, `deep_dynamics_iac`, and `io_vnbd`. The legacy LiRA PFR-ECR
end-to-end benchmark is D1 (`lira_cd`); D2--D4 are kept in their distinct
scientific tasks and are not fabricated into the D1 friction table.

```bash
safegrip datasets
safegrip baselines --dataset lira_cd
safegrip benchmark --dataset lira_cd --preset trust --protocol controlled
```

---

## Universal heterogeneous-sensor research track

A parallel research implementation now lives under `safegrip.universal`, `safegrip.model`, `safegrip.training`, and `safegrip.sensor_io`. It preserves the existing SafeGrip-PFR-ECR benchmark as the reproduction/reference implementation while developing a dataset-independent physically typed sensor-set model.

Run the architecture-only smoke audit with:

```bash
safegrip universal-smoke
```

See `UNIVERSAL_SAFEGRIP_METHOD.md`, `docs/research_specification.md`, and `docs/universal_experiment_protocol.md` before extending it to additional measured datasets. The universal track must not fabricate missing targets or timing merely to pool incompatible sources.

### One-proposal D1--D4 real-data Kaggle experiment

For a cross-dataset comparison, use the single runner
`scripts/KAGGLE_ALL_REAL_DATASETS_ONE_PROPOSAL_1SEED_10EPOCHS.py`.
It trains the **same `UniversalSafeGrip` proposal architecture and training
method on D1--D4**.  Dataset adapters and physical target queries change only
the measurement interface; they do not select a different proposal model.

This runner has a strict real-data/paper-baseline contract:

- it never substitutes synthetic/generated records or targets;
- D3 consumes only the five real IAC racecar CSV logs and excludes Bayesrace
  simulator files;
- D2 evaluates the real U6ICRX slip-angle condition from the three published
  microstrain channels; it does not fabricate unavailable force labels;
- D4 resolves the real synchronized Git-LFS payload (not pointer stubs), uses
  measured vehicle channels, and evaluates GPS-derived real displacement;
- every comparator must be registered with a publication DOI and
  `paper_verified: true`; dummy/placeholder baselines are rejected before
  training starts;
- if a required real archive/target is unavailable, the dataset is marked
  `SKIPPED_REAL_DATA_UNAVAILABLE` instead of being silently replaced.

For repeated-seed evidence, run:

```bash
SAFEGRIP_SEEDS=3101,3102,3103 python scripts/KAGGLE_ALL_REAL_DATASETS_MULTI_SEED.py
```

Each seed has an independent resumable checkpoint and the wrapper writes
`safegrip_multiseed_summary.csv` with mean/std/count for MAE, RMSE and R2.
