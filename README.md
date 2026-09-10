# SafeGrip-Open v0.8.0

## SafeGrip-CI (v0.8.0)

SafeGrip-CI redesigns the proposal around **counterfactual friction identifiability** rather than handcrafted excitation. The full estimator maintains a persistent friction state, predicts a raw-sensor candidate innovation, and accepts that innovation only when a friction-conditioned dynamics model indicates that nearby friction hypotheses are distinguishable and the candidate explains the observed dynamics better than the prior.

The full point estimator uses raw sensors; the old handcrafted excitation score is retained only as an explicit comparator ablation. The mechanics lower endpoint remains a conditional feasibility constraint, and post-hoc block-conformal UQ remains separate from the novelty claim. See `SAFEGRIP_CI_METHOD.md`.

The literature baselines, LiRA leakage safeguards and paper-readiness gates remain independent of proposal-only changes.

## Final research-release safeguards (v0.5.0)

v0.5.0 keeps the LiRA Table-2 CAN decoding introduced in v0.4.0 and adds the safeguards required for a defensible paper run:

- mandatory `speed/ax/ay` signals fail loudly instead of being replaced by zeros;
- reference traces and large temporal gaps are separated into `trajectory_id` / `segment_id`, so splitting, imputation, resampling and temporal windows cannot bridge discontinuous road pieces;
- the physics lower endpoint is the trailing-window maximum used by the partial-identification theorem (`physics.window_samples`);
- calibration labels are never used in gradient training; calibration only changes the inference-time admissible lower endpoint;
- benchmark outputs include a SHA-256 reproducibility manifest and train-only constant sanity baselines;
- `scripts/check_paper_readiness.py` verifies the scientific health gate, five-seed coverage, tuning records, preprocessing/physics audits and complete ablations;
- `scripts/package_paper_results.py` refuses to create a paper-release ZIP unless every readiness check passes.

`PASS` / `PAPER_READY` means the automatic implementation and degeneracy checks passed. It does **not** guarantee novelty, statistical significance, or journal acceptance; manuscript claims must match the observed multi-seed results.



## Scientific-validity safeguard (v0.4.0)

The LiRA platoon TXT files contain several CAN signals that require the LiRA-CD
Table-2 offset/resolution translation before they are physical units. v0.4.0
applies those source-documented translations, audits signal plausibility, and
refuses to let conformal calibration hide an invalid mechanics bound. See
`SCIENTIFIC_VALIDITY_FIX.md`.

For a meaningful Kaggle validation run (3 seeds, 40-epoch practical cap, full
literature architectures for the selected baselines):

```bash
bash scripts/run_kaggle_trust.sh
```

Inspect `results/lira_trust/result_health.json`. `PASS` means automatic
degeneracy/sanity gates passed; `REVIEW` means the run completed but should not
be used as supporting paper evidence yet.

Reproducible research code for **counterfactual-identifiability friction-state estimation with conditional physics constraints**.

The repository follows two strict rules:

1. **Direct baselines must correspond to published tire/friction-estimation papers.** Generic ML models are not inserted into the paper table simply because they are easy to code.
2. **Each open dataset is used only for a target it actually measures.** LiRA's VIAFRIK value is treated as an external standardized road-friction reference, not silently relabeled as the Renault Zoe tire's exact peak friction.

## One-command paper run

```bash
bash scripts/run_paper.sh
```

The paper script:

1. installs paper dependencies;
2. auto-downloads LiRA through the public Figshare API;
3. groups the official asynchronous `task_7505_*` LiRA sensor files by task, synchronizes CAN/GPS streams, then aligns the assembled task to VIAFRIK traces with distance/heading/monotonic constraints before leakage-safe spatial splitting;
4. tunes **SafeGrip and every literature comparator** on train/calibration/validation with the same trial budget (test is locked);
5. keeps recoverable source architecture/preprocessing constraints while allowing model-specific temporal context;
6. runs the final paper table over five independent seeds and reports mean/std;
7. exports supplementary same-physics-projection parity controls without relabelling them as published methods;
8. runs the controlled five-seed SafeGrip ablation and exports provenance/tuning records;
9. exposes excitation, physical-assumption robustness, data-scarcity, cross-route and force-mechanics validation commands.

Use fewer tuning trials during development:

```bash
TRIALS=5 TUNE_EPOCHS=8 bash scripts/run_paper.sh
```

For the smallest real-data Kaggle development run (SafeGrip + Todorovic CNN + Lampe GRU + the full SafeGrip ablation), use:

```bash
bash scripts/run_kaggle_small.sh
```

This uses `configs/kaggle_small.yaml` (10 Hz, 16-sample proposal context, stride 16, one seed, three quick epochs) only as a plumbing/sanity run; it is not a final paper configuration.

Skip tuning and use `configs/default.yaml`:

```bash
SKIP_TUNING=1 bash scripts/run_paper.sh
```

Download/prepare every supported auxiliary source too:

```bash
DOWNLOAD_AUX=1 bash scripts/run_paper.sh
# or independently
bash scripts/download_all_datasets.sh
```

## Literature-backed direct baselines

The `paper` preset contains only methods tied to actual tire/friction papers:

| ID | Paper method | DOI | Reproduction level |
|---|---|---|---|
| `todorovic2022_cnn` | temporal CNN friction-potential estimator | `10.1088/1742-6596/2234/1/012005` | architecture-faithful adaptation (100 samples; Conv 128/128/256; Dense 400) |
| `lampe2023_lstm` | two-layer LSTM estimator | `10.1016/j.ifacol.2023.12.056` | architecture/preprocessing-faithful adaptation; source training defaults included |
| `lampe2023_gru` | two-layer GRU estimator | `10.1016/j.ifacol.2023.12.056` | architecture/preprocessing-faithful adaptation; source training defaults included |
| `schaefke2023_transformer` | onboard-sensor Transformer | `10.1109/CDC49753.2023.10384175` | explicitly methodology-level adapted implementation |
| `chen2025_svdkl` | spatio-temporal + stochastic variational deep-kernel learning | `10.1109/TIE.2024.3440510` | explicitly adapted SV-DKL; source category-selection stage is not claimed reproduced |

The original papers do not all expose the same sensors or public training data. Therefore the code explicitly exports `fidelity` in `baseline_manifest.csv`; it does **not** claim exact reproduction where that would be false.

Paper-reported scores from non-matching protocols are stored as literature context only and are never merged into our direct numerical table.

## Proposal and ablations

Full SafeGrip-CI:

```text
raw sensor window
   -> context prior encoder
   -> candidate innovation encoder
   -> friction-conditioned dynamics model G(context, mu)

previous friction state + context prior -> persistent prior mu^-
G(mu^- +/- delta) ---------------------> local identifiability I
G(mu^-), G(mu_candidate) --------------> candidate acceptance A
K = A * I
q = q_prior + K * innovation
mu = lower + (mu_upper-lower) * sigmoid(q)

frozen selected point estimator
   -> residual-scale head
   -> low-identifiability uncertainty inflation
   -> block-max split-conformal calibration
   -> physical interval intersection
```

Primary controlled ablations:

| Variant | Question |
|---|---|
| `safegrip_backbone_raw` | does CI beat plain raw temporal regression? |
| `safegrip_persistent` | is persistent state alone useful? |
| `safegrip_neural_innovation` | does unconditional neural innovation help? |
| `safegrip_no_identifiability` | is local counterfactual identifiability necessary? |
| `safegrip_excitation_proxy` | is counterfactual identifiability better than handcrafted excitation? |
| `safegrip_no_acceptance` | does candidate-consistency acceptance matter? |
| `safegrip_no_innovation_supervision` | does direct innovation-direction supervision matter? |
| `safegrip_no_bound` | what does the mechanics lower endpoint contribute? |
| `safegrip_no_uq` | point estimator without post-hoc UQ |
| `safegrip` | full SafeGrip-CI |

The ablation registry is explicit and unit-tested so primary variants cannot silently resolve to identical behavior. See `SAFEGRIP_CI_METHOD.md` and `ABLATION_AND_TUNING.md`.

## Fair validation tuning

Proposal:

```bash
safegrip tune --dataset lira --trials 30 --no-test
```

Literature comparators (same trial budget):

```bash
safegrip tune-baselines --dataset lira --trials 30
```

The primary selection objective for both is **validation RMSE**. Each baseline may use its own temporal context and source-appropriate train-only scaler; a common warm-up makes validation/test endpoints identical across methods.

Tuned proposal parameters:

- sequence length and encoder widths;
- learning rate, weight decay, batch size and Huber transition;
- candidate-innovation window and scale;
- persistent-state blend;
- counterfactual friction displacement and identifiability scale;
- acceptance temperature;
- innovation/dynamics/counterfactual/do-no-harm loss weights;
- identifiability-conditioned uncertainty inflation.

**Not tuned:** `mu_upper` and calibration coverage `alpha`. They are physical/safety assumptions, not validation-score knobs.

Outputs:

- `best_hparams.yaml`
- `trials.csv`
- `param_importance.json`
- `tuning_summary.json`
- Optuna SQLite study

The test partition is locked during the search. A test result is produced only after a configuration has been selected, unless `--no-test` is used (the paper script uses `--no-test` and then evaluates the selected setup in the common benchmark).

## Supported open data

Eight sources are registered and auto-discover/download where upstream public access permits it:

1. **LiRA-CD platoon friction test** — primary road-friction-reference benchmark.
2. **KU Leuven LMSD Concept Car** — real wheel-force validation using Kistler RoaDyn WFTs.
3. **KIT tire force-transmission dataset** — measured dry-asphalt tire mechanics.
4. **Deep Dynamics / IAC** — high-dynamics auxiliary/domain-shift vehicle data.
5. **comma2k19** — unlabeled CAN/IMU temporal/domain data; safe default downloads the repository/example, not the ~100 GB full archive.
6. **Extreme Road Image Dataset** — six road-condition image classes for optional multimodal work.
7. **Bicycle Tyre Data, Zenodo** — open lateral-force/self-aligning-torque mechanics data; auxiliary only.
8. **Mendeley tire-pavement friction data** — friction coefficient across road/speed conditions.

```bash
safegrip datasets
safegrip download --datasets all
safegrip prepare --dataset kit
```

See `DATASETS.md` for targets, licenses and scientific roles.

### Access-policy behavior

The downloader never bypasses repository restrictions. For example, if KU Leuven requires a guestbook/terms acceptance, accept it on the dataset page and, if Dataverse requires it, provide:

```bash
export KULEUVEN_API_TOKEN="..."
```

The rest of the datasets continue even if one upstream host changes its API.

## Leakage-safe LiRA protocol

Adjacent GPS samples share road condition and must not be randomly scattered between train/test. The corrected preprocessor performs per-trip alignment and split construction before any feature filling. Default blocks are:

- 60% train;
- 10% lower-bound calibration;
- 10% validation;
- 20% test.

The platoon-test vehicle channels are distributed as separate asynchronous `task_7505_*` TXT streams. SafeGrip now synchronizes these streams first, interpolates the low-rate GPS signal onto the common task timeline with a bounded time gap, and only then performs reference matching. Candidate VIAFRIK traces/directions are aligned independently using a configurable metric tolerance (10 m by default), heading consistency and monotonic reference progress; the nearest valid trace match is retained per vehicle timestamp. Interpolation and resampling after split assignment are restricted to one `(trip_id, split)` block. Temporal windows are also built per trip, so a sequence can never bridge two independent drives. GPS/route/matching metadata are excluded from model features.

The preprocessor writes `lira_stream_assembly_report.json`, `lira_alignment_report.csv` and `lira_preprocessing_report.json` so raw-stream resolution, retained matches and protocol settings are auditable. See `KAGGLE_LIRA_FIX.md` for the regression that fixed the original Kaggle preparation failure.

## Reviewer-oriented experiments

After the main benchmark:

```bash
safegrip experiment --dataset lira --study excitation --results results/lira_paper
safegrip experiment --dataset lira --study robustness --results results/lira_paper
safegrip experiment --dataset lira --study scarcity --preset paper \
  --proposal-hparams results/lira_tuning/best_hparams.yaml
safegrip experiment --dataset lira --study cross-route --preset paper \
  --proposal-hparams results/lira_tuning/best_hparams.yaml

safegrip force-validate --dataset kit
safegrip force-validate --dataset kuleuven
```

`excitation` and `robustness` use the frozen paper predictions. `scarcity` retrains SafeGrip and the data-only prior/evidence backbone over 10/25/50/75/100% training fractions. `cross-route` only runs when at least two explicit route IDs are available; it does not infer route names from GPS.

## Quick/offline verification

```bash
bash scripts/smoke_test.sh
```

or:

```bash
DATASET=synthetic bash scripts/run_all.sh
```

Synthetic data exists only to test software plumbing. It is **never** a paper baseline or evidence for the research claim.

## Key outputs

Main benchmark:

```text
results/lira_paper/
  baseline_manifest.csv
  literature_only.json
  metrics.csv                 # mean across final seeds
  metrics_by_seed.csv         # individual final runs
  predictions.csv              # includes SafeGrip raw mean, sigma and projected 95% interval
  features.json
  calibration.json
  proposal_hparams.json
  baseline_selected_hparams.json
  evaluation_protocol.json
  projection_control_metrics.csv
  projection_control_metrics_by_seed.csv
  *.png
```

Ablation:

```text
results/lira_ablation_paper/
  ablation_metrics.csv          # mean/std across final seeds
  ablation_metrics_by_seed.csv
  ablation_predictions.csv
  ablation_design.json
```

Tuning:

```text
results/lira_tuning/
  best_hparams.yaml
  trials.csv
  param_importance.json
  tuning_summary.json
  optuna.sqlite3

results/lira_baseline_tuning/
  best_hparams.yaml
  tuning_summary.json
  <baseline-id>/trials.csv
  <baseline-id>/best_hparams.yaml
  <baseline-id>/optuna.sqlite3
```

## Documentation

- `SAFEGRIP_CI_METHOD.md` — active v0.8 counterfactual-identifiability formulation.
- `RESEARCH_PROTOCOL.md` — exact claim boundaries and evaluation protocol.
- `LITERATURE_BASELINES.md` — why every direct baseline is included and what is *not* directly comparable.
- `DATASETS.md` — open-data inventory and auto-download behavior.
- `ABLATION_AND_TUNING.md` — controlled ablation and hyperparameter protocol.
- `references.bib` — citations used by the benchmark registry.
- `VALIDATION.md` — commands executed on the packaged repository and current test status.
- `IMPLEMENTED_IMPROVEMENTS.md` — concise map from the review issues to the implemented code changes.

## 2026-09 fairness/ablation hardening

The current release adds global tuning-endpoint parity, feature-parity and label-budget controls, common conformal-UQ controls, corrected component-isolating ablations, retuned supplementary ablations, a fail-closed fairness audit, and paired-bootstrap statistics. See `FAIRNESS_AND_ABLATION_V2.md` for the protocol and commands.
