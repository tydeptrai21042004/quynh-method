# SafeGrip-Open v0.3.1

Reproducible research code for **physics-constrained partial identification of tire-road friction under limited excitation**.

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

This uses `configs/kaggle_small.yaml` (10 Hz, 32-sample context, stride 16, one seed, three quick epochs) only as a plumbing/sanity run; it is not a final paper configuration.

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

Full SafeGrip:

```text
production-sensor time window
        -> compact TCN
        -> mean + aleatoric scale
        -> soft physics loss
        -> calibrated identified set [mu_lower, mu_upper]
        -> hard projection
```

Controlled ablation:

| Variant | What is removed? |
|---|---|
| `safegrip_data_only` | hard projection + soft physics loss |
| `safegrip_no_projection` | hard identified-set projection |
| `safegrip_no_uq` | heteroscedastic uncertainty head |
| `safegrip_no_physics_loss` | soft physics penalty |
| `safegrip_no_calibration` | one-sided external-reference calibration |
| `safegrip_no_temporal` | TCN temporal encoder; uses last-state MLP |
| `safegrip` | full proposal |

Run separately:

```bash
safegrip ablation --dataset lira --preset paper \
  --proposal-hparams results/lira_tuning/best_hparams.yaml
```

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

Tuned parameters:

- sequence length;
- hidden width;
- number of TCN blocks;
- kernel size;
- dropout;
- learning rate;
- weight decay;
- batch size;
- supervised MSE weight;
- soft physics-loss weight.

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

`excitation` and `robustness` use the frozen paper predictions. `scarcity` retrains SafeGrip and the data-only TCN+UQ backbone over 10/25/50/75/100% training fractions. `cross-route` only runs when at least two explicit route IDs are available; it does not infer route names from GPS.

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

- `RESEARCH_PROTOCOL.md` — exact claim boundaries and evaluation protocol.
- `LITERATURE_BASELINES.md` — why every direct baseline is included and what is *not* directly comparable.
- `DATASETS.md` — open-data inventory and auto-download behavior.
- `ABLATION_AND_TUNING.md` — controlled ablation and hyperparameter protocol.
- `references.bib` — citations used by the benchmark registry.
- `VALIDATION.md` — commands executed on the packaged repository and current test status.
- `IMPLEMENTED_IMPROVEMENTS.md` — concise map from the review issues to the implemented code changes.
