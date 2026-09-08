# SafeGrip research protocol

## Primary question

Under limited tire excitation, can tire/road grip be represented more honestly as a **partially identified physically admissible set** and then combined with a learned temporal estimator, rather than forcing an overconfident unconstrained point prediction?

## Mathematical claim boundary

SafeGrip separates three layers that must not be conflated:

1. **Conditional mechanics lower bound.** Under explicitly bounded sensing/model error and an assumed upper admissible coefficient `mu_upper`, the production-sensor mechanics layer returns a conservative lower endpoint `L_phys`.
2. **Statistical calibration.** A dedicated calibration split computes a one-sided relaxation `q`, giving `L_cal = max(0, L_phys - q)`. Coverage relies on the usual split-conformal calibration assumptions; it is not a deterministic physical guarantee under arbitrary spatial shift.
3. **Learned estimator and projection.** The TCN predicts a mean and aleatoric scale. The final point estimate is projected onto `[L_cal, mu_upper]`, and the Gaussian interval is optionally projected endpoint-wise onto the same set.

Use the phrase **physics-constrained partial identification** in the paper. A guarantee should always be written conditionally on the stated bounded-error assumptions and on `mu* <= mu_upper`.

## Core propositions implemented by the code

For a closed interval `Theta=[L,U]` containing the true value `mu*`, Euclidean projection satisfies

`|Pi_Theta(z)-mu*| <= |z-mu*|`.

For a predictive interval `C=[a,b]`, endpoint projection gives `Pi_Theta(C)` with width no larger than `C`; if `mu*` is in both `C` and `Theta`, projection preserves inclusion of `mu*`.

For nested observation windows, the identified lower endpoint based on a maximum utilization is non-decreasing as observations are added, so the admissible set cannot widen solely because more observations are included.

## LiRA target boundary

Never write that VIAFRIK is an exact measurement of the Renault Zoe tire's instantaneous peak `mu_max`.

Use LiRA for:

- standardized road-friction-reference estimation;
- production-sensor usefulness;
- calibrated admissible-set behavior under normal driving;
- spatial/cross-route generalization when route metadata permit it.

Use KIT/KU Leuven for force/mechanics validation.

## Leakage-safe LiRA preprocessing

The corrected pipeline is:

```text
vehicle/ref files
 -> group official task_<id>_<sensor>.txt files by task ID
 -> preserve common timestamps and synchronize asynchronous CAN streams
 -> interpolate low-rate GPS onto the common vehicle timeline
 -> explicit route/direction metadata extraction when present
 -> align each candidate VIAFRIK trace independently
      * maximum metric distance
      * heading consistency
      * monotone reference progress
      * nearest valid trace kept per vehicle timestamp
 -> contiguous spatial split inside each task/trip
 -> interpolation only inside (trip, split)
 -> optional fixed-rate resampling only inside (trip, split)
 -> recompute physics lower bound
 -> trip-safe temporal windows
```

Default matching protocol in `configs/default.yaml`:

- maximum GPS match distance: 10 m;
- heading tolerance: 45 degrees;
- up to 8 nearest reference candidates;
- monotonic reference progress enabled;
- 20 Hz task-stream synchronization/resampling when timestamps are usable;
- maximum 0.50 s nearest-sensor gap and 2.50 s GPS interpolation gap by default;
- no model feature may contain GPS, route ID, split position or matching metadata.

If route/direction are not explicit in a path, the parser records `unknown`; it does not invent route labels.

### Split and sequence rules

Default contiguous blocks inside each trip:

- 60% train;
- 10% lower-bound calibration;
- 10% validation;
- 20% final test.

Optional purge samples can be configured around split boundaries. Even with zero purge, windows and feature filling cannot cross a trip or split boundary because they are grouped explicitly in code.

Different models may use different temporal context lengths, but a common warm-up and stable `sample_uid` endpoint IDs make validation/test endpoints exactly identical across methods. The benchmark aborts if endpoint IDs differ.

## Direct baselines

Only paper-backed friction estimators are admitted to the direct paper table:

- Todorovic et al. 2022 CNN (`10.1088/1742-6596/2234/1/012005`)
- Lampe et al. 2023 LSTM (`10.1016/j.ifacol.2023.12.056`)
- Lampe et al. 2023 GRU (same paper)
- Schäfke et al. 2023 Transformer (`10.1109/CDC49753.2023.10384175`)
- Chen et al. 2025 SV-DKL uncertainty method (`10.1109/TIE.2024.3440510`)

Where private sensors/protocols differ, the result is an **adapted common-sensor reimplementation**, not an exact reproduction. Baseline provenance is exported in `baseline_manifest.csv`.

All methods receive the same train/calibration/validation/test population, the same validation selection metric (RMSE), and the same hyperparameter trial budget. Model-specific history length and source-appropriate train-only scalers are allowed.

## Proposal ablation

Required variants:

1. data-only TCN+UQ;
2. no hard projection;
3. no uncertainty head;
4. no soft physics loss;
5. no external-reference calibration;
6. no temporal encoder;
7. full SafeGrip.

The deterministic no-UQ variant uses exactly one MSE term plus the optional physics penalty; it does not double-weight MSE. Paper-mode ablations use the same five final seeds as the main benchmark and report mean/std.

## Hyperparameter selection

Tune the proposal **and every literature comparator** on train/calibration/validation only. Give each direct comparator the same Optuna trial budget and never query test metrics from a tuning objective.

Use validation RMSE as the primary selection metric. `mu_upper`, `alpha`, GPS matching thresholds and physical uncertainty margins are protocol assumptions, not validation-error knobs.

## Required metrics

Point prediction:

- MAE;
- RMSE;
- R2.

Safety/physics:

- positive overestimation magnitude;
- unsafe overestimation rate above +0.05;
- physical lower-bound violation rate;
- identified-set width;
- point outside physical set rate;
- projection correction rate.

Probabilistic:

- Gaussian NLL;
- raw 95% PICP / MPIW;
- physics-truncated 95% PICP / MPIW for SafeGrip;
- interval-outside-physics-set rate.

## Reviewer-oriented experiments implemented

### Excitation stratification

```bash
safegrip experiment --dataset lira --study excitation \
  --results results/lira_paper
```

Uses `sqrt(ax^2+ay^2)/g` and test-set quantiles. Report RMSE, safety metrics and identified-set width by excitation level.

### Physics-layer robustness

```bash
safegrip experiment --dataset lira --study robustness \
  --results results/lira_paper
```

Freezes the neural predictor and recomputes the physics/calibration/projection layer under mass, acceleration-error and `mu_upper` perturbations. This isolates assumption sensitivity rather than mixing it with retraining variance.

### Training-data scarcity

```bash
safegrip experiment --dataset lira --study scarcity --preset paper \
  --proposal-hparams results/lira_tuning/best_hparams.yaml
```

Compares full SafeGrip against the data-only TCN+UQ backbone at 10/25/50/75/100% of training windows over the final seeds.

### Cross-route holdout

```bash
safegrip experiment --dataset lira --study cross-route --preset paper \
  --proposal-hparams results/lira_tuning/best_hparams.yaml
```

Runs leave-one-explicit-route-out evaluation when at least two route IDs can be recovered from LiRA file metadata. The command fails transparently if route IDs are unavailable rather than fabricating them from GPS.

### Wheel-force mechanics validation

```bash
safegrip force-validate --dataset kit
safegrip force-validate --dataset kuleuven
```

These experiments validate the conservative utilization calculation on measured force channels. They do not relabel force utilization as a direct peak-friction ground truth.
