# SafeGrip research protocol — v0.7.0

## Primary question

Under limited tire excitation, can road-friction reference be estimated more honestly by combining a **physically admissible partially identified set**, a persistent friction prior, and excitation-controlled dynamic evidence, rather than forcing an unconstrained point predictor to react equally to informative and uninformative maneuvers?

## Claim boundary

SafeGrip separates four layers.

1. **Conditional mechanics lower bound.** Under the documented vehicle model and bounded sensing/model-error assumptions, the production-sensor mechanics layer returns a lower endpoint `L_phys`. The guarantee is conditional on those assumptions and on `mu* <= mu_upper`.
2. **Statistical lower-bound relaxation.** A dedicated calibration role computes one-sided correction `q_lower`, giving `L_cal=max(0,L_phys-q_lower)`. This statistical coverage statement is distinct from the mechanics claim.
3. **SafeGrip-v3 point estimator.** A long-context prior is updated by short-context dynamic evidence through a monotone excitation reliability function. The final point estimate is parameterized directly inside `[L,U]`; there is no post-hoc point projection in the full proposal.
4. **Predictive UQ.** After point-model selection, validation residuals fit a positive scale model. A disjoint calibration role computes a block-max split-conformal multiplier. The interval is intersected with the physical support.

Use **physics-constrained partial identification with excitation-aware evidence update** for the proposal. Do not describe VIAFRIK as an exact instantaneous peak-friction measurement of the Renault Zoe tire.

## Current point estimator

For recent label-free excitation score `E_t`,

```text
r_t = sigmoid(softplus(a) * (E_t - sigmoid(tau)))
q_t = q_prior,t + r_t * delta_t
mu_hat,t = L_t + (mu_upper-L_t) * sigmoid(q_t).
```

Hence `L_t <= mu_hat,t <= mu_upper` by construction and `dr_t/dE_t >= 0` for every learned parameter value.

The long branch estimates persistent friction state. The short branch estimates a bounded evidence correction rather than a second absolute friction estimate.

## Training objective

The primary point loss is Huber/SmoothL1. Optional low-weight training-only regularizers use previous endpoints from the same trajectory segment:

- relative friction-change loss;
- pairwise ranking loss for nontrivial target changes;
- weak-excitation smoothness weighted by `(1-E_t)`.

Calibration labels and test labels never enter point-model gradient training.

## Label-free excitation

SafeGrip-only engineered features include acceleration utilization, jerk, relative wheel-speed spread/imbalance, pressure spread and torque utilization. Their instantaneous composite is bounded to `[0,1]` and contains no friction labels.

A causal peak-memory state retains recent excitation and resets at every segment/split boundary. The resulting `sg_excitation_score` remains in physical `[0,1]` scale after feature scaling and is the input to the monotone reliability and uncertainty inflation mechanisms.

## Physics lower bound

The production-sensor lower bound uses the conditional vector force balance

```text
F_x,tire ~= m*a_x + F_drag + F_rr
F_y,tire ~= m*a_y
```

with bounded acceleration-vector and unmodelled-force uncertainty subtracted in norm and a conservative upper normal-load surrogate in the denominator. The implementation deliberately does not hard-clip unit/schema failures before the physics audit.

A fixed trailing window maximum creates the endpoint lower bound used by every method at the same evaluation endpoint.

## LiRA target boundary

Use LiRA for:

- standardized road-friction-reference estimation;
- production-sensor usefulness;
- excitation-stratified behavior;
- calibrated admissible-set behavior;
- spatial/cross-route generalization when route metadata permit it.

Do not claim that VIAFRIK directly measures the vehicle tire's exact instantaneous `mu_max`. Use KIT/KU Leuven force data for mechanics/utilization validation.

## Leakage-safe LiRA preprocessing

The pipeline is:

```text
vehicle/ref files
 -> assemble official task_<id>_<sensor>.txt streams
 -> source-documented CAN translation
 -> synchronize asynchronous sensors and GPS
 -> align candidate VIAFRIK traces with distance/heading/monotonic constraints
 -> create trajectory segments
 -> spatial split before imputation/resampling
 -> impute/resample only inside (segment, split)
 -> recompute mechanics lower bound
 -> construct segment-safe temporal windows
```

GPS, route IDs, split position and matching metadata are never model features.

Different methods may use different context lengths, but a common evaluation warm-up and stable `sample_uid` values enforce identical validation/test endpoints. The benchmark aborts if endpoint targets or IDs differ.

## Direct literature comparators

Only paper-backed friction estimators are admitted to the direct paper table:

- Todorovic et al. 2022 CNN (`10.1088/1742-6596/2234/1/012005`);
- Lampe et al. 2023 LSTM/GRU (`10.1016/j.ifacol.2023.12.056`);
- Schäfke et al. 2023 Transformer (`10.1109/CDC49753.2023.10384175`);
- Chen et al. SV-DKL comparator (`10.1109/TIE.2024.3440510`).

Where sensors/protocols differ, results are explicitly **adapted common-sensor reimplementations**, not exact reproductions. Every direct comparator receives the same validation objective and trial budget while retaining source-appropriate architecture/preprocessing constraints.

## Required SafeGrip-v3 ablations

1. `safegrip_data_only`: no sample-specific lower endpoint and fixed dynamic-evidence reliability;
2. `safegrip_static_only`: no temporal prior/evidence branch;
3. `safegrip_no_gate`: fixed reliability 0.5 instead of monotone excitation reliability;
4. `safegrip_no_bound`: only global `[0,mu_upper]` support;
5. `safegrip_no_uq`: full point estimator without post-hoc UQ;
6. `safegrip_no_calibration`: raw mechanics lower endpoint;
7. `safegrip`: full v3 proposal.

Paper-mode ablations use the same selected proposal hyperparameters and final seed list.

## Selection protocol

Tune the proposal and every literature comparator on train/validation only, with calibration reserved for its predefined roles and test locked until final evaluation. Primary model-selection metric: validation RMSE.

Do not tune `mu_upper`, conformal `alpha`, alignment thresholds or physical uncertainty margins against validation error.

## Required metrics

Point prediction:

- MAE, RMSE, R2;
- prediction standard deviation and train-constant sanity comparators.

Safety/physics:

- positive overestimation magnitude;
- unsafe overestimate rate above +0.05;
- lower-bound violation rate;
- identified-set width;
- point/interval outside physical support;
- physics-bound nonzero/informativeness diagnostics.

Proposal diagnostics:

- prior RMSE;
- final-vs-prior update magnitude;
- excitation/reliability relationship;
- evidence-delta distribution;
- learned reliability slope and threshold.

Predictive UQ:

- raw and physics-intersected PICP/MPIW;
- mean predicted residual scale;
- conformal multiplier and effective block count.

## Reviewer-oriented experiments

### Excitation stratification

```bash
safegrip experiment --dataset lira --study excitation --results results/lira_paper
```

Stratifies final test performance using the same causal label-free excitation score used by SafeGrip-v3.

### Physics robustness

```bash
safegrip experiment --dataset lira --study robustness --results results/lira_paper
```

Freezes learned latent coordinates and recomputes the physical lower endpoint under mass, acceleration-error, external-force, vertical-margin and `mu_upper` scenarios.

### Training-data scarcity

```bash
safegrip experiment --dataset lira --study scarcity --preset paper \
  --proposal-hparams results/lira_tuning/best_hparams.yaml
```

Compares full SafeGrip against the data-only prior/evidence backbone as training data decrease.

### Cross-route holdout

```bash
safegrip experiment --dataset lira --study cross-route --preset paper \
  --proposal-hparams results/lira_tuning/best_hparams.yaml
```

Runs only when explicit route IDs support a legitimate leave-one-route-out study; the code must not fabricate route identity from GPS.

### Wheel-force mechanics validation

```bash
safegrip force-validate --dataset kit
safegrip force-validate --dataset kuleuven
```

These validate force-utilization mechanics. They do not relabel utilization as direct peak-friction ground truth.
