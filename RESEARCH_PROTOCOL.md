# SafeGrip research protocol

## Primary question

Under limited tire excitation, can road/tire grip information be represented more honestly as a **partially identified physically admissible set** and then combined with a compact learned estimator, rather than forcing an overconfident unconstrained point prediction?

## Mathematical object

For tangential tire force `F_t=[Fx,Fy]` and normal load `Fz`, minimal friction-cone physics implies

`rho = ||F_t||_2/Fz <= mu`.

Over a time window `W`, a minimal model gives an admissible set

`Theta_W = [max_t rho_t, mu_U]`.

With bounded force/load error, SafeGrip computes a robust relaxed lower bound. For production-sensor LiRA data, the implementation uses a conservative whole-vehicle analogue and then performs a one-sided calibration on the dedicated calibration block because the target is an external VIAFRIK reference rather than the same ego tire.

## Claim boundary for LiRA

Never write that VIAFRIK is an exact measurement of the Renault Zoe tire's instantaneous `mu_max`.

Use LiRA for:

- standardized road-friction-reference estimation;
- low-cost production-sensor usefulness;
- calibration/safety behavior under normal driving;
- spatially held-out road generalization.

Use KIT/KU Leuven for force/mechanics validation.

## Split and leakage control

No random row split of adjacent route samples.

Default ordered/spatial blocks:

- 60% train
- 10% lower-bound calibration
- 10% validation
- 20% final test

GPS may be used for alignment and split construction only. It is excluded from learned input features.

## Direct baselines

Only paper-backed friction estimators are admitted to the direct paper table:

- Todorovic et al. 2022 CNN (`10.1088/1742-6596/2234/1/012005`)
- Lampe et al. 2023 LSTM (`10.1016/j.ifacol.2023.12.056`)
- Lampe et al. 2023 GRU (same paper)
- Schäfke et al. 2023 Transformer (`10.1109/CDC49753.2023.10384175`)
- Chen et al. 2025 SV-DKL uncertainty method (`10.1109/TIE.2024.3440510`)

Where original private sensors/protocols differ, the result must be called a common-open-protocol reimplementation, not exact reproduction.

Generic Ridge/RandomForest/XGBoost/MLP baselines are intentionally excluded from the paper table.

## Proposal ablation

Required variants:

1. data-only TCN+UQ;
2. no hard projection;
3. no uncertainty head;
4. no soft physics loss;
5. no external-reference calibration;
6. no temporal encoder;
7. full SafeGrip.

## Hyperparameter selection

Tune the proposal only on train/calibration/validation. Never query test metrics from the Optuna objective.

Search network/training parameters only. `mu_upper` and `alpha` stay fixed by protocol.

After selecting the proposal configuration, use the selected sequence length as the common evidence window for the direct baseline table and reuse the same proposal hyperparameters across all ablations.

## Required metrics

Point prediction:

- MAE
- RMSE
- R2

Safety/physics:

- positive overestimation magnitude
- unsafe overestimation rate above +0.05
- physical lower-bound violation rate
- identified-set width
- point outside physical set rate

Probabilistic:

- Gaussian NLL
- 95% PICP
- 95% mean prediction interval width

## Recommended paper experiments

1. Main direct LiRA table with cited baselines only.
2. Full proposal ablation.
3. Hyperparameter study/importance with test locked.
4. Excitation-stratified performance and identified-set width.
5. Training-data scarcity sweep.
6. Sensor noise/bias, mass and effective-radius mismatch.
7. KU Leuven wheel-force auxiliary validation.
8. KIT tire-force/utilization validation.
9. External road/speed sanity check with Mendeley friction data.
10. Downstream conservative braking/planning demonstration.
