# SafeGrip research protocol — v0.8.0

## Primary question

Can friction-conditioned counterfactual identifiability determine when raw vehicle dynamics contain enough information to update a persistent tire-road friction state more reliably than direct regression or a handcrafted excitation proxy?

## Claim boundary

SafeGrip-CI is not presented as a complete tire model. The mechanics lower endpoint remains a conditional safety/feasibility constraint. Counterfactual identifiability is learned from a friction-conditioned dynamics model and must be validated empirically through ablation and held-out performance.

## Current point estimator

The full estimator is described in `SAFEGRIP_CI_METHOD.md`:

1. persistent prior friction state with segment reset;
2. raw-sensor candidate innovation;
3. friction-conditioned dynamics model;
4. local counterfactual identifiability;
5. candidate consistency acceptance;
6. bounded update inside `[L,U]`.

The full proposal does not consume the handcrafted SafeGrip excitation feature.

## Training

Training labels are restricted to the training split. Lower-bound calibration and UQ calibration remain disjoint. The dynamics model is warm-started on training data, followed by joint optimization using point, innovation, dynamics, counterfactual-ranking and do-no-harm terms. Test labels are never used for selection or calibration.

## Physics lower bound

The existing conditional vector force-balance lower endpoint is retained. It constrains the final point estimate by parameterization; the paper should not claim it improves RMSE unless the ablation supports that claim.

## LiRA target boundary

LiRA/VIAFRIK provides a standardized road-friction reference aligned to the vehicle trajectory. It should not be described as direct instantaneous tire `mu_max` measurement from the passenger vehicle.

## Required primary ablations

1. raw temporal backbone;
2. persistent state only;
3. persistent state + unconditional neural innovation;
4. no counterfactual identifiability;
5. handcrafted excitation proxy instead of identifiability;
6. no candidate-consistency acceptance;
7. no innovation supervision;
8. no mechanics lower bound;
9. no UQ;
10. full SafeGrip-CI.

## Selection and reporting

Validation RMSE remains the point-estimator selection metric. Report MAE, RMSE, R2, unsafe overestimation, per-seed variability, paired bootstrap comparisons, prior/candidate/final RMSE, mean identifiability, mean acceptance, update magnitude, coverage and interval width. The test endpoint count and segment count must remain visible in the health report.
