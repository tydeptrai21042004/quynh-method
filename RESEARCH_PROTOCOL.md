# SafeGrip research protocol — v0.9.0

## Primary question

Can a persistent friction-state estimator improve reliability by separating three jobs that standard end-to-end regression usually entangles: candidate friction correction, local friction identifiability, and counterfactual rejection of harmful updates?

## Claim boundary

SafeGrip-CI is not presented as a complete tire model. The mechanics lower endpoint remains a conditional feasibility constraint. Counterfactual identifiability and inverse-dynamics agreement are learned/model-based quantities and must be validated empirically through ablation, held-out routes and external data where possible.

## Current point estimator

The full estimator is described in `SAFEGRIP_CI_METHOD.md`:

1. persistent prior friction state with segment reset;
2. independently supervised raw-sensor candidate innovation;
3. friction-conditioned causal dynamics model;
4. local counterfactual identifiability;
5. normalized asymmetric counterfactual veto;
6. detached local inverse-dynamics agreement objective;
7. bounded update inside `[L,U]`.

The full proposal does not consume the handcrafted SafeGrip excitation feature.

## Training

Training labels are restricted to the training split. Lower-bound calibration and UQ calibration remain disjoint. The dynamics model is warm-started, followed by joint optimization using point, raw-innovation, candidate, direction, inverse-dynamics-agreement, dynamics, counterfactual-ranking and do-no-harm terms. Test labels are never used for selection or calibration.

## Physics lower bound

The conditional vector force-balance lower endpoint is retained. It constrains the final point estimate by parameterization; the paper should not claim it improves RMSE unless the ablation supports that claim.

## LiRA target boundary

LiRA/VIAFRIK provides a standardized road-friction reference aligned to the vehicle trajectory. It should not be described as direct instantaneous tire `mu_max` measurement from the passenger vehicle.

## Required primary ablations

1. raw temporal backbone;
2. persistent state only;
3. persistent state + unconditional neural innovation;
4. no counterfactual identifiability;
5. handcrafted excitation proxy instead of identifiability;
6. no asymmetric veto (identifiability-only authority);
7. no inverse-dynamics agreement objective;
8. no direct innovation supervision;
9. no mechanics lower bound;
10. no UQ;
11. full SafeGrip-CI.

## Selection and reporting

Validation RMSE remains the point-estimator selection metric. Report MAE, RMSE, R2, unsafe overestimation, per-seed variability, paired hierarchical bootstrap comparisons, prior/candidate/final RMSE, candidate- and accepted-update direction correlations, mean identifiability, mean trust/veto, inverse-dynamics agreement, update magnitude, coverage and interval width. Test endpoint and independent segment counts must remain visible in the health report.
