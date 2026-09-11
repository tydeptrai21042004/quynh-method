# SafeGrip research protocol — v1.2.0

## Primary question

Can a strong raw-sensor temporal friction estimator be made **safer without sacrificing accuracy** by selectively applying a bounded inverse-dynamics residual only when inference-time evidence predicts that the correction is useful?

## Claim boundary

SafeGrip-CI v1.2 is a hybrid estimator, not a complete tire model. Counterfactual sensitivity is interpreted as local friction observability, not as correctness. The inverse-dynamics branch is a local residual correction, not an independent ground-truth physics estimator. Physical projection is a safety constraint and must not be credited with RMSE improvement unless controlled ablations support that statement.

No v1.2 superiority claim is valid until a new locked real-LiRA TRUST/PAPER run is completed. Historical v1.0/v1.1 results only motivate the redesign.

## Full point estimator

The full estimator is specified in `SAFEGRIP_CI_V12_METHOD.md`:

1. Conv1D + multi-layer GRU temporal representation from raw causal sensor windows;
2. direct neural friction estimate;
3. separate friction-conditioned causal dynamics model;
4. single-scale local counterfactual observability;
5. damped, trust-region-bounded inverse-dynamics residual correction;
6. learned correction-utility gate using inference-available evidence only;
7. explicit friction-change and regime-change supervision;
8. asymmetric unsafe-overestimation loss;
9. physical feasible-set projection;
10. heteroscedastic evidence plus disjoint block-max split-conformal UQ.

The full proposal does not consume the handcrafted SafeGrip excitation proxy and does not recursively feed the previous friction prediction into the point estimator.

## Leakage and split roles

Training labels are used only on training endpoints. Hyperparameter selection is validation-only. Lower-bound calibration and UQ calibration remain disjoint roles. Test labels are used only for final evaluation/statistics. Tuning manifests must prove proposal/baseline validation-endpoint parity.

## Dynamics branch separation

The dynamics model is trained with true training friction labels and a counterfactual ranking loss. Local physics evidence is detached before entering the point correction/gate path, preventing point loss from improving by distorting the dynamics model. The utility-gate target is constructed only from training labels; inference receives no friction label.

## Required primary ablations

1. `safegrip_base_temporal`;
2. `safegrip_no_dynamic_loss`;
3. `safegrip_no_safety_loss`;
4. `safegrip_no_physics_residual`;
5. `safegrip_no_utility_gate`;
6. `safegrip_no_identifiability`;
7. `safegrip_no_bound`;
8. `safegrip_no_regime_head`;
9. `safegrip_no_heteroscedastic`;
10. `safegrip_no_uq`;
11. `safegrip`.

The primary table uses frozen full-model hyperparameters. Optional retuned ablations are supplementary only.

## Selection and reporting

Validation RMSE remains the point-estimator selection metric. Final reporting must include MAE, RMSE, R2, unsafe-overestimation mean/rate, per-seed variability, paired hierarchical bootstrap comparisons, base/final/full-physics-candidate RMSE, correction-vs-needed correlation, utility-gate diagnostics, observability, change probability, aleatoric evidence, physical projection rates, PICP and MPIW.

A strong paper claim should require positive R2 and a competitive accuracy-safety trade-off against the strongest literature comparator, not merely a lower unsafe rate with substantially worse point accuracy.
