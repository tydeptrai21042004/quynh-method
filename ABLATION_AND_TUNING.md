# SafeGrip-CI v1.4 ablation and hyperparameter protocol

## Primary question

The ablation table must test whether the **counterfactual-energy selective correction** improves a matched temporal estimator and which pieces are necessary. Historical v1.0/v1.1/v1.2 mechanisms are not mixed into the primary table.

## Primary controlled variants

1. `safegrip_base_temporal` — matched direct temporal estimator, no physics correction.
2. `safegrip_no_target_standardization` — removes training-only friction-target standardization.
3. `safegrip_no_physics_residual` — removes counterfactual physics correction.
4. `safegrip_no_utility_gate` — applies the physics correction unconditionally.
5. `safegrip_no_identifiability` — removes entropy-identifiability selector evidence.
6. `safegrip_no_magnitude_head` — uses help probability without a learned correction fraction.
7. `safegrip_no_energy_improvement` — removes candidate-vs-base energy-improvement evidence.
8. `safegrip_no_contrastive_dynamics` — removes friction-discriminative counterfactual dynamics loss.
9. `safegrip_no_do_no_harm` — removes correction-harm regularization.
10. `safegrip_no_bound` — removes the conditional physics projection.
11. `safegrip_no_selector_warmup` — removes the frozen-base controller warm-up stage.
12. `safegrip_no_uq` — identical point path, no predictive UQ layer.
13. `safegrip` — full v1.4 proposal.

All variants use the same selected point-model hyperparameters unless a separately labelled retuned-ablation study is run.

## Validation-only point-model tuning

The main v1.4 search includes temporal capacity/optimizer settings plus:

- sequence length and scaler (`minmax` / `standard`);
- counterfactual grid points, radius and energy temperature;
- physics correction scale/trust-region size;
- base-pretraining duration;
- base and safety loss weights;
- help-head and correction-fraction loss weights;
- do-no-harm weight;
- counterfactual dynamics loss weight/temperature/negative count;
- dynamics-pretraining duration;
- heteroscedastic representation loss.

The primary selection metric is validation RMSE, identical in role to comparator selection. Test labels are never consulted by tuning.

## Sensitivity analysis

After selection, one-factor-at-a-time validation sensitivity focuses on:

`physics_correction_scale`, `energy_grid_points`, `energy_grid_radius`, `energy_temperature`, `base_loss_weight`, `safety_loss_weight`, `benefit_gate_loss_weight`, `correction_fraction_loss_weight`, `do_no_harm_weight`, `counterfactual_loss_weight`, `contrastive_temperature`, and `dynamics_pretrain_epochs`.

## Interpretation rule

A mechanism earns a paper claim only if the locked ablation shows a meaningful and reproducible effect. Near-identical predictions are evidence that the mechanism is inactive, not evidence of robustness. `safegrip_no_uq` is expected to have identical point predictions to the full model.
