# SafeGrip-CI v1.2 ablation and hyperparameter protocol

## 1. Scientific question

The primary question is whether selective physics correction improves the **accuracy-safety trade-off** of a strong raw-sensor temporal estimator. The ablation table therefore tests one v1.2 mechanism at a time instead of mixing historical v1.0/v1.1 gates.

## 2. Controlled primary ablation

All primary variants use identical data splits, endpoint identities and seeds. In the frozen primary table, every variant reuses the selected full-model hyperparameters unless disabling a component necessarily makes its associated weight irrelevant.

- `safegrip_base_temporal`: direct Conv1D+GRU estimator, no physics residual path.
- `safegrip_no_dynamic_loss`: removes explicit friction-change supervision.
- `safegrip_no_safety_loss`: removes asymmetric unsafe-overestimation training penalty.
- `safegrip_no_physics_residual`: disables inverse-dynamics point correction while retaining the strong base estimator.
- `safegrip_no_utility_gate`: applies the available bounded physics correction without learned selective gating.
- `safegrip_no_identifiability`: removes counterfactual observability from utility-gate evidence.
- `safegrip_no_bound`: removes final physical projection.
- `safegrip_no_regime_head`: removes regime-change supervision/feature.
- `safegrip_no_heteroscedastic`: removes heteroscedastic training evidence.
- `safegrip_no_uq`: identical point model with predictive UQ disabled.
- `safegrip`: full v1.2 proposal.

The decisive comparisons are the full model versus each variant above. The `no_uq` variant is expected to have identical point predictions and is excluded from point-path distinctness checks.

## 3. Frozen versus retuned ablations

The primary ablation must remain frozen after selecting the full proposal. This isolates mechanisms. A separately labelled supplementary retuned-ablation experiment may be used to test whether a removed component simply changes the optimum hyperparameter region; it must not replace the frozen table.

## 4. v1.2 point-model search space

Point-RMSE tuning is validation-only. The test set remains locked.

### Temporal representation

- `sequence_length`: 32, 48, 64, 100
- `hidden`: 64, 96, 128
- `gru_hidden`: 64, 96, 128
- `conv_channels`: 32, 48, 64
- `gru_layers`: 1, 2
- dropout, learning rate, weight decay, batch size and Huber beta

### Physics residual

- `counterfactual_delta`
- `identifiability_lambda`
- `inverse_dynamics_ridge`
- `inverse_dynamics_max_step`
- `physics_correction_scale`
- `dynamics_loss_weight`
- `counterfactual_loss_weight`
- `dynamics_pretrain_epochs`

### Risk/dynamics/gate objectives

- `base_loss_weight`
- `dynamic_loss_weight`
- `safety_loss_weight`
- `unsafe_margin`
- `utility_gate_loss_weight`
- `change_loss_weight`
- `change_threshold`
- `smooth_loss_weight`
- `heteroscedastic_loss_weight`

UQ-only inflation/calibration parameters are not selected by final test RMSE.

## 5. Tuning fairness

For paper mode:

1. proposal and literature baselines receive equal stated search-trial budgets;
2. each tunes only on the locked validation endpoints;
3. endpoint hashes must match across proposal/baseline tuning;
4. final paper metrics use untouched test endpoints and five fixed seeds;
5. hierarchical paired bootstrap resamples matched seeds and trajectory segments rather than treating overlapping windows as independent samples.

## 6. Trust-mode success criteria

TRUST is a scientific screening stage, not a paper claim. At minimum, require the automatic health/fairness/statistics/ablation gates to pass. For a strong final claim, target GRU-level or better RMSE while preserving the proposal's safety advantage, positive R2, non-degenerate prediction variance and calibrated but non-pathological intervals.

No v1.2 superiority claim should be made from v1.1 results or synthetic smoke tests.
