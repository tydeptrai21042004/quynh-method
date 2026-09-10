# SafeGrip-CI v1.0 ablation and hyperparameter protocol

## 1. Controlled primary ablation

All primary variants use identical split definitions, validation/test endpoint identities, seeds, and the **same frozen full-model hyperparameters**. This makes the primary table a component-removal study rather than a collection of separately optimized systems.

The primary set is:

- `safegrip_backbone_raw`
- `safegrip_persistent`
- `safegrip_neural_innovation`
- `safegrip_no_identifiability`
- `safegrip_excitation_proxy`
- `safegrip_no_acceptance`
- `safegrip_no_cf_agreement`
- `safegrip_single_scale_cf`
- `safegrip_no_linearity_consistency`
- `safegrip_no_agreement_veto`
- `safegrip_no_counterfactual_ranking`
- `safegrip_no_innovation_supervision`
- `safegrip_no_bound`
- `safegrip_no_uq`
- `safegrip`

The most important comparisons are:

1. `safegrip_excitation_proxy` vs `safegrip`: learned counterfactual observability vs handcrafted excitation;
2. `safegrip_single_scale_cf` vs `safegrip`: single-point finite difference vs multi-scale counterfactual observability;
3. `safegrip_no_linearity_consistency` vs `safegrip`: magnitude-only sensitivity vs cross-scale-consistent sensitivity;
4. `safegrip_no_acceptance` vs `safegrip`: contribution of the normalized residual veto;
5. `safegrip_no_agreement_veto` vs `safegrip`: contribution of inference-time inverse-dynamics disagreement veto;
6. `safegrip_no_cf_agreement` vs `safegrip`: total contribution of inverse-dynamics agreement (training + inference);
7. `safegrip_no_counterfactual_ranking` vs `safegrip`: whether the dynamics model actually needs to discriminate friction hypotheses;
8. `safegrip_no_innovation_supervision` vs `safegrip`: contribution of separating candidate learning from authority.

## 2. Supplementary optimization ablations

These variants test how the estimator is trained rather than changing its central inference mechanism:

- `safegrip_no_state_update_loss`
- `safegrip_no_direction_loss`
- `safegrip_no_dynamics_pretrain`

Run them in the supplement or optimization section instead of expanding the main ablation table excessively.

## 3. Fixed-vs-retuned ablation rule

The **primary** ablation must reuse the selected full-model hyperparameters. Otherwise each removal is confounded by a new optimization problem.

A separate robustness check can retune each ablation:

```bash
safegrip tune-ablation --dataset lira --trials 15
```

Report this as a supplementary result only. If a component-removal model is poor under frozen settings but recovers after retuning, say so explicitly.

## 4. Point-model hyperparameter search

The proposal search is now expanded to the parameters that can materially alter the point estimator.

### Representation / optimization

- `sequence_length`: 8, 16, 24, 32, 64
- `hidden`: 32, 64, 96
- `gru_hidden`: 16, 32, 64
- `dropout`: continuous 0.0–0.3
- `lr`: log-uniform 1e-4–3e-3
- `weight_decay`: log-uniform 1e-6–1e-3
- `batch_size`: 128, 256, 512
- `huber_beta`: 0.03, 0.05, 0.10

### Persistent innovation state

- `evidence_window`: 4, 8, 12, 16
- `delta_scale`: 0.2, 0.3, 0.5, 0.8
- `state_persistence`: 0.5, 0.7, 0.8, 0.9, 0.97

### Multi-scale counterfactual observability

- `counterfactual_delta`: 0.03/0.04–0.12 depending on preset
- `counterfactual_scale_span`: 1.5, 2.0, 3.0
- `identifiability_lambda`: 1e-4, 1e-3, 1e-2
- `linearity_penalty`: 0.0, 0.25, 0.5, 1.0

### Residual trust-region veto

- `acceptance_temperature`: 6, 12, 20
- `acceptance_margin`: -0.02, 0.0, 0.02
- `acceptance_tolerance`: 0.05, 0.10, 0.20
- `acceptance_strength`: 0.2, 0.35, 0.5

### Inverse-dynamics agreement

- `agreement_temperature`: 6, 10, 16
- `agreement_threshold`: 0.2, 0.35, 0.5
- `agreement_strength`: 0.0, 0.1, 0.2, 0.35
- `inverse_dynamics_ridge`: 1e-4, 1e-3, 1e-2
- `inverse_dynamics_max_step`: 0.06, 0.12, 0.20

### Training-objective weights

- `innovation_loss_weight`
- `state_update_loss_weight`
- `candidate_loss_weight`
- `cf_agreement_loss_weight`
- `direction_loss_weight`
- `dynamics_loss_weight`
- `counterfactual_loss_weight`
- `counterfactual_margin`
- `dynamics_pretrain_epochs`
- `do_no_harm_weight`

`information_beta` is intentionally **not tuned against point RMSE**. It affects UQ inflation after the point estimator is selected and is therefore fixed during point-model search.

## 5. Trial budgets

The default full-development configuration uses 80 proposal trials. `kaggle_trust.yaml` uses 60, while `kaggle_small.yaml` uses 20 as a plumbing/development compromise.

The expanded space is much larger than v0.9, so small trial counts should not be interpreted as strong optimization evidence. For the final paper, report the exact search space, trial count, seed, objective, and validation endpoint hash.

## 6. Hyperparameter sensitivity study

After choosing `best_hparams.yaml`, run a stability study without test labels:

```bash
safegrip sensitivity \
  --dataset lira \
  --proposal-hparams results/lira_tuning/best_hparams.yaml
```

By default it sweeps:

- `state_persistence`
- `counterfactual_delta`
- `counterfactual_scale_span`
- `identifiability_lambda`
- `linearity_penalty`
- `acceptance_strength`
- `agreement_strength`
- `inverse_dynamics_max_step`
- `innovation_loss_weight`
- `dynamics_pretrain_epochs`

Outputs:

- `hyperparameter_sensitivity.csv`
- `hyperparameter_sensitivity.json`

The CSV includes validation RMSE/MAE and mechanism diagnostics such as mean authority, identifiability, acceptance, scale-CV, local linearity, and counterfactual agreement. Use this to show whether performance degrades smoothly or collapses around a narrow setting.

## 7. Statistical comparison

Final conclusions should use independently trained seeds and matched trajectory-aware resampling. Do not treat heavily overlapping temporal endpoints as independent samples. Point accuracy, physical-bound behavior, uncertainty calibration, authority diagnostics, and low-excitation behavior should be reported separately.
