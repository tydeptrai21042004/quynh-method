# Proposal improvement report — SafeGrip-CI v1.0

## Executive assessment

The previous v0.9 proposal already had a meaningful idea: a persistent bounded friction state, a separately supervised candidate innovation, a learned friction-conditioned dynamics model, counterfactual identifiability, and an asymmetric residual veto. Its main weakness was not lack of components; it was that the strongest concepts were still only partially connected and some experimental controls were too coarse to prove which mechanism mattered.

v1.0 therefore focuses on three goals:

1. make counterfactual observability harder to game by one arbitrary finite-difference step;
2. make the local inverse-dynamics path affect inference in a controlled, label-free way rather than only act as a training auxiliary;
3. make the ablation and hyperparameter protocol capable of isolating those claims.

## What was changed in the proposal

### A. Multi-scale counterfactual observability

The old estimator measured friction sensitivity at one displacement. The new estimator evaluates three perturbation scales, takes a robust median information score, and discounts sensitivity whose magnitude changes too strongly across scales.

This turns the old question, “is the learned dynamics model sensitive to friction at this delta?” into the stronger question, “is there stable local friction sensitivity throughout a small counterfactual neighborhood?”

New parameters:

- `counterfactual_scale_span`
- `linearity_penalty`

New diagnostics:

- `information_scale_cv`
- `local_linearity`

### B. Corrected inverse-dynamics agreement

The old agreement normalization unintentionally compressed disagreement into roughly the upper half of its intended range. v1.0 maps equal corrections near 1 and opposite-direction corrections near 0.

### C. Agreement-aware asymmetric veto

The inverse-dynamics correction now participates at inference through a mild disagreement veto. Importantly, its veto is multiplied by counterfactual identifiability, because a local inverse correction should not be trusted when friction itself is unobservable.

New parameters:

- `agreement_temperature`
- `agreement_threshold`
- `agreement_strength`
- `inverse_dynamics_ridge`
- `inverse_dynamics_max_step`

New diagnostic:

- `agreement_veto_probability`

### D. Ablation confounds corrected

The handcrafted-excitation comparator now keeps the same raw proposal feature mode and the same residual/agreement trust-region machinery, changing only the authority source from learned counterfactual identifiability to the excitation proxy. The no-identifiability ablation also no longer leaks the learned information score through the agreement veto. These changes make the two most important authority comparisons substantially more defensible.

### E. Training bug corrected

The previous-sample branch used the current endpoint excitation value when constructing the persistent prior during training. It now uses the previous endpoint excitation consistently. This primarily affects the excitation-proxy control and removes an avoidable temporal mismatch.

## Why the novelty claim is stronger

The defensible novelty is **not** “physics-informed neural network for friction estimation.” That space is already populated. The sharper contribution is the combination of:

- a persistent bounded friction state;
- a separately supervised neural state innovation;
- a learned friction-conditioned causal dynamics model;
- a multi-scale local counterfactual observability certificate;
- a cross-scale sensitivity-consistency discount;
- a residual-based asymmetric veto;
- a second, identifiability-weighted inverse-dynamics disagreement veto.

The paper should call this a **multi-scale counterfactual-observability trust-region estimator** or **dual-evidence counterfactual trust-region state estimator**. Avoid claiming that no previous work has ever used any individual ingredient.

## Result-oriented improvements

No code change can honestly guarantee lower LiRA RMSE before the full real-data experiment. v1.0 instead improves the parts most likely to affect final performance and makes them tunable:

- state persistence is now actually searched;
- the observability regularization scale is searched over a range aligned with the implementation;
- residual-veto tolerance/strength/margin are searchable;
- inverse-dynamics trust-region size and ridge are searchable;
- agreement-veto strength/threshold/temperature are searchable;
- dynamics pretraining length and ranking margin are searchable;
- candidate, authorized-update, direction, agreement, dynamics and ranking losses can be tuned independently;
- UQ-only `information_beta` is removed from point-RMSE optimization so trials are not wasted on a parameter that cannot improve the point objective.

The full default proposal search is increased to 80 trials, with 60 for the Kaggle trust preset and 20 for the small development preset.

## Added ablations

New mechanism ablations:

- `safegrip_single_scale_cf`
- `safegrip_no_linearity_consistency`
- `safegrip_no_agreement_veto`
- `safegrip_no_counterfactual_ranking`

New supplementary optimization ablations:

- `safegrip_no_state_update_loss`
- `safegrip_no_direction_loss`
- `safegrip_no_dynamics_pretrain`

These are in addition to the existing raw-backbone, persistent-state, unconditional-innovation, no-identifiability, excitation-proxy, no-acceptance, no-agreement, no-innovation-supervision, no-bound and no-UQ controls.

## Added hyperparameter sensitivity experiment

A new CLI command performs a one-factor-at-a-time validation sensitivity study around the selected proposal configuration:

```bash
safegrip sensitivity --dataset lira \
  --proposal-hparams results/lira_tuning/best_hparams.yaml
```

This produces a paper-ready CSV plus protocol JSON and does not touch test labels.

## Validation status of this code release

- Unit/integration tests: **59 passed**.
- Two warnings remain and are non-fatal: pandas date-format inference in one data test and PyTorch's documented `padding='same'` warning for an even CNN kernel.
- A short synthetic smoke run was used only as a plumbing/behavior check. It did **not** establish a statistically meaningful improvement over v0.9; therefore no accuracy-improvement claim should be made from that run.

## Recommended experiment order

1. Run proposal tuning on LiRA with the trust preset and inspect whether selected authority is non-degenerate.
2. Run the frozen-hyperparameter primary ablation.
3. Run five independent final seeds for SafeGrip and the published baselines.
4. Run the sensitivity command using the selected hyperparameters.
5. Run the retuned ablation only as a supplementary robustness check.
6. Report low-excitation bins and abrupt-friction-transition subsets separately; these are the conditions where the counterfactual authority mechanism should have the clearest scientific reason to help.

## Claim boundary

The architecture is more distinctive and the experiment is more falsifiable than v0.9, but publication-level novelty and accuracy remain empirical questions. The final manuscript should claim improved accuracy only if the locked multi-seed LiRA results support it, and should claim novelty as the specific combined mechanism rather than the broad use of physics, observability, or neural networks.
