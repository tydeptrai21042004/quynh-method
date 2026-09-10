# SafeGrip-CI v0.9 ablation and hyperparameter protocol

## Controlled primary ablation

All primary variants use identical split definitions, endpoint identities, seeds, and the same frozen full-model hyperparameters unless the optional retuned-ablation study is requested. This makes the primary table a component-removal study rather than a collection of separately optimized models.

The v0.9 primary set is:

- `safegrip_backbone_raw`
- `safegrip_persistent`
- `safegrip_neural_innovation`
- `safegrip_no_identifiability`
- `safegrip_excitation_proxy`
- `safegrip_no_acceptance`
- `safegrip_no_cf_agreement`
- `safegrip_no_innovation_supervision`
- `safegrip_no_bound`
- `safegrip_no_uq`
- `safegrip`

Three comparisons are scientifically central:

1. `safegrip_excitation_proxy` vs `safegrip` tests learned counterfactual identifiability against the legacy handcrafted excitation authority.
2. `safegrip_no_acceptance` vs `safegrip` isolates the asymmetric counterfactual veto.
3. `safegrip_no_cf_agreement` vs `safegrip` isolates the local inverse-dynamics agreement objective while leaving the inference architecture unchanged.

The ablation registry is explicit rather than inferred from variant names. Unit tests verify that primary point-estimation variants have distinct semantic specifications; `safegrip_no_uq` is intentionally allowed to share the same point prediction because it removes only post-hoc uncertainty estimation.

## Point objective

The full v0.9 proposal separates candidate learning from authority. The raw latent innovation is directly supervised against the required latent correction, the candidate point receives an auxiliary regression loss, and a directional penalty discourages sign-inverted updates. The friction-conditioned dynamics model is warm-started and trained with reconstruction plus counterfactual ranking. A detached local inverse-dynamics Gauss--Newton step supplies an auxiliary agreement target, and a do-no-harm loss discourages accepted updates that worsen the persistent prior. UQ is fitted only after point-model selection.

## Hyperparameter search

Validation-only proposal tuning may search:

- sequence length and encoder sizes;
- learning rate, weight decay, and batch size;
- evidence/innovation scales and state persistence;
- counterfactual friction displacement and identifiability regularization;
- veto temperature, tolerance, and maximum attenuation strength;
- innovation, candidate, inverse-dynamics-agreement, dynamics, counterfactual-ranking, and do-no-harm loss weights;
- post-hoc information-conditioned UQ inflation.

The proposal uses the Optuna study name `safegrip_ci_v090_rmse`, so existing v0.8 proposal trials cannot be silently reused. The test partition remains locked during search. Proposal and literature-comparator tuning use the same validation endpoint definition; paper-readiness checks verify endpoint-hash parity before final packaging.

## Statistical comparison

`metrics_by_seed.csv` is the primary source for single-model multi-seed performance. `predictions_by_seed.csv` stores the corresponding endpoint predictions. Paired significance analysis resamples matched random seeds and then trajectory segments, not individual overlapping windows. `predictions.csv` remains an ensemble-mean artifact and is not silently mixed with the per-seed estimand.
