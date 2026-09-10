# Fairness and ablation protocol — SafeGrip-CI v0.8.0

## Fairness controls

The benchmark preserves exact validation/test endpoint identity across proposal and literature comparators, train-only scaler fitting, locked test labels, disjoint lower-bound/UQ calibration roles, equal-label-budget controls, common conformal controls, and projection-parity controls where applicable.

The full SafeGrip-CI point estimator uses raw sensor channels. Proposal-specific handcrafted excitation features are not used by the full model.

## Primary ablations

The v0.8 primary set is:

- `safegrip_backbone_raw`
- `safegrip_persistent`
- `safegrip_neural_innovation`
- `safegrip_no_identifiability`
- `safegrip_excitation_proxy`
- `safegrip_no_acceptance`
- `safegrip_no_innovation_supervision`
- `safegrip_no_bound`
- `safegrip_no_uq`
- `safegrip`

The explicit semantic registry in `benchmark.py` is unit-tested so primary variants cannot become duplicate aliases. Legacy v0.7 names remain accepted only for backward compatibility and are not part of the primary v0.8 paper ablation.

The central novelty control is `safegrip_excitation_proxy` versus `safegrip`, which replaces counterfactual identifiability with the earlier handcrafted excitation proxy while leaving the rest of the estimator as comparable as possible.

## Statistical reporting

Use per-seed metrics and paired endpoint bootstrap comparisons. Do not report a component as beneficial when its controlled ablation performs equally or better. A fairness-audit `PASS` establishes protocol consistency, not superiority of the proposal.
