# SafeGrip-Open v1.0.0 — multi-scale counterfactual trust-region release

v1.0.0 keeps the leakage controls, LiRA decoding, trajectory segmentation, baseline provenance, split-role separation, physical-support audit, persistent state and counterfactual-identifiability core. It strengthens the proposal with multi-scale counterfactual observability, cross-scale consistency discounting, corrected inverse-dynamics agreement, and an identifiability-weighted agreement veto.

## Proposal changes

- three-scale finite-difference friction sensitivity instead of a single displacement;
- robust median information aggregation;
- cross-scale coefficient-of-variation diagnostic and local-linearity discount;
- corrected 0–1 candidate/inverse-dynamics agreement;
- inference-time disagreement veto, activated only in proportion to local friction identifiability;
- additional mechanism diagnostics exported to benchmark predictions and metrics.

## Experimental hardening

- new primary ablations for single-scale counterfactuals, linearity consistency, agreement veto and counterfactual ranking;
- supplementary loss/pretraining ablations;
- expanded point-model hyperparameter search, including state persistence and inverse-dynamics trust-region settings;
- UQ-only `information_beta` removed from point-RMSE search;
- one-factor-at-a-time validation sensitivity command;
- explicit endpoint-hash outputs for tuning/sensitivity reproducibility.

## Validation

The repository test suite passes **59 tests**. This release does not assert improved LiRA accuracy before a locked real-data tuning and multi-seed paper run is executed.
