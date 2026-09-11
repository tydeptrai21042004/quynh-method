# Implemented improvements

## v1.1.0 — dual-expert adaptive-state revision

- added learned prior/neural/inverse arbitration;
- promoted inverse dynamics from veto-only evidence to a correction expert;
- added adaptive state persistence;
- factorized neural innovation into direction and magnitude;
- added scheduled teacher forcing for previous-state training;
- removed multi-scale linearity discount and agreement veto from the full proposal path;
- added disagreement-aware UQ inflation;
- added v1.1 component ablations and regression tests.


## v1.0.0 — multi-scale counterfactual-observability revision

- replaced single-displacement friction sensitivity with three-scale counterfactual sensitivity;
- added robust median information aggregation and cross-scale sensitivity-consistency discounting;
- corrected the inverse-dynamics agreement normalization to use the full 0–1 range;
- added an identifiability-weighted inverse-dynamics disagreement veto at inference;
- added mechanism diagnostics for scale CV, local linearity and agreement-veto probability;
- added primary ablations for single-scale counterfactuals, linearity consistency, agreement veto and counterfactual ranking;
- added supplementary ablations for state-update loss, direction loss and dynamics pretraining;
- expanded proposal tuning to state persistence, trust-region, agreement and optimization parameters that materially affect point estimates;
- removed UQ-only `information_beta` from point-RMSE search;
- added one-factor-at-a-time proposal sensitivity analysis on locked validation endpoints;
- corrected previous-sample excitation alignment during proposal training;
- synchronized package/readiness documentation with v1.0.0.

# Implemented improvements

## v0.9.0 — SafeGrip-CI trust-region revision

- retained persistent friction-state carry with segment reset;
- retained friction-conditioned counterfactual identifiability;
- separated raw candidate-innovation supervision from accepted state-update authority;
- replaced the nearly symmetric candidate acceptance gate with normalized counterfactual evidence and an asymmetric harmful-update veto;
- added a detached local inverse-dynamics agreement target;
- added `safegrip_no_cf_agreement` as a primary ablation;
- kept handcrafted excitation only as an explicit comparator ablation;
- retained bounded identified-set parameterization and disjoint lower-bound/UQ calibration roles;
- exported `predictions_by_seed.csv` so per-seed performance and ensemble performance are not conflated;
- replaced endpoint-only paired bootstrap assumptions with matched seed/trajectory-aware resampling;
- restricted statistical comparisons to actual registered model predictions;
- strengthened paper readiness checks for fairness outputs, tuning-endpoint parity, statistics outputs and the complete v0.9 ablation set;
- synchronized package version and Kaggle trust entry points with v0.9.0.

## v0.8.0 — historical baseline

v0.8 introduced the persistent-state + counterfactual-identifiability redesign. Its real LiRA trust diagnostics motivated the v0.9 correction of innovation supervision, update authority, and statistical inference.
