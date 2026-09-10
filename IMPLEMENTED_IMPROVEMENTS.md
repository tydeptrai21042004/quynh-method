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
