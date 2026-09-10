# Implemented improvements

## v0.8.0 — SafeGrip-CI

- removed handcrafted excitation features from the full point estimator;
- added persistent friction-state carry with segment reset and context-prior blending;
- added raw-sensor latent innovation prediction;
- added friction-conditioned causal dynamics model;
- added finite-difference counterfactual friction-identifiability score;
- added candidate-vs-prior dynamics-consistency acceptance;
- final update authority is `acceptance * identifiability`;
- added direct latent innovation supervision;
- added dynamics reconstruction and counterfactual ranking losses;
- added do-no-harm update penalty;
- added dynamics warm-start before joint optimization;
- changed native UQ inflation from excitation-dependent to identifiability-dependent;
- replaced name-inferred ablations with an explicit semantic registry;
- added tests preventing duplicate/no-op primary ablations;
- retained bounded identified-set parameterization and all leakage/fairness safeguards.
