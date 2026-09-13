# Implemented improvements

## v1.3.0 — Counterfactual Energy-Guided Selective Physics Correction

- Added `SafeGripCI13Net` / `SafeGripV5Net` while retaining v1.2 for reproducibility.
- Replaced the one-step finite-difference/Gauss--Newton correction with a local K-hypothesis friction-energy landscape.
- Added energy-posterior physics candidate, normalized posterior entropy, entropy-derived identifiability, energy improvement and local curvature diagnostics.
- Split the previous scalar utility gate into `benefit_probability` and `correction_fraction`; their product controls correction strength.
- Added training-only useful-candidate classification and oracle correction-fraction supervision without leaking oracle signals to inference.
- Added multi-negative contrastive friction discrimination to dynamics pretraining.
- Added staged base pretraining, dynamics pretraining and selective-correction optimization; dynamics is frozen after pretraining by default.
- Aligned unsafe-overestimation training to the reported +0.05 evaluation threshold.
- Added explicit do-no-harm regularization against degradation from the direct base estimate.
- Simplified the active v1.3 default by disabling legacy dynamic/regime/smoothness losses whose v1.2 ablations did not justify them.
- Changed the proposal default scaler to MinMax while retaining validation-only scaler tuning against Standard scaling.
- Added v1.3 ablations for magnitude head, energy-improvement evidence, contrastive dynamics, and do-no-harm training.
- Extended prediction/benchmark exports with benefit probability, correction fraction, posterior entropy and energy curvature.
- Updated tuning/sensitivity, Kaggle TRUST/PAPER drivers, paper-readiness checks and current method documentation for v1.3.
- Added v1.3 model/registry/tuning tests and end-to-end synthetic training validation.
