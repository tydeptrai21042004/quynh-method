# Implemented improvements

## v1.4.0 — Strong-Base Innovation-Energy Residual Correction

- Replaced the active point backbone with a matched-scale two-layer raw-sensor GRU (TRUST/default: 256 recurrent units, 100-sample context).
- Added training-only target standardization and removed sigmoid compression from the active base regressor.
- Changed Stage A to pure MSE accuracy pretraining with no safety/UQ/counterfactual losses.
- Changed the dynamics target from absolute endpoint reconstruction to endpoint innovation.
- Added boundary-aware masked counterfactual grids with uniform-centered posterior displacement.
- Added absolute energy-margin strength to entropy identifiability.
- Removed the v1.3 product gate from the active prediction path; one continuous correction fraction now controls the residual.
- Replaced hard helpful/not-helpful selector targets with soft utility supervision.
- Added controller warm-up with frozen base and low-LR joint refinement.
- Added candidate-direction, oracle-switch, and oracle-continuous RMSE diagnostics.
- Kept v1.3/v1.2/v1.1 classes available for reproducibility.

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
