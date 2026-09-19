# Implemented SafeGrip-FRC redesign

## Primary proposal changes

- Added `src/safegrip/friction_resolution.py` with the executable mathematical core:
  - friction candidate grid;
  - residual objective;
  - finite-window separation margin;
  - grid-aware resolution certificate;
  - adaptive horizon selection;
  - theorem-audit helper.
- Added `FrictionResponseNet` and `DirectGRUControl` to `models.py`.
- Added `src/safegrip/frc_benchmark.py` for the complete theorem-driven training/evaluation path.
- Primary FRC inference has no learned correction gate, no benefit head, no entropy posterior, no heteroscedastic head, and no physical projection.
- Calibration residual radii are fitted only from the calibration split.
- Input scaling and response normalization are fitted only from training data.
- The response context is strictly causal.

## Baseline/fairness changes

- Added controlled and source-setting comparison protocols.
- Controlled baseline tuning uses fixed common tuning seeds.
- Controlled benchmark uses a common optimization budget.
- Added `direct_gru_control` as a same-encoder scientific control.
- Physical projection is exported only as a common secondary control.
- Updated Lampe fidelity wording to avoid implying exact experiment reproduction.
- Legacy CI-v1.4 tuning no longer wastes Optuna dimensions on inactive `conv_channels`, inactive `huber_beta`, or fixed zero-valued losses.

## Data/split changes

- Retained the default contiguous within-trajectory split.
- Added `split.mode: group_holdout` for whole-trajectory generalization studies.

## Test changes

- Added closed-form separation test.
- Added separation monotonicity test.
- Added finite-grid theorem test.
- Added certificate/grid-step test.
- Added adaptive-horizon/fallback test.
- Added FRC model shape/conditioning test.
- Added whole-trajectory split test.

Current complete test status after the redesign: **95 passed**.
