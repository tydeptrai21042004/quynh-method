# Implemented correction summary

This version keeps the existing SafeGrip proposal and literature baseline set, but strengthens the parts most likely to be challenged in review.

## Data protocol

- Replaced one global LiRA reference join with per-trip route-local candidate selection.
- Uses explicit route/direction tokens only when they exist; missing metadata remain `unknown`.
- When several reference files share a route/direction, selects the closest reference trace by geometry before alignment.
- Added configurable maximum match distance, heading consistency and monotonic reference progress.
- Added source-name-based speed/acceleration unit conversion instead of magnitude guessing.
- Assigns spatial train/calibration/validation/test blocks before interpolation or resampling.
- Interpolates/resamples only within one `(trip_id, split)` partition.
- Added 20 Hz fixed-rate resampling when timestamps are usable, with a maximum nearest-source time gap.
- Recomputes the physics lower bound after signal preprocessing.
- Adds stable `sample_uid` endpoint IDs.
- Temporal windows are grouped by trip; they cannot bridge independent drives.
- Baseline endpoint parity is checked by exact endpoint IDs, not only by equal target values.

## Proposal / mathematics

- Reworded the implementation around **physics-constrained partial identification** rather than an unconditional physics guarantee.
- Added endpoint projection utilities for predictive intervals.
- Exports raw Gaussian and physics-truncated 95% uncertainty intervals.
- Added nested identified-lower helper matching the monotonic information theorem.
- Clarified deterministic mechanics versus split-conformal statistical relaxation.
- Corrected the no-UQ ablation so MSE is not accidentally counted twice.

## Evaluation

- Paper-mode ablations now use the same final seed list as the main benchmark and export mean/std plus per-seed results.
- Main predictions now export SafeGrip raw mean, projected mean, sigma and physics-truncated interval endpoints.
- Added excitation-stratified analysis.
- Added frozen-network physics-layer robustness analysis for mass, acceleration-error and `mu_upper` assumptions.
- Added training-data scarcity experiments comparing SafeGrip with its data-only TCN+UQ backbone.
- Added explicit leave-one-route-out evaluation when route IDs are available.
- Added KIT/KU Leuven force-mechanics validation command.

## Reproducibility

- Test suite increased to 22 passing tests.
- Added `scripts/run_extended_experiments.sh`.
- `scripts/run_paper.sh` automatically runs the low-cost excitation and robustness analyses after the main benchmark; expensive extended experiments remain opt-in with `RUN_EXTENDED=1`.
