# SafeGrip-Open v0.6.0 — SafeGrip-v2 research release

This release keeps the v0.5 leakage, LiRA decoding, trajectory segmentation, data-audit and literature-baseline safeguards, but replaces the proposal architecture after the v0.5 trust run correctly failed its scientific-health gate.

## Proposal changes

- default proposal context reduced from 64 to 16 samples at 10 Hz;
- label-free derived dynamics features: acceleration utilization, jerk, wheel-speed spread/imbalance, pressure spread and torque utilization;
- explicit bounded excitation score in `[0,1]`;
- window-statistics MLP + compact GRU branches;
- excitation-aware learned fusion gate;
- point estimate parameterized directly inside the identified set:
  `lower + (mu_upper-lower)*sigmoid(z)`;
- Huber/SmoothL1 point loss instead of joint Gaussian NLL;
- uncertainty scale fitted only after the point network is frozen;
- excitation-dependent uncertainty inflation;
- disjoint calibration roles for lower-bound relaxation and predictive UQ;
- block-max conformal calibration to mitigate overlapping-window dependence;
- automatic health gate also checks predictive interval coverage.

## Benchmark correction

The benchmark warm-up no longer depends on unused sequence lengths listed in the tuning search space. It is the maximum of the configured benchmark warm-up and the context lengths of the methods actually evaluated. Hyperparameter search has its own fixed tuning warm-up.

## What is intentionally unchanged

- official LiRA decoding and preprocessing safeguards;
- split-before-imputation and trajectory/segment-safe windows;
- train-only feature scaling;
- cited literature comparator architectures/provenance;
- locked test partition during tuning;
- five-seed paper mode and result-readiness gates.

A `PASS`/`PAPER_READY` status is still a reproducibility/sanity gate, not evidence that the method must outperform baselines. Negative results remain valid outputs and must not be rewritten as positive claims.
