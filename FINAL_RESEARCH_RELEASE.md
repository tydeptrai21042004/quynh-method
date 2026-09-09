# SafeGrip-Open v0.7.0 — SafeGrip-v3 proposal release

v0.7.0 keeps the leakage, LiRA decoding, trajectory segmentation, data-audit, baseline-fidelity and scientific-health safeguards from v0.6.x, but redesigns the proposal in response to the v0.6 trust result.

## Main proposal changes

- replaces hidden static/GRU gated fusion with an explicit **long-context prior + short-context dynamic-evidence update**;
- uses monotone excitation reliability, so stronger label-free excitation cannot reduce evidence weight;
- preserves the excitation score in physical `[0,1]` scale instead of feeding a standardized gate input;
- adds causal recent-excitation memory so a maneuver remains informative shortly after its peak;
- retains identified-set output parameterization, now as `q_prior + reliability*evidence_delta`;
- adds same-segment relative-change, ranking and weak-excitation smoothness regularization to reduce flat mean-regression behavior;
- changes the vehicle lower-bound calculation to a conditional vector force balance with explicit uncertainty margins;
- initializes the residual-scale head from validation residual magnitude rather than the oversized default softplus scale;
- retains disjoint lower-bound/UQ calibration roles and block-max conformal calibration;
- exports prior, evidence-delta and reliability diagnostics for direct audit.

## What remains intentionally unchanged

- split-before-imputation and segment-safe temporal windows;
- train-only fitting of feature scalers;
- calibration labels excluded from point-model gradient training;
- cited literature comparator architectures/provenance;
- test partition locked during tuning;
- multi-seed trust/paper modes and result-readiness gates.

A `PASS`/`PAPER_READY` status remains a reproducibility/sanity gate, not evidence that the method must outperform baselines. Negative results remain valid outputs and must not be rewritten as positive claims.
