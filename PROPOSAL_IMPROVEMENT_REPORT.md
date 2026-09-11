# Proposal improvement report — SafeGrip-CI v1.2

## Evidence that motivated the redesign

The v1.1 trust experiment demonstrated a meaningful safety advantage but not overall point-accuracy superiority. Several simpler ablations also matched or improved the full v1.1 RMSE, adaptive persistence varied little, and the learned arbitration strongly favored the prior while the inverse expert received very little weight. At the same time, removing inverse dynamics degraded the full result, suggesting that physics contained useful information but was being used in the wrong role.

The redesign therefore follows four principles:

1. make the neural temporal estimator strong enough to stand alone;
2. use physics as a bounded residual correction rather than a competing full estimator;
3. learn the **utility of the correction** directly instead of chaining heuristic authority factors;
4. optimize the safety quantity that the proposal actually improved: dangerous friction overestimation.

## v1.2 changes

### Strong direct temporal estimator

A Conv1D frontend and multi-layer GRU estimate friction directly from raw causal windows. There is no recursive previous-friction state in the active v1.2 point path.

### Selective inverse-dynamics correction

A separately trained friction-conditioned dynamics model supplies a local damped inverse correction. The correction is clipped to a trust region and cannot become an unconstrained alternative estimator.

### Utility gate

One gate predicts the useful fraction of the bounded physics residual from inference-time temporal, observability, residual, regime and uncertainty evidence. Its training target measures how much of the proposed correction would reduce the training error. Test labels are never used by the gate.

### Observability semantics

Counterfactual sensitivity now means only local identifiability/observability of friction. It is not multiplied directly into neural correctness or treated as a calibrated confidence probability.

### Dynamic and risk objectives

The loss explicitly supervises friction changes and surface-change regimes and adds an asymmetric penalty for dangerous overestimation. Stable-regime smoothing replaces recursive persistence.

### UQ

A heteroscedastic aleatoric head supplies uncertainty evidence, followed by the existing leakage-safe residual-scale and block-max split-conformal calibration.

## What was removed from the full method

Adaptive persistence, split direction/magnitude innovation, three-expert arbitration, agreement vetoes, multi-scale local-linearity authority and chained trust multipliers are not part of the active v1.2 proposal. Historical implementations remain only to reproduce earlier experiments.

## Required evidence before a paper claim

A new TRUST run must show non-degenerate dynamic tracking, positive R2, acceptable uncertainty width/coverage and clean fairness/statistical/ablation gates. The desirable endpoint is GRU-level or better RMSE while retaining a substantially lower unsafe-overestimation rate. v1.1 numbers must not be relabelled as v1.2 results.
