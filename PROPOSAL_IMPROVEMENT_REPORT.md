# Proposal improvement report — SafeGrip-CI v1.3

## Why v1.3 was necessary

The v1.2 TRUST evidence showed a useful safety direction but three proposal-level weaknesses: the single local inverse-dynamics correction could harm RMSE, the learned utility gate was nearly constant, and Jacobian-based identifiability/bound ablations had little measurable effect. v1.3 changes the mechanism rather than adding another heuristic gate.

## Implemented proposal changes

### Counterfactual energy landscape
The finite-difference/Gauss--Newton step is replaced by an odd local grid of friction hypotheses scored by the friction-conditioned dynamics model. A temperature-controlled energy posterior yields a smooth posterior-mean physics candidate.

### Entropy identifiability
Identifiability is now one minus normalized posterior entropy. A flat landscape therefore has low authority by construction rather than relying on a small Jacobian norm to be numerically meaningful.

### Two-stage selector
A benefit head predicts whether the candidate should help; a separate magnitude head predicts what fraction of the correction should be used. Their product is the correction gate.

### Friction-discriminative dynamics
Dynamics pretraining now includes multi-negative counterfactual classification in addition to endpoint reconstruction. This directly penalizes dynamics models that ignore friction.

### Stable staged optimization
The base estimator is pretrained, then dynamics is pretrained, then the selector/point path is optimized with dynamics frozen by default. Physics evidence is detached from the point loss.

### Risk alignment
The smooth unsafe-overestimation loss now targets the same +0.05 threshold used by evaluation. An explicit do-no-harm term penalizes corrections that make the base estimate worse by more than a margin.

## Evidence required next

The code changes create a stronger and more testable proposal, but they do not establish better real-LiRA performance. The next TRUST run should prioritize: matched base-vs-full gain, matched GRU comparison, seed stability, help-selector discrimination/calibration, correction coverage, harm rate, and the 13 controlled ablations.
