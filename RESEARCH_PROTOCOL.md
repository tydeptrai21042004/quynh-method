# SafeGrip research protocol — v1.4.0

## Primary research question

Can a strong raw-sensor temporal friction estimator obtain a better **accuracy-safety trade-off** when a separately trained dynamics model is used only to construct an inference-time counterfactual friction-energy landscape and the resulting physics correction is selectively applied?

## Claim discipline

SafeGrip-CI v1.4 is a hybrid estimator, not a complete tire model. Low counterfactual energy is evidence of compatibility with the learned dynamics model, not proof of physical truth. Entropy-derived identifiability measures concentration of the local energy landscape, not correctness. Physical projection is a safety constraint. No superiority claim is valid until the locked real-LiRA TRUST/PAPER protocol is rerun.

## Locked point path

1. causal raw-sensor two-layer GRU base estimate trained first with standardized-target MSE;
2. separately pretrained friction-conditioned dynamics model;
3. odd local grid of counterfactual friction hypotheses around the detached base estimate;
4. dynamics-energy posterior and posterior-mean physics candidate;
5. entropy identifiability, energy-improvement and curvature evidence;
6. help-probability head and correction-fraction head;
7. selective correction with explicit do-no-harm and metric-aligned overestimation training;
8. optional conditional physical projection;
9. disjoint block-max split-conformal UQ calibration.

## Leakage controls

- Dataset split precedes partition-local imputation/scaling.
- Temporal windows cannot cross trajectory/segment/split boundaries.
- Test labels are locked during tuning.
- Oracle help/fraction targets are training-only and are never inference inputs.
- Physics evidence is detached before selector/point use.
- Dynamics is pretrained separately and frozen by default during selector training.
- Lower-bound calibration and predictive-UQ calibration use disjoint roles.

## Required evaluation

- matched seeds and test endpoints;
- strong published literature baselines;
- matched temporal-base control;
- projection-parity and label/feature-budget fairness controls;
- hierarchical seed/trajectory statistics;
- all 13 primary v1.4 ablations;
- selector diagnostics: help probability, correction fraction, identifiability/entropy, correction coverage and harm rate;
- real-data scientific-health gates before any paper-ready claim.
