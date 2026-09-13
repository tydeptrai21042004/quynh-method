# SafeGrip-Open v1.3.0 — counterfactual-energy selective-physics release

v1.3.0 redesigns the active proposal around a local counterfactual friction-energy landscape. It is motivated by the v1.2 real-LiRA evidence showing that the previous single Gauss--Newton correction and nearly constant utility gate did not yet justify their complexity.

The active method now contains:

- a strong causal raw-sensor Conv1D+GRU base estimator;
- a separately pretrained friction-conditioned dynamics model;
- a K-hypothesis local friction-energy landscape;
- posterior-mean physics candidate and entropy-derived identifiability;
- separate help-probability and correction-magnitude heads;
- multi-negative contrastive dynamics pretraining;
- metric-aligned unsafe-overestimation and explicit do-no-harm objectives;
- optional physical projection and disjoint block-max conformal UQ;
- a 13-variant primary ablation family designed around the v1.3 claims;
- validation-only tuning/sensitivity with test labels locked.

Historical v1.1/v1.2 implementations and method documents remain for reproducibility.

Local tests/smoke runs validate implementation behavior only. A new real-LiRA TRUST/PAPER run is required before claiming v1.3 improves accuracy, safety, or the accuracy-safety trade-off over published baselines.
