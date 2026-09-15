# SafeGrip-Open v1.4.0 — strong-base innovation-energy residual release

v1.4.0 redesigns the active point path around a strong matched-scale GRU base, friction-conditioned innovation energy, and a single continuous residual controller. It is motivated by prior LiRA evidence that the correction branch was being asked to compensate for a weaker primary regressor.

The active method now contains:

- a strong causal raw-sensor Conv1D+GRU base estimator;
- a separately pretrained friction-conditioned dynamics model;
- a K-hypothesis local friction-energy landscape;
- posterior-mean physics candidate and entropy-derived identifiability;
- separate help-probability and correction-magnitude heads;
- multi-negative contrastive dynamics pretraining;
- metric-aligned unsafe-overestimation and explicit do-no-harm objectives;
- optional physical projection and disjoint block-max conformal UQ;
- a 13-variant primary ablation family designed around the v1.4 claims;
- validation-only tuning/sensitivity with test labels locked.

Historical v1.1/v1.2 implementations and method documents remain for reproducibility.

Local tests/smoke runs validate implementation behavior only. A new real-LiRA TRUST/PAPER run is required before claiming v1.4 improves accuracy, safety, or the accuracy-safety trade-off over published baselines.
