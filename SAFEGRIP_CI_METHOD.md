# SafeGrip-CI method — current release v1.3

The active proposal is **Counterfactual Energy-Guided Selective Physics Correction**.

A causal Conv1D+GRU produces the primary friction estimate. A separately pretrained friction-conditioned dynamics network scores a local grid of nearby friction hypotheses. Their energy posterior produces a physics candidate, posterior entropy gives local identifiability, and two learned heads separately estimate (1) whether physics is likely to help and (2) what fraction of the proposed correction should be used. The final correction is followed by the optional physical feasible-set projection and leakage-safe conformal UQ.

The dynamics network is trained with endpoint reconstruction plus multi-negative counterfactual friction discrimination and is frozen by default during selective-correction training. The point objective includes direct base supervision, metric-aligned unsafe-overestimation loss, selector supervision, and an explicit do-no-harm term.

See `SAFEGRIP_CI_V13_METHOD.md` for the complete method and equations. Historical v1.1/v1.2 implementations remain available for reproducibility in `SAFEGRIP_CI_V11_METHOD.md` and `SAFEGRIP_CI_V12_METHOD.md`.
