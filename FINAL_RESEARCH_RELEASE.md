> **Archived research note.** The current active proposal is **SafeGrip-PFR-ECR** as defined in `SAFEGRIP_PFR_METHOD.md`. Statements below that call another method "active" describe the historical revision documented by this file only.

# SafeGrip-Open v2.2 — SafeGrip-PFR release

The proposal in that historical release was **SafeGrip-PFR: Physics-Feasible Residual Estimation**.

The v2.2 redesign deliberately removes the inactive PNTR response-refinement stack and keeps the mathematically supported core:

- one GRU trained on the signed residual $r^*=\mu-L_0$;
- one-sided split-conformal relaxation of the mechanics lower endpoint;
- one deterministic projection $\Pi_{[L_\alpha,U]}$;
- a pointwise theorem directly controlling squared friction-estimation error;
- a three-row decisive ablation: direct GRU, raw residual GRU, projected PFR.

The old FRC, PNTR, and CI implementations remain available only for reproducibility.

Local software validation is separated from empirical evidence. Paper claims require multi-seed evaluation on real measured datasets; generated benchmark data are not part of the executable data suite.
