# SafeGrip-Open v2.2 — SafeGrip-PFR release

The active proposal is **SafeGrip-PFR: Physics-Feasible Residual Estimation**.

The v2.2 redesign deliberately removes the inactive PNTR response-refinement stack and keeps the mathematically supported core:

- one GRU trained on the signed residual $r^*=\mu-L_0$;
- one-sided split-conformal relaxation of the mechanics lower endpoint;
- one deterministic projection $\Pi_{[L_\alpha,U]}$;
- a pointwise theorem directly controlling squared friction-estimation error;
- a three-row decisive ablation: direct GRU, raw residual GRU, projected PFR.

The old FRC, PNTR, and CI implementations remain available only for reproducibility.

Local validation for this release: **108 tests passed**.  The synthetic quick end-to-end benchmark also completed successfully and the projection theorem audit reported zero covered-point violations.  These software checks do not replace the required multi-seed real-LiRA paper run.
