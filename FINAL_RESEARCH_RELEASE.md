# SafeGrip-Open v1.2.0 — risk-aware selective physics-correction release

v1.2.0 is a proposal redesign driven by the real v1.1 trust evidence. The previous full estimator was safer than the strongest GRU comparator but did not win on RMSE and its arbitration was dominated by the persistent prior. v1.2 therefore simplifies the point path instead of adding more gates.

## Active proposal

- strong raw-sensor Conv1D + GRU temporal base estimator;
- no recursive friction-state dependence in the full path;
- separate friction-conditioned causal dynamics model;
- counterfactual sensitivity used strictly as observability evidence;
- bounded damped inverse-dynamics residual correction;
- one learned correction-utility gate rather than prior/neural/inverse arbitration;
- explicit friction-change and regime-change learning;
- asymmetric unsafe-overestimation objective;
- conditional physics projection;
- heteroscedastic evidence and block-max split-conformal UQ.

## Experimental hardening

- 11-variant primary v1.2 ablation with one mechanism question per control;
- v1.2 validation-only tuning/sensitivity space;
- same validation-endpoint hash parity checks for proposal and tuned baselines;
- same matched-seed/trajectory hierarchical bootstrap protocol;
- same feature, label-budget, common-conformal and projection fairness controls;
- release hash audit and paper-readiness gate updated to the v1.2 mechanism set.

## Validation status

The local repository unit/regression suite passes. A synthetic end-to-end execution smoke test is used only to check interfaces/training stability. It is **not** evidence that v1.2 beats any real-data baseline. Run TRUST on real LiRA before PAPER mode and report REVIEW if any scientific-health/fairness/statistics/ablation gate fails.
