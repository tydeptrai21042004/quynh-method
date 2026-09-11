# SafeGrip-CI method — current release v1.2

The active SafeGrip-CI method is defined in **`SAFEGRIP_CI_V12_METHOD.md`**.

SafeGrip-CI v1.2 is a risk-aware selective-physics-correction estimator:

1. a strong raw-sensor Conv1D+GRU network predicts the primary friction estimate;
2. a separate friction-conditioned dynamics model measures local counterfactual observability;
3. a damped inverse-dynamics step proposes a bounded residual correction;
4. a learned utility gate predicts how much of that correction to apply using inference-available evidence;
5. point training includes explicit friction-change, regime-change and unsafe-overestimation objectives;
6. the final estimate is physically projected and uncertainty is calibrated by disjoint block-max split conformal prediction.

The full v1.2 proposal has no recursive friction-state prior, adaptive persistence, direction/magnitude split, three-expert arbitration, agreement veto or multi-scale linearity authority chain.

For exact equations, objective definitions, semantics and the primary ablation set, use `SAFEGRIP_CI_V12_METHOD.md`. `SAFEGRIP_CI_V11_METHOD.md`, `SAFEGRIP_V3_METHOD.md` and `SAFEGRIP_V2_METHOD.md` are historical implementation records only.
