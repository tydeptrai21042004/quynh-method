> **Legacy note (v2.2):** SafeGrip-PFR is the active proposal. This file is retained only to reproduce earlier experiments.

# Proposal improvement report — SafeGrip-CI v1.4

## Why v1.4 was necessary

Earlier LiRA TRUST results showed that the counterfactual branch could make small internal improvements, but the directly supervised SafeGrip base estimator itself remained weaker than the strongest Lampe GRU comparator. That made selector tuning structurally unable to close the full RMSE gap. v1.4 therefore changes the optimization priority: first make the base estimator competitive, then ask counterfactual physics to provide a residual improvement.

## Implemented changes

### Matched-scale strong GRU base
The active proposal now uses a raw-sensor two-layer GRU with a 100-sample context and 256 recurrent units in the TRUST/default configuration. This removes the previous 96/128-unit capacity disadvantage relative to the strongest GRU comparator.

### Accuracy-first Stage A
The base estimator is pretrained with MSE only. Asymmetric safety, UQ, counterfactual and selector losses are excluded from Stage A. The friction target is standardized using training-only statistics, and the active base head is linear in standardized target space rather than using `mu_upper * sigmoid(z)`.

### Innovation dynamics
The friction-conditioned dynamics branch predicts endpoint sensor innovation instead of the absolute endpoint. This reduces the easy temporal-persistence shortcut and forces friction hypotheses to explain changes in motion-sensitive channels.

### Boundary-aware counterfactual grid
Impossible hypotheses are masked instead of duplicated through clipping. Posterior displacement is centered against the uniform distribution on the valid grid, guaranteeing zero correction for a flat landscape even at a physical boundary.

### Calibrated identifiability
Normalized posterior entropy is multiplied by an absolute normalized energy-margin term. Microscopic numerical energy differences therefore cannot produce high identifiability solely because each sample is normalized by its own tiny spread.

### Single correction controller
The v1.3 product gate `benefit_probability * correction_fraction` is removed from the active point path. `correction_fraction` is now the single continuous controller. Benefit probability remains an auxiliary diagnostic and calibration target.

### Soft utility supervision
Candidate gain is converted to a soft helpfulness target. Correction-fraction supervision is applied with soft utility weights rather than only to a hard helpful subset.

### Stable selector training
After base and dynamics pretraining, the controller is first warmed up with zero base learning rate. Joint refinement then uses a much smaller base LR than controller LR.

### Oracle ceiling diagnostics
Evaluation now exports candidate direction accuracy, oracle switch RMSE and oracle continuous-correction RMSE. These distinguish base weakness, candidate weakness and selector weakness before more tuning is attempted.

## Evidence required next

The implementation is designed to address the observed failure mode, but it does not itself prove improved LiRA accuracy. The next locked experiment should compare:

1. strong v1.4 base versus Lampe GRU;
2. v1.4 full model versus its own base;
3. candidate direction accuracy and candidate-help rate;
4. oracle continuous RMSE versus actual final RMSE;
5. multi-seed stability and bootstrap uncertainty;
6. safety metrics separately from primary point RMSE.

If the oracle continuous RMSE still cannot beat Lampe, the next work should focus on the counterfactual candidate/dynamics representation rather than the controller.
