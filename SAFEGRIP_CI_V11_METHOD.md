# SafeGrip-CI v1.1 method

SafeGrip-CI v1.1 replaces the v1.0 chain of multiplicative authority gates with an explicit three-action state update. At each endpoint the estimator can keep the persistent prior, apply a learned neural correction, or apply a local inverse-dynamics correction. The arbitration policy is trained only on training labels; inference uses raw sensor context, counterfactual observability, residual evidence, and expert agreement.

## State prior

The context prior is blended with the previous predicted friction using a learned persistence coefficient

`mu_prior = rho_t * mu_{t-1} + (1-rho_t) * mu_context`,

with `rho_t` constrained to a configured interval. Training uses scheduled teacher forcing that anneals from the previous ground-truth state toward the model's own previous prediction, reducing train/inference mismatch.

## Neural correction

The neural innovation is factorized into direction and magnitude. Direction is bounded by `tanh`; magnitude is positive and bounded by the configured innovation scale. The existing innovation, direction, candidate and final-state losses remain explicit.

## Inverse-dynamics correction

The friction-conditioned dynamics model supplies a damped local Gauss-Newton correction. Counterfactual sensitivity is used as an observability/availability certificate for this inverse expert, rather than as a monotone statement that the neural correction is correct.

## Learned arbitration

A softmax arbitration head produces weights for `[prior, neural, inverse]`. Its inference features include counterfactual information, residual improvement, both correction proposals, direction agreement, magnitude agreement and local sensitivity variability. During training, a best-expert classification target is computed from training labels only. The final correction is the weighted neural plus inverse correction, followed by the physical support constraint.

## Uncertainty

The residual-scale head is retained. Calibration inflation now uses both low observability and neural/inverse disagreement before block-max split conformal calibration.

## v1.1 core ablations

The default controlled set includes no inverse expert, no learned arbitration, fixed persistence, unsplit innovation, no identifiability, no counterfactual ranking, no dynamics pretraining, no innovation supervision, no physical bound and no UQ, plus staged backbone/persistent/neural controls.

The v1.0 multi-scale linearity and agreement-veto variants remain registered as legacy controls, but they are no longer part of the v1.1 central mechanism.
