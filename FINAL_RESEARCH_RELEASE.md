# SafeGrip-Open v0.8.0 — SafeGrip-CI proposal release

v0.8.0 keeps the LiRA decoding, trajectory segmentation, leakage controls, baseline provenance, split-role separation, physics-bound audit, and scientific-health gate from v0.7.0, while replacing the proposal core.

## Proposal change

The full proposal now uses raw sensors and a persistent friction state. A neural branch proposes a latent friction innovation. A friction-conditioned dynamics model provides two independent checks before that innovation is accepted: local counterfactual identifiability and improvement of dynamics consistency relative to the prior. Handcrafted excitation is no longer part of the full estimator and is retained only as an explicit comparator ablation.

## Training safeguards

- previous predicted friction is carried only within a trajectory segment;
- the context prior remains trainable during state blending;
- dynamics-model warm-start prevents a random zero-sensitivity gate;
- innovation direction is supervised directly in latent friction coordinates;
- UQ is still trained after point selection;
- lower-bound and UQ calibration roles remain disjoint;
- test labels remain excluded from training/tuning/calibration;
- primary ablation semantics are unit-tested.

A `PASS` health gate is a reproducibility/scientific-sanity condition, not a novelty or publication guarantee.
