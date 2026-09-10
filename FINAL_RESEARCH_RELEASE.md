# SafeGrip-Open v0.9.0 — SafeGrip-CI proposal release

v0.9.0 keeps the LiRA decoding, trajectory segmentation, leakage controls, baseline provenance, split-role separation, physics-bound audit, persistent state, and counterfactual-identifiability core, while correcting the v0.8 innovation/acceptance failure mode.

## Proposal change

The full proposal uses raw sensors and a persistent friction state. A neural branch proposes a friction innovation, trained independently from update authority. Counterfactual friction sensitivity measures local identifiability, while a normalized candidate-vs-prior dynamics comparison provides an asymmetric veto against harmful updates. A detached local inverse-dynamics agreement target supplies an additional physically interpretable learning signal without becoming a handcrafted proposal input. Handcrafted excitation remains only as an explicit comparator ablation.

## Training and evaluation safeguards

- persistent state is carried only within trajectory segments and resets at boundaries;
- candidate innovation supervision is separated from authority learning;
- neutral counterfactual evidence no longer automatically halves the update;
- harmful candidate updates can be asymmetrically vetoed;
- inverse-dynamics agreement is isolated by the `safegrip_no_cf_agreement` ablation;
- final point output remains inside the conditional identified set;
- lower-bound and predictive-UQ calibration roles remain disjoint;
- test labels remain excluded from training, tuning and calibration;
- per-seed predictions are exported explicitly;
- statistical comparisons use matched seed/trajectory-aware resampling instead of treating overlapping windows as independent;
- primary ablation semantics are explicit and unit-tested.

A `PASS`/`PAPER_READY` gate is a reproducibility and scientific-sanity condition, not a novelty, significance, or publication guarantee.
