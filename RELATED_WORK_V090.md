# SafeGrip-CI v0.9 related-work positioning

This note is a claim-boundary aid, not a proof of novelty. A complete manuscript literature review is still required.

Recent tire/road friction work already includes hybrid model-based learning, vehicle-dynamics-informed deep learning, uncertainty-aware neural estimation, and friction-informed neural operators. SafeGrip-CI therefore should **not** be described broadly as the first physics-informed or hybrid friction estimator.

The v0.9 proposal is intended to be differentiated by the combination of:

1. a persistent latent friction state that is reset at trajectory boundaries;
2. a learned raw-sensor innovation trained independently from update authority;
3. friction-conditioned local counterfactual sensitivity used as an identifiability authority;
4. an asymmetric normalized counterfactual veto that attenuates only materially harmful candidate updates rather than multiplying neutral evidence by an approximately 0.5 sigmoid;
5. a damped local inverse-dynamics Gauss--Newton friction correction used as a detached agreement target;
6. a physics-derived identified set and separate post-hoc predictive uncertainty layer;
7. matched-seed, trajectory-blocked statistical evaluation for overlapping temporal windows.

## Nearby recent work to discuss explicitly

- Yan Wang, Guodong Yin, Peng Hang, Jing Zhao, Yilun Lin, Chao Huang, *Fundamental Estimation for Tire Road Friction Coefficient: A Model-Based Learning Framework*, IEEE Transactions on Vehicular Technology 74(1), 481--493 (2025), DOI: 10.1109/TVT.2024.3464524. It combines event-triggered cubature Kalman filtering, a nonlinear tire model, and EKFNet.
- Xixi Li, Minglun Ren, *Road adhesion coefficient Estimation: Physics-informed deep learning method with vehicle dynamics model*, Expert Systems with Applications 260, 125387 (2025), DOI: 10.1016/j.eswa.2024.125387. It fuses vehicle-dynamics physical features with CNN image features.
- Lintao Yang, Huizhao Tu, Hongren Gong, Yiik Diew Wong, Hao Li, Bo Zhou, *μ-FINO: A friction-informed neural operator for cross-scale tire/road friction prediction using vehicle-mounted 3D lasers*, Measurement (2026 issue/2025 online metadata), DOI: 10.1016/j.measurement.2025.120130. It integrates Persson-theory friction structure with a neural operator for pavement-texture inputs.

These papers are methodologically adjacent but use different sensing assumptions and estimator structures. The manuscript should state the exact structural distinction and then let the ablation/validation establish whether it matters empirically.

## Claim boundary after the v0.8 diagnostic-driven redesign

The v0.8 LiRA test results were inspected to choose the v0.9 redesign. Therefore that exact test split is no longer pristine confirmatory evidence for a newly selected v0.9 architecture. It may be used as development/trust evidence, but strong final claims should be confirmed on a new route-level holdout, an external dataset, or another untouched pre-declared split.
