# Related-work positioning for SafeGrip-CI v1.0

SafeGrip-CI should **not** be positioned as the first hybrid, physics-informed, observer-assisted, uncertainty-aware, or excitation-aware tire-road friction estimator. Recent literature already covers each of those broad categories.

## Closest methodological neighborhoods

### Adaptive filtering and excitation handling

Li et al. (IEEE TIE 2024, DOI `10.1109/TIE.2024.3370986`) use an adaptive strong-tracking Kalman filter with both residual-driven fading and a real-time excitation factor. This is important prior art for any claim involving low-excitation reliability. SafeGrip-CI should distinguish itself by using a **learned counterfactual friction sensitivity** rather than a handcrafted excitation factor, and by testing that sensitivity over multiple friction perturbation scales.

### Model-based learning

Wang et al. (IEEE TVT 2025, DOI `10.1109/TVT.2024.3464524`) combine event-triggered filtering, a nonlinear tire model, EKF structure and a neural network. Therefore “hybrid model + learning” is not sufficient novelty. SafeGrip-CI's sharper distinction is that the learned dynamics model is queried under nearby **counterfactual friction hypotheses** to determine update authority.

### Physics-informed deep learning

Li and Ren (Expert Systems with Applications 2025, DOI `10.1016/j.eswa.2024.125387`) fuse vehicle-dynamics physical information with a deep-learning road-adhesion estimator. SafeGrip-CI should avoid a generic “physics-informed network” claim and instead emphasize the local counterfactual-observability/trust-region mechanism.

### Data-driven spatiotemporal estimation with uncertainty

Chen et al. (IEEE TIE 2025, DOI `10.1109/TIE.2024.3440510`) use spatiotemporal stochastic-variational deep-kernel learning with uncertainty evaluation. SafeGrip-CI's post-hoc conformal UQ is therefore an evaluation/reliability feature, not the central novelty claim.

### Gain-scheduled observer + neural estimation

Shi and Li (Vehicles 2026, DOI `10.3390/vehicles8070148`) combine a piecewise gain-scheduled observer with neural friction estimation and explicitly discuss observability variation with vehicle motion. SafeGrip-CI should distinguish its **sample-wise learned counterfactual information certificate** from piecewise state-dependent observer scheduling.

### Multimodal low-excitation fusion

Wu et al. (Mechanical Systems and Signal Processing 2026, DOI `10.1016/j.ymssp.2026.114546`) use visual priors with an adaptive UKF to improve low-excitation estimation. Xiong et al. (Information Fusion 2026, DOI `10.1016/j.inffus.2026.104296`) develop uncertainty-informed multimodal fusion. These papers make it especially important to state that SafeGrip-CI targets a different question: **when only the current vehicle-dynamics evidence is available, how much should a learned friction-state innovation be trusted?**

## Defensible v1.0 contribution statement

A conservative paper-ready contribution statement is:

> We propose a persistent bounded friction-state estimator in which a neural state innovation is authorized by a learned, multi-scale counterfactual observability score. A friction-conditioned causal dynamics model is queried over neighboring friction hypotheses; robust sensitivity magnitude and cross-scale consistency determine local identifiability, while normalized candidate-vs-prior dynamics residuals and an identifiability-weighted inverse-dynamics disagreement provide asymmetric vetoes. This separates candidate estimation from evidence-dependent update authority without relying on a handcrafted excitation factor.

This is a **combination-level contribution claim**, not a claim that each ingredient is individually unprecedented.

## Experiments needed to support that claim

The main paper should show, under identical endpoints and frozen full-model hyperparameters, that:

1. counterfactual observability outperforms or behaves more robustly than the handcrafted excitation proxy;
2. multi-scale sensitivity improves over the single-scale finite difference;
3. cross-scale consistency is useful beyond sensitivity magnitude alone;
4. the residual veto and agreement veto provide distinct value;
5. the full method does not obtain its gain solely from extra losses or dynamics pretraining;
6. behavior under low-excitation and abrupt-friction-transition subsets matches the mechanism's stated motivation.
