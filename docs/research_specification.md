# Universal SafeGrip-PFR-ECR — Research Specification

## Central research question

Can a single physically typed sensor-set model learn friction, force, utilization, and grip-related quantities from heterogeneous real sensor systems without dataset-specific predictive branches, while preserving a mathematically interpretable PFR-ECR safety layer?

## Scope

The existing SafeGrip-PFR-ECR implementation remains the reproducibility/reference branch. The universal research track is additive. Raw-file parsing may be dataset-specific, but the predictive model must not consume a dataset identifier or switch neural architecture by dataset.

## Proposed contributions

### C1 — Physically typed sensor representation

Each channel is represented as

\[
S_c = \{(t_i,x_i)\}_{i=1}^{n_c},\qquad
M_c=(q_c,a_c,l_c,u_c,f_c),
\]

where quantity, axis, location, unit/dimension and sampling information are explicit. Values are numerically converted to canonical units before neural processing.

### C2 — Sensor-set invariant latent state

For token set \(Z=\{z_1,\ldots,z_N\}\), a learned latent bank forms

\[
L=\Phi(Z)\in\mathbb{R}^{K\times d},
\]

with no index-based positional encoding. Physical time is part of each token. Therefore token ordering is not semantically meaningful.

### C3 — Compositional physical query decoding

Targets are represented by physical semantics rather than separate dataset heads:

\[
\hat y_q=D(q,L).
\]

Examples include friction, force components, utilization and grip margin.

### C4 — Cross-dataset mechanics supervision

Datasets may expose different targets. Training uses a masked objective so force-only samples can supervise mechanics without fabricating friction labels, and friction-labeled samples can supervise friction without requiring measured wheel forces.

### C5 — PFR-ECR safety integration

Force utilization is treated as a physical utilization quantity,

\[
u=\frac{\sqrt{F_x^2+F_y^2}}{|F_z|+\varepsilon},
\]

not as an automatic estimate of available friction. Under explicitly stated friction-limit assumptions, \(u\le \mu\) may be used as lower evidence/consistency information. It must not be equated with \(\mu\) without an additional identifiability assumption.

The existing one-sided conformal lower estimate is retained:

\[
C_\alpha=\hat\mu-q_\alpha\hat\sigma.
\]

When an independently justified mechanics lower estimate \(L_\beta\) is available, the controller-facing value can use

\[
\mu_{\rm safe}=\max(C_\alpha,L_\beta),
\]

subject to the validity assumptions of both component lower estimates.

## Research hypotheses

- **H1:** Physical metadata improves transfer/generalization compared with waveform-only encoding.
- **H2:** Sensor-set fusion is more robust to missing/variable sensor configurations than the fixed-schema GRU reference.
- **H3:** Auxiliary force/utilization supervision improves friction prediction on friction-labeled domains, including when force sensors are absent at inference.
- **H4:** Sensor-channel dropout improves robustness to missing and unseen sensor combinations.
- **H5:** PFR-ECR improves conservative operational behavior without requiring dataset-specific predictive heads.

## Claims deliberately excluded until empirically demonstrated

The project must not claim that the model works with arbitrary unseen physical modalities, that utilization equals available friction, that pooled conformal calibration guarantees per-domain coverage, or that an auxiliary dataset supplies friction labels when the source does not contain them.

## Dataset roles grounded in the current repository

- **LiRA-CD:** primary real road-friction benchmark with synchronized vehicle sensing and VIAFRIK reference.
- **MSSP 2023:** second friction benchmark only after the authors' real dynamics payload is supplied and passes the existing schema audit.
- **KU Leuven:** wheel-force/vehicle sensing source for mechanics and potential universal training after sequence/timing semantics are verified.
- **KIT:** measured tire-force/utilization mechanics source; whether it is used as a temporal training domain depends on the source table semantics rather than fabricated sample timing.
- **Mendeley friction:** real friction/speed/surface reference. The present repository explicitly lacks the synchronized dynamics required by the legacy PFR benchmark. Numeric context can now be represented, but it should not be presented as equivalent to a synchronized dynamics sequence.
- **Deep Dynamics/comma2k19:** auxiliary dynamics/domain data without direct friction ground truth.

## Acceptance criterion

Adding a compatible new measured dataset may require a parser and physical metadata mapping, but must not require a new neural encoder, a dataset-specific predictive head, or an `if dataset == ...` branch inside Universal SafeGrip.
