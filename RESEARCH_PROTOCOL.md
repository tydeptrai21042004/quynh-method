# Research protocol — SafeGrip-PFR-ECR

## Primary question

Does excitation-aware residual estimation improve point friction prediction over
matched recurrent controls, and can a separately calibrated mechanics anchor
and normalized statistical lower estimate be fused into a conservative
controller-facing friction value without using test labels?

## Data roles and primary split

The four partitions have non-overlapping roles:

- **train**: neural parameter fitting and input/target scaling;
- **calibration**: mechanics and normalized statistical conformal corrections;
- **validation**: early stopping and approximation hyperparameter selection;
- **test**: final reporting only.

The primary paper configuration uses `split.mode: group_holdout`, so complete
trajectory IDs are assigned to one partition only. Temporal windows are then
built inside one segment and one partition; they never cross trajectory or
split boundaries.

Adjacent windows inside a trajectory can still overlap. Therefore the active
PFR-ECR calibration path uses `uq.method: block_max_split_conformal`. With
`uq.block_size: 0`, the executable block size is
`ceil(sequence_length / stride)`. One worst-case score is retained per
consecutive block inside each trajectory segment before the finite-sample
quantile is selected. This is **dependence mitigation only**; it does not claim
validity under arbitrary temporal dependence.

No test friction label may affect fitting, calibration, early stopping, model
selection, or protocol selection.

## Point estimator

Let

\[
A_{0,t}^{W}=\max_{j\in W_t}L_{0,j}
\]

be the trailing mechanics-derived **anchor**. It is not called an unconditional
deterministic lower bound for the current friction: that stronger
interpretation would require a local temporal-persistence assumption.

The learned residual target and point estimate are

\[
r_t^\star=\mu_t-A_{0,t}^{W},
\qquad
\mu_{\rm point,t}=A_{0,t}^{W}+r_\theta(h_t^{\rm phys}).
\]

The recurrent hidden states are pooled with deterministic excitation weights
obtained from the instantaneous mechanics signal. A positive scale head returns
`\sigma_t` for normalized one-sided conformal calibration.

## Risk allocation and exact finite-sample quantiles

Let

\[
\beta+\alpha_s=\alpha_{\rm total}.
\]

For any finite calibration score vector of length `n`, the implementation uses
the direct split-conformal order statistic

\[
k=\min\{n,\lceil(n+1)(1-\alpha)\rceil\},
\]

and selects the `k`-th ordered score. It does not map that rank back through a
NumPy quantile level, avoiding the historical one-rank over-conservatism.

The mechanics score is

\[
s_i^{\rm phys}=A_{0,i}^{W}-\mu_i,
\qquad
L_{\beta,t}=\max\{0,A_{0,t}^{W}-q_\beta\}.
\]

The normalized statistical score is

\[
z_i=\frac{\mu_{{\rm point},i}-\mu_i}{\sigma_i},
\qquad
C_{\alpha_s,t}=\mu_{\rm point,t}-q_s\sigma_t.
\]

The controller-facing output is

\[
\mu_{\rm safe,t}=\max\{L_{\beta,t},C_{\alpha_s,t}\}.
\]

If each component lower estimate covers the true friction, their maximum also
covers it. Hence the union bound gives joint miscoverage at most
`\beta+\alpha_s`; independence of the two component events is not required.
Population finite-sample coverage still requires exchangeability of the
relevant calibration/test units, or a justified exchangeable-block argument.

## Primary controlled comparison

The D1 LiRA controlled table contains exactly:

- `safegrip_pfr`;
- `du2023_inceptiontime`;
- `todorovic2022_cnn`;
- `lampe2023_gru`;
- `levenberg2023_stft`.

All rows use locked evaluation endpoints and train-only preprocessing. The
Levenberg row is explicitly a low-rate method-structure adaptation because the
20-Hz common LiRA stream cannot reproduce the source high-frequency vibration
band.

`direct_gru_control` and the remaining PFR variants are internal component
controls, not literature baselines.

## Required audits

A paper run must export:

- `paper_baseline_provenance.csv` with all four comparator rows;
- `fairness_audit.json` confirming train/calibration/test role separation;
- `pfr_theorem_audit.json` recording split mode, UQ method, block size, risk
  allocation, and deterministic fusion violations;
- `pfr_safety_fusion_audit.csv` with endpoint-level component/fused coverage;
- `reproducibility_manifest.json` with the effective protocol and PFR settings.

The readiness gate must fail if the four primary comparators are absent, if the
primary split is not `group_holdout`, if block-max calibration is inactive, or
if the deterministic fusion implication is violated.

Quick mode is a software/development run and must not be presented as the final
paper comparison.
