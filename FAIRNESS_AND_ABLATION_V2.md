# Fairness and Ablation Protocol v2

This release hardens the scientific comparison around SafeGrip-v3 without changing the core proposal.

## Implemented controls

- **Global tuning endpoint parity**: proposal and literature searches now use the maximum context length across every search space. Each tuning run exports exact validation endpoint IDs and a SHA-256 hash.
- **Feature-parity controls**: selected literature models (default: Lampe GRU and Schäfke Transformer) are re-run with the same label-free SafeGrip engineered representation.
- **Equal supervised-label-budget controls**: literature baselines may additionally train on the lower-bound-calibration labels that affect SafeGrip's point-support calibration. The disjoint UQ-calibration role remains held out.
- **Common conformal UQ controls**: all point estimators can be evaluated with the same symmetric split-conformal procedure on the same UQ-calibration role.
- **Projection parity** remains separate from raw point-estimation comparison.
- **Automatic fairness audit**: trust/paper benchmark runs export `fairness_audit.json` and fail if validation/test endpoint identities differ.

## Corrected ablations

- `safegrip_data_only`: raw sensors only; no proposal engineered features, sample-specific lower bound, excitation gate, weak-excitation regularizer, or excitation-dependent UQ inflation.
- `safegrip_static_only`: genuinely endpoint-only. It uses `x[t]` and cannot use window mean/std, GRU state, or dynamic evidence.
- `safegrip_no_excitation`: raw sensors with the temporal prior/evidence architecture but no excitation-derived features, monotone gate, weak-excitation regularizer, or excitation UQ inflation.
- `safegrip_no_gate`: removes only the learned monotone reliability gate while retaining other excitation-aware mechanisms.
- `safegrip_no_bound`, `safegrip_no_uq`, and `safegrip_no_calibration` retain their one-component interpretations.

The main ablation remains a **controlled/frozen-hyperparameter** study. A separate `tune-ablation` command implements **retuned supplementary ablations** to answer the objection that a removed component may require a different hyperparameter optimum.

## Reviewer-oriented commands

```bash
# Proposal tuning with global validation endpoint parity
python -m safegrip.cli tune --dataset lira

# Literature baseline tuning on the same endpoint universe
python -m safegrip.cli tune-baselines --dataset lira

# Controlled ablation
python -m safegrip.cli ablation --dataset lira --preset paper

# Retuned supplementary ablation
python -m safegrip.cli tune-ablation --dataset lira --trials 15

# Main benchmark; trust/paper modes emit fairness controls
python -m safegrip.cli benchmark --dataset lira --preset paper

# Excitation-stratified analysis
python -m safegrip.cli experiment --dataset lira --study excitation

# Leave-one-route-out analysis
python -m safegrip.cli experiment --dataset lira --study cross-route

# Paired bootstrap RMSE differences on identical endpoints
python -m safegrip.cli statistics --results results/lira_paper --bootstrap 2000
```

## Key outputs

A paper/trust benchmark can now emit:

- `fairness_audit.json`
- `feature_parity_metrics.csv`
- `label_budget_parity_metrics.csv`
- `common_conformal_uq_metrics.csv`
- `projection_control_metrics.csv`
- `evaluation_protocol.json`
- exact endpoint manifests from tuning

These controls should be reported as supplementary analyses when the main table is intended to preserve literature-native inputs/methods.
