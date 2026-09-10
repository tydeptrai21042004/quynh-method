# Validation

## v0.9.0 SafeGrip-CI checks

The repository test suite verifies literature comparator contracts, leakage-safe LiRA preprocessing, segment-safe windows, physics-bound behavior, bounded SafeGrip-CI outputs, persistent-state handling, explicit counterfactual components, distinct primary ablations, the asymmetric counterfactual veto behavior, the inverse-dynamics agreement path, and the corrected statistical comparison protocol.

The v0.9 evaluation pipeline additionally requires per-seed predictions and separates single-seed/multi-seed model performance from ensemble-averaged predictions. Statistical inference is restricted to registered model prediction columns and uses matched seed/trajectory-aware resampling.

## Release validation executed in this patch build

- `PYTHONPATH=src pytest -q` -> **55 passed, 2 warnings**;
- `python -m safegrip.cli --help` succeeded;
- shell syntax validation succeeded for the trust, small-development, paper, and extended workflow scripts;
- the full synthetic smoke workflow was started, but exceeded the packaging-session command limit, so no synthetic performance claim is made from this build;
- real LiRA trust mode remains the required empirical gate before paper mode.

The two warnings are the existing pandas fallback date-parser warning in a schema test and PyTorch's `padding='same'` warning for an even Conv1D kernel in a literature baseline. Neither is a SafeGrip-CI runtime failure.
