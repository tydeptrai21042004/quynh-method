# Validation

## v1.0.0 SafeGrip-CI checks

The repository test suite verifies literature comparator contracts, leakage-safe LiRA preprocessing, segment-safe windows, physics-bound behavior, persistent-state handling, explicit counterfactual components, distinct primary ablations, multi-scale observability consistency, asymmetric residual/agreement veto behavior, inverse-dynamics agreement, and the seed/trajectory-aware statistical comparison protocol.

## Release validation executed in this patch build

- `PYTHONPATH=src pytest -q` -> **59 passed, 2 warnings**;
- `PYTHONPATH=src python -m safegrip.cli --help` succeeded;
- the new `safegrip sensitivity` command completed an executable synthetic smoke sweep and produced its CSV/JSON outputs;
- all three tuning presets contain the proposal parameters needed by the default sensitivity protocol;
- real LiRA trust mode remains the required empirical gate before any accuracy-improvement claim.

The two warnings are pandas fallback date-format inference in a schema test and PyTorch's `padding='same'` warning for an even Conv1D kernel in a literature baseline. Neither is a SafeGrip-CI runtime failure.

The synthetic sensitivity smoke run is a software-path check only. It is not a scientific comparison with v0.9 and is not used to claim better accuracy.
