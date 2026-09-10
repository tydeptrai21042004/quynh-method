# Validation

## v0.8.0 SafeGrip-CI checks

The repository test suite verifies:

- literature comparator shapes and provenance checks;
- leakage-safe LiRA preprocessing and segment-safe windows;
- physical lower-bound behavior;
- bounded SafeGrip-CI point output;
- explicit counterfactual component outputs;
- zero dynamics sensitivity blocks the innovation;
- a supplied persistent prior is actually used;
- residual-scale initialization remains numerically sensible;
- endpoint-only controls do not use window history;
- primary ablation semantic specifications are distinct;
- the full proposal uses raw features and counterfactual authority;
- the excitation proxy is an explicit comparator rather than the full method.

In addition to unit tests, release validation must execute an end-to-end synthetic benchmark and controlled ablation through training, stateful prediction, metrics and UQ export. Real LiRA trust mode remains the required empirical gate before paper mode.

## Release validation executed

For the packaged v0.8.0 source tree:

- `PYTHONPATH=src pytest -q` -> **51 passed, 2 warnings**;
- `python -m py_compile` succeeded for package modules, paper-readiness script and Kaggle single-cell scripts;
- `python -m safegrip.cli --help` succeeded;
- a synthetic end-to-end quick benchmark completed through training, persistent stateful inference, CI diagnostics, UQ and result export;
- the synthetic controlled ablation exported all **10** primary v0.8 variants with distinct semantic specifications.

The two warnings are the existing pandas fallback date parser warning in a schema test and PyTorch's `padding='same'` warning for an even Conv1D kernel in a literature baseline. Neither is a SafeGrip-CI runtime failure.
