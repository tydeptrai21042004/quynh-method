# Validation

## SafeGrip-CI v1.2 checks

The repository regression suite verifies the literature comparator contracts, leakage-safe LiRA preprocessing, segment-safe windows, physics-bound behavior, historical v1.0/v1.1 compatibility, and the active v1.2 selective-physics path.

The v1.2-specific tests cover:

- direct temporal estimator + selective-physics outputs;
- valid bounded point predictions;
- zero friction sensitivity producing zero physics correction;
- unconditional-physics ablation behavior;
- inverse-dynamics trust-region step bounds;
- deterministic removal of the heteroscedastic gate feature;
- raw inverse-candidate construction before physical projection;
- distinct semantic specifications for every primary v1.2 ablation;
- v1.2 tuning-space parameters and sensitivity defaults.

A synthetic end-to-end smoke run is used only to verify training/prediction interfaces and numerical finiteness. It is not a real-data performance result.

## Required real-data validation

Run TRUST on LiRA before PAPER mode. TRUST must re-run the test suite, release audit, preprocessing/physics diagnostics, three-seed proposal/baseline benchmark, fairness controls, matched-seed/trajectory hierarchical statistics, and the complete primary v1.2 ablation. Any scientific-health failure keeps the release at `REVIEW` even when code/fairness audits pass.

Paper mode additionally requires validation-only proposal/baseline tuning with identical endpoint hashes, five-seed final evaluation, controlled five-seed ablations and the repository paper-readiness gate.
