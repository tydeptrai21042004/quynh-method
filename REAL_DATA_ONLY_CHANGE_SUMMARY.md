# Real-data-only revision

## Primary SafeGrip-PFR-ECR benchmark datasets

- `lira` — real LiRA-CD platoon friction data; automatic download and full proposal/baseline/ablation benchmark.
- `mssp2023_friction` — Guo et al. MSSP 2023 real vehicle-dynamics/friction source; full benchmark is enabled only after the authors' real payload is supplied locally.

No generated benchmark dataset is exposed by `prepare`, `benchmark`, `tune`, or `tune-baselines`.

## Additional real-data support

- `kit` — measured tire-force validation via `force-validate`.
- `kuleuven` — real-vehicle wheel-force validation via `force-validate`.
- `mendeley_friction` — measured friction/speed/surface validation via `friction-reference-validate`.
- `deep_dynamics`, `comma2k19`, `extreme_road`, and `bicycle_tire` remain real auxiliary/domain/mechanics sources.

## MSSP-2023 rule

The public metadata repository points to an author-supplied Baidu data link. `safegrip download --datasets mssp2023_friction` records the source and acquisition instructions. The preparation adapter requires measured friction/adhesion coefficient, speed, longitudinal acceleration, and lateral acceleration columns. Missing channels cause a hard failure; no values are fabricated.

## Kaggle

`KAGGLE_PFR_SINGLE_CELL.py` now:

1. loads the complete default YAML and modifies only the 10-epoch budget;
2. runs the full LiRA proposal + ablations + four paper baselines;
3. runs MSSP-2023 only when a real Kaggle payload is mounted;
4. optionally runs KIT, KU Leuven, and Mendeley real-data validations;
5. packages only real-data outputs.

## Validation

- `pytest -q`: 112 passed, 2 warnings.
- CLI primary benchmark choices: `{lira,mssp2023_friction}`.
- MSSP-2023 measured-channel adapter tested.
- Missing MSSP-2023 dynamics channels tested to fail without generated fallback.
