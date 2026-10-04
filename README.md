# Universal SafeGrip-NPI

This repository contains **one active proposal only**:

> **Universal SafeGrip-NPI — Normalized Physical Innovation (NPI-v3)**

The method is dataset-agnostic at the model level. Dataset adapters only resolve raw columns, timestamps, units, and measured targets; they do not select a dataset-specific neural backbone or output head.

## Core method

For physical query `q`, let `P_q(X)` be a deterministic semantic physical reference computed only from visible measurements. Let `a_q(X)` indicate whether that reference is observable. Training-only robust statistics are fitted separately for the reference-present residual coordinate and the reference-absent direct coordinate.

The decoder predicts a dimensionless innovation `z_hat = R_theta(X,q)` and reconstructs

```text
reference present:  y_hat = P_q(X) + c_q,ref    + s_q,ref    * z_hat
reference absent:   y_hat =          c_q,direct + s_q,direct * z_hat
```

Equivalently,

```text
y_hat = a_q P_q(X) + c_q,a + s_q,a R_theta(X,q).
```

`c` is the training median and `s = 1.4826*MAD` with a standard-deviation fallback. They are deterministic buffers indexed by physical query semantics, not trainable calibration parameters.

### Physical references

- observed vehicle state: latest visible matching state value;
- road friction: horizontal specific-force demand used only as a physical reference coordinate;
- displacement: curvature-aware endpoint displacement from measured speed and yaw-rate integration;
- unsupported/missing reference: neutral reference, with NPI switching to the robust direct-target coordinate.

Whole-channel dropout is respected when computing the reference, so the same decoder remains well-conditioned when an anchor channel is removed.

## Architecture

The active proposal uses:

1. canonical physical units;
2. physically typed temporal sensor tokens;
3. permutation-invariant latent cross-attention;
4. one shared compositional physical-query decoder;
5. NPI-v3 reconstruction in physical units.

There is **no RA-NPI gain**, **no CSI affine calibration**, **no semantic relational-contrast transform**, **no dataset-specific proposal head**, and **no dataset-name routing** in the active method.

## Paper datasets and comparators

The real-data study uses:

- D1 LiRA-CD — road friction;
- D2 UC3M intelligent tire — slip angle;
- D3 real Indy Autonomous Challenge logs — longitudinal velocity, lateral velocity, yaw rate;
- D4 IO-VNBD — GPS endpoint displacement.

Only publication-backed comparators registered in `src/safegrip/literature.py` are used in the primary comparison. Dataset-specific code is restricted to measurement adapters and local comparator reproductions.

## Reproduce the proposal

Single seed, all real datasets:

```bash
python scripts/KAGGLE_ALL_REAL_DATASETS_ONE_PROPOSAL_1SEED_10EPOCHS.py
```

Three-seed paper evidence:

```bash
SAFEGRIP_SEEDS=3101,3102,3103 \
python scripts/KAGGLE_ALL_REAL_DATASETS_MULTI_SEED.py
```

The Kaggle runner rejects synthetic scientific fallback data. If required real data cannot be resolved, the stage is marked unavailable rather than replaced with generated targets.

## Validation

```bash
python -m pytest -q
```

The final NPI-only revision passes the repository test suite locally. See `NPI_VALIDATION.md` and `NPI_CHANGE_PLAN.md` for the method rationale and validation scope.

## Main implementation files

- `src/safegrip/universal/innovation.py` — NPI coordinate and robust training-only statistics;
- `src/safegrip/universal/reference.py` — semantic physical reference operator;
- `src/safegrip/model/safegrip_universal.py` — shared model and NPI reconstruction;
- `src/safegrip/model/query_decoder.py` — compositional query decoder with zero-initialized innovation head;
- `src/safegrip/training/trainer.py` — normalized-innovation optimization after sensor dropout;
- `scripts/KAGGLE_ALL_REAL_DATASETS_ONE_PROPOSAL_1SEED_10EPOCHS.py` — one-proposal real-data benchmark.

Historical proposal documents and launchers are intentionally removed from the active release to avoid ambiguity about which method is being evaluated.
