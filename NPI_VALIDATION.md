# NPI validation note

## Repository regression tests

After the Normalized Physical Innovation (NPI) revision, the complete repository test suite passes:

- `169 passed`
- no test failures
- warnings are pre-existing numerical/library warnings (pandas parsing, PyTorch padding/nested-tensor notices)

Targeted NPI tests verify:

1. observed state targets are normalized in residual coordinates rather than full-state scale;
2. if channel dropout removes the physical reference, normalization switches to a direct-target coordinate instead of mixing absolute and residual scales;
3. the zero-initialized point head starts from the physical/reference coordinate;
4. curved-motion displacement uses planar endpoint geometry and remains below travelled path length for a quarter-circle test.

## Controlled synthetic state-prediction sanity check

A small synthetic next-state experiment was run with 20% whole-channel dropout. The target was a next-step longitudinal velocity whose physical innovation was small relative to the absolute speed. The legacy formulation was emulated with physical-output target-scale Huber training; the revised formulation used NPI with the same small UniversalSafeGrip backbone.

| Seed | NPI MAE | Legacy-emulated MAE |
|---:|---:|---:|
| 1 | 0.1516 | 0.1376 |
| 2 | 0.1365 | 0.2614 |
| 3 | 0.1034 | 0.1551 |
| Mean | 0.1305 | 0.1847 |

Mean synthetic MAE decreased by approximately 29%. One seed was slightly worse, so this is evidence of improved conditioning rather than a guarantee of real-data superiority.

## Required real-data confirmation

The real D1-D4 datasets are not bundled in this uploaded repository. Therefore the new code does **not** fabricate replacement results. The supplied Kaggle real-data runner has been updated to fit NPI statistics on training data only and restore the best validation checkpoint; rerunning that script is the required confirmation for LiRA, UC3M, Deep Dynamics IAC, and IO-VNBD.
