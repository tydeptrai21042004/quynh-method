# Final NPI-v3 code change manifest

## Active proposal

`Universal SafeGrip-NPI / normalized_physical_innovation_v3`

## Reverted from the uploaded RA-NPI-v5 repository

1. Removed the RA-NPI semantic reference gain `alpha_q` and its identity-anchor hyperparameter.
2. Removed CSI affine/reference calibration behavior.
3. Removed training-only semantic feature-standardization proposal preprocessing.
4. Removed semantic relational-contrast preprocessing.
5. Restored exact NPI reconstruction: `y_hat = P_q + c_q + s_q z_hat` when the reference is visible.
6. Preserved the NPI direct-target coordinate when the reference is missing after channel dropout.
7. Preserved curvature-aware speed+yaw displacement integration.
8. Preserved the zero-initialized innovation head.
9. Preserved best-validation-checkpoint evaluation.
10. Set the experiment revision to `normalized_physical_innovation_v3`.

## Removed active/historical proposal launch material

Later CSI-v4 / RA-NPI-v5 files and old top-level proposal launchers/documents are removed from the final release so the repository exposes only NPI-v3 as the proposal. Publication-backed baseline code and shared preprocessing utilities are retained.

## Tests

- targeted NPI/universal tests: 24 passed;
- complete retained test suite: 170 passed.
