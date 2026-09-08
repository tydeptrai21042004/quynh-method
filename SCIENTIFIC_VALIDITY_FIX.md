# LiRA scientific-validity fix (v0.4.0)

This revision fixes the failure discovered in the September 2026 Kaggle development run.

## Root cause

The public LiRA-CD article states that selected CAN channels require an additional offset/resolution translation. The previous parser treated the stored values in `task_7505_acc_lon.txt` and `task_7505_acc_trans.txt` directly as m/s². In the failed run their training means were approximately 396 and 65,537, so the mechanics lower bound saturated at 2.0 for every test endpoint. Conformal calibration then converted that broken constant into 0.6055, causing every SafeGrip prediction to be projected to the same value.

## Corrections

- Implements LiRA-CD Table-2 source translations for longitudinal/transverse acceleration, yaw, electric brake torque, and estimated wheel torque.
- Uses the documented encoded zero point to identify the correct numeric channel when a TXT export has generic column names.
- Rejects implausible mandatory physical signals instead of silently continuing.
- Drops optional encoded/degenerate model features and exports `lira_signal_audit.csv`.
- Removes the arbitrary 2.0 cap from the mechanics lower bound.
- Fails preprocessing if the mechanics lower bound exceeds `mu_upper` on more than the configured tolerance; calibration cannot hide this error.
- Exports `lira_physics_audit.json`.
- Adds train-mean/train-median sanity baselines.
- Adds `result_health.json` with constant-prediction, projection-dominance, train-mean, R², sample-size, and physics-support gates.
- Adds a `trust` preset: three seeds, full literature architectures/preprocessing, 30-epoch practical cap.

## Interpretation

A `PASS` result is not manufactured. If SafeGrip does not beat the train-only mean or remains projection-dominated, the run is labelled `REVIEW`. This is deliberate: the repository should tell you when the evidence is not strong enough rather than optimizing code to guarantee a favorable result.

## v0.5.0 follow-up

The v0.4.0 LiRA decoding fix remains unchanged. v0.5.0 additionally makes the temporal/partial-identification implementation match the theory: discontinuous matched traces are segmented, the physical lower endpoint is a fixed trailing-window maximum, calibration labels are excluded from gradient training, and UQ metrics use the raw Gaussian center. See `FINAL_RESEARCH_RELEASE.md`.
