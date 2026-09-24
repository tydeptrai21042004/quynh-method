# SafeGrip-PFR-ECR baseline implementation change

## Active proposal

`SafeGrip-PFR-ECR` (`safegrip_pfr`) is the only proposal exposed by the CLI. The backward-compatible `--proposal` and `--method` flags accept only `pfr`.

## Primary paper-supported baselines

The main benchmark registry now contains exactly:

1. `du2023_inceptiontime` — Du2023 Dynamics-InceptionTime (dynamics-only branch for common-input fairness)
2. `todorovic2022_cnn` — Todorovic2022 CNN
3. `lampe2023_gru` — Lampe2023 GRU
4. `levenberg2023_stft` — Levenberg2023 vibration/STFT low-rate common-input adaptation

`direct_gru_control` is kept only in `pfr_component_ablation*.csv`; it is removed from the primary `metrics*.csv` table.

## New/changed outputs

- `paper_baseline_provenance.csv` records DOI, task, fidelity, source constraints and common-protocol differences.
- The main metrics table contains only SafeGrip-PFR-ECR plus the four primary paper baselines.
- PFR safety and component-ablation outputs remain separate.

## Fairness note

The Levenberg source method uses high-rate transverse vibration data (~250 Hz or higher) and high-frequency spectral content. The repository common benchmark is 20 Hz. Therefore this comparator is explicitly labelled a low-rate method-structure adaptation rather than a source-faithful frequency reproduction. Its usable frequency is selected from training data only.

## Validation

- Unit/integration tests: `109 passed`.
- Integration tests verify the proposal + four requested baseline paths without exposing a generated benchmark dataset.
- `direct_gru_control` was verified to remain only in the ablation output.
