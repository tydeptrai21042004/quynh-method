# Paper-baseline implementation audit

The primary baseline registry contains exactly four paper-supported friction/grip methods.
No generic Transformer, SV-DKL, LSTM duplicate, constant predictor, or direct-GRU control is
allowed into `metrics.csv` as a primary baseline. `direct_gru_control` remains only in the
PFR ablation table.

| Registry ID | Paper support | Implementation status | Allowed in primary table |
|---|---|---|---|
| `du2023_inceptiontime` | Du et al., TRR 2023, DOI 10.1177/03611981231165029 | Dynamics-only InceptionTime paper-structure adaptation; verified kernels/branches and optimizer schedule; vision intentionally excluded | Yes |
| `todorovic2022_cnn` | Todorovic et al., JPCS 2022, DOI 10.1088/1742-6596/2234/1/012005 | Paper-supported CNN/input-policy adaptation. Exact layer topology is **not** claimed because it was not verifiable from the accessible source text used for this audit | Yes |
| `lampe2023_gru` | Lampe et al., IFAC-PapersOnLine 2023, DOI 10.1016/j.ifacol.2023.12.056 | Two-layer 256-unit GRU with source optimizer/training settings available; common LiRA inputs/target remain an adaptation | Yes |
| `levenberg2023_stft` | Levenberg, TRR 2023, DOI 10.1177/03611981231152249 | Low-rate method-structure adaptation only. LiRA 20 Hz cannot reproduce the source high-frequency vibration analysis | Yes, but must be labelled adaptation |

## Protocol rule

`controlled` is the main fair-comparison protocol: same locked endpoints and training budget.
`source-faithful`/source-settings mode is now rejected for baselines for which the source
settings cannot be reproduced honestly (`todorovic2022_cnn`, `levenberg2023_stft`).

## Removed from baseline code/configuration

- `lampe2023_lstm` (not one of the selected four primary comparators)
- `schaefke2023_transformer` generic adaptation
- `chen2025_svdkl` partial adaptation and its `svdkl.py` implementation
- generic Transformer / spatio-temporal CNN baseline classes

Internal ablations are not baselines and remain separated in `pfr_component_ablation.csv`.
