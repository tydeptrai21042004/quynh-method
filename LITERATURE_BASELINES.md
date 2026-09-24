# Primary paper-supported baselines for SafeGrip-PFR-ECR

SafeGrip-PFR-ECR is the **only active proposal** in this repository. Its main prediction table compares `safegrip_pfr` only with the four friction-estimation methods below. `direct_gru_control` is retained strictly as an internal ablation/control and is written only to `pfr_component_ablation*.csv`.

| Benchmark key | Paper-supported method | Why it is comparable | Implementation status |
|---|---|---|---|
| `du2023_inceptiontime` | **Du2023 Dynamics-InceptionTime** | Maximum tire-road/pavement friction estimation from vehicle dynamics; LiRA/DTU research lineage | Dynamics-only InceptionTime paper-structure adaptation with the published multi-scale branches, residual grouping and global-average pooling. Vision is excluded for common-input fairness; no bit-exact source-code claim is made. |
| `todorovic2022_cnn` | **Todorovic2022 CNN** | Friction-potential estimation under longitudinal/lateral vehicle excitation | Paper-supported CNN/input-policy common-target adaptation. The publication supports CNN regression and the vehicle-signal inputs, but this repository does not claim an exact layer-by-layer reproduction. |
| `lampe2023_gru` | **Lampe2023 GRU** | Maximum tire-road friction coefficient from serial onboard vehicle sensors | Architecture-faithful two-layer 256-unit GRU; reported Adam/training and initialization settings are encoded for the source-settings protocol. |
| `levenberg2023_stft` | **Levenberg2023 vibration/STFT** | Tire-pavement grip estimation from vehicle vibration spectra; LiRA/DTU research environment | Method-structure adaptation: transverse acceleration -> STFT amplitude -> relative dB -> positive linear mapping. |

## Fair-comparison rules

All primary rows use the same continuous benchmark target, locked validation/test endpoint IDs, train-only preprocessing, and the same available vehicle-signal input policy. SafeGrip `mu_point` is used in the accuracy table; `mu_safe` is evaluated separately in the safety table.

Du et al.'s vision branch is intentionally not used because SafeGrip-PFR-ECR is sensor-only. The Todorovic source output is adapted to the common scalar friction target. Lampe is retrained on the common LiRA split rather than mixing the paper's original reported metrics with this benchmark.

### Levenberg sampling limitation

The original Levenberg vibration method uses substantially higher-rate transverse acceleration (about 250 Hz or higher) and analyzes high-frequency vibration content, including a reported 70--125 Hz range in one experiment. The common benchmark in this repository is resampled to 20 Hz, whose Nyquist frequency is 10 Hz. Therefore `levenberg2023_stft` is **not** described as a source-faithful frequency reproduction. It preserves the paper's signal-processing structure, selects a usable non-DC frequency using **training data only**, normalizes spectral amplitude in dB, and learns a non-negative linear mapping. Route-level smoothing is omitted to avoid leakage across locked benchmark endpoints.

The benchmark writes `paper_baseline_provenance.csv` with DOI, source task, fidelity level, source constraints, and common-protocol differences for every primary comparator.

## Primary citations

- Du et al. (2023), *Pavement Friction Evaluation Based on Vehicle Dynamics and Vision Data Using a Multi-Feature Fusion Network*, DOI `10.1177/03611981231165029`.
- Todorovic et al. (2022), *Neural Network Based Model for Friction Potential Estimation under Longitudinal and Lateral Excitations*, DOI `10.1088/1742-6596/2234/1/012005`.
- Lampe, Kortmann, and Westerkamp (2023), *Neural Network based Tire-Road Friction Estimation Using Experimental Data*, DOI `10.1016/j.ifacol.2023.12.056`.
- Levenberg (2023), *Estimating the Tire-Pavement Grip Potential From Vehicle Vibrations*, DOI `10.1177/03611981231152249`.
