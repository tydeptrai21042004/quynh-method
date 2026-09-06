# Literature-backed baseline policy

The direct numerical table contains only methods tied to published tire/road-friction work. All reported numbers are **our retrained results on the same LiRA target/split**, never copied paper numbers.

Because the source papers use different vehicles, sensors and targets, the benchmark uses one common LiRA sensor set and scalar road-friction reference. Fidelity is stated explicitly rather than calling every implementation an exact reproduction.

## Fair-comparison rules

- Same LiRA train/calibration/validation/test partition.
- Same available model input information; GPS remains matching/splitting metadata only.
- Model-specific temporal context is allowed.
- Validation/test **endpoints remain identical** across context lengths via a common warm-up.
- Train-only preprocessing follows the source paper where recoverable.
- Proposal and literature comparators receive the same Optuna trial budget.
- Primary validation selection metric is RMSE for every method.
- Final paper table uses five independent seeds and reports mean/std.
- Same-physics projection of baseline outputs is exported only as a supplementary parity control, never relabelled as the published method.

## Direct runnable baselines

### Todorovic et al. 2022 — CNN

**Smiljana Todorovic, Andreas Wagner, Sven Müller, Jens Neubeck.**  
*Neural Network Based Model for Friction Potential Estimation under Longitudinal and Lateral Excitations.*  
Journal of Physics: Conference Series 2234, 012005.  
DOI: `10.1088/1742-6596/2234/1/012005`

Paper-mode implementation preserves the recoverable source architecture:

```text
100-sample temporal input
-> Conv1D 128, k=14 -> MaxPool(2)
-> Conv1D 128, k=10 -> MaxPool(2)
-> Conv1D 256, k=10 -> MaxPool(2)
-> Flatten (3072 values at L=100)
-> Dense 400
-> scalar output
```

The source predicts longitudinal/lateral friction potentials and uses its own channel set. LiRA therefore changes the input-channel count and final output to the common scalar reference.

**Fidelity:** architecture-faithful adaptation; source training details not claimed exact where not recoverable.

### Lampe, Kortmann & Westerkamp 2023 — LSTM / GRU

*Neural Network based Tire-Road Friction Estimation Using Experimental Data.*  
IFAC-PapersOnLine 56(3), 397–402.  
DOI: `10.1016/j.ifacol.2023.12.056`

Recoverable source details used in paper mode:

- LSTM: two recurrent layers with 256 units plus a 256-unit `tanh` dense layer.
- GRU: two recurrent layers with 256 units.
- Adam, initial learning rate `1e-3`.
- batch size `64`.
- `500` epochs as source default.
- L2 regularization `1e-4`.
- train-only min-max normalization.
- orthogonal recurrent-weight initialization.
- Glorot input/dense initialization.
- original study trained each network structure five times and selected using validation RMSE.

LiRA does not provide the full original sensor set or maneuver segmentation. We preserve the architecture/initialization/preprocessing and tune the fixed-window history on validation under the harmonized benchmark.

**Fidelity:** architecture/preprocessing-faithful adaptation with source training defaults available in config.

### Schäfke, Lampe & Kortmann 2023 — Transformer

*Transformer Neural Networks for Maximum Friction Coefficient Estimation of Tire-Road Contact using Onboard Vehicle Sensors.*  
IEEE CDC 2023, pp. 5331–5338.  
DOI: `10.1109/CDC49753.2023.10384175`

The paper establishes the onboard-sensor Transformer method family, but the exact architecture used by the source is not treated here as publicly recoverable. The repository therefore tunes the adapted Transformer's hidden size/layers/heads/feed-forward multiplier only on validation.

**Fidelity:** methodology-level adapted Transformer; exact reproduction is not claimed.

### Chen et al. 2025 — adapted SV-DKL uncertainty comparator

**Liang Chen, Zhaobo Qin, Yougang Bian, Manjiang Hu, Xiaoyan Peng.**  
*Data-Driven Tire-Road Friction Estimation for Electric-Wheel Vehicle With Data Category Selection and Uncertainty Evaluation.*  
IEEE Transactions on Industrial Electronics 72(3), 3048–3060.  
DOI: `10.1109/TIE.2024.3440510`

The source method includes both:

1. a vehicle-state/parameter based data-category-selection scheme for stationary/nonstationary longitudinal maneuvers; and
2. stochastic variational deep-kernel learning for spatial-temporal feature mapping and uncertainty evaluation.

The common LiRA input protocol cannot reproduce the original electric-wheel-vehicle category-selection variables exactly. The runnable comparator therefore implements the spatio-temporal feature + SV-DKL part and states that limitation in `baseline_manifest.csv`.

**Fidelity:** methodology-level adapted SV-DKL; the full Chen method is not claimed reproduced.

## Literature-only references

Some important methods are discussed but excluded from the direct table when their inputs/model setup cannot be reproduced from the common benchmark:

- Du et al. 2023 multimodal vehicle-dynamics + vision method, DOI `10.1177/03611981231165029`.
- Wang et al. 2025 EKFNet/model-based learning framework, DOI `10.1109/TVT.2024.3464524`.

## Machine-readable provenance

Each paper run writes:

```text
results/lira_paper/baseline_manifest.csv
results/lira_paper/baseline_selected_hparams.json
results/lira_paper/evaluation_protocol.json
results/lira_paper/metrics_by_seed.csv
results/lira_paper/projection_control_metrics.csv
```
