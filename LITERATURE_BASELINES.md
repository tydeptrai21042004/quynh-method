# Literature-backed baseline policy

The **direct numerical baseline table contains no ad-hoc generic models**. A runnable baseline must have a paper title and DOI in `src/safegrip/literature.py`.

The point is not to pretend we reproduced proprietary experiments exactly. All methods are retrained on the same open-data feature/target/split protocol, and `baseline_manifest.csv` states the reproduction fidelity.

## Direct runnable baselines

### Todorovic et al. 2022 — CNN

**Smiljana Todorovic, Andreas Wagner, Sven Müller, Jens Neubeck.**  
*Neural Network Based Model for Friction Potential Estimation under Longitudinal and Lateral Excitations.*  
Journal of Physics: Conference Series 2234, 012005.  
DOI: `10.1088/1742-6596/2234/1/012005`

The source work estimates friction potential under longitudinal and lateral excitation with a neural regression approach. The repository implements a temporal CNN in this paper family on the harmonized open production-sensor window.

**Fidelity:** methodology-level, not claimed as bit-for-bit reproduction.

### Lampe, Kortmann & Westerkamp 2023 — LSTM

*Neural Network based Tire-Road Friction Estimation Using Experimental Data.*  
IFAC-PapersOnLine 56(3), 397–402.  
DOI: `10.1016/j.ifacol.2023.12.056`

The code reproduces the selected recurrent architecture at architecture level: two recurrent LSTM layers with 256 hidden units and the documented dense hidden stage. Inputs are restricted to the common open-sensor protocol.

### Lampe, Kortmann & Westerkamp 2023 — GRU

Same paper/DOI. The paper evaluates recurrent alternatives; the repository includes the two-layer 256-unit GRU variant as a separately reported baseline.

### Schäfke, Lampe & Kortmann 2023 — Transformer

*Transformer Neural Networks for Maximum Friction Coefficient Estimation of Tire-Road Contact using Onboard Vehicle Sensors.*  
IEEE CDC 2023, pp. 5331–5338.  
DOI: `10.1109/CDC49753.2023.10384175`

The paper applies a Transformer to onboard-sensor time series and reports comparison with UKF and prior recurrent estimators. The open benchmark implements the Transformer method family on the common feature window.

**Fidelity:** methodology-level because the exact original simulated/experimental sensor protocol is not the LiRA protocol.

### Chen et al. 2025 — SV-DKL uncertainty baseline

**Liang Chen, Zhaobo Qin, Yougang Bian, Manjiang Hu, Xiaoyan Peng.**  
*Data-Driven Tire-Road Friction Estimation for Electric-Wheel Vehicle With Data Category Selection and Uncertainty Evaluation.*  
IEEE Transactions on Industrial Electronics 72(3), 3048–3060.  
DOI: `10.1109/TIE.2024.3440510`

The source method constructs spatio-temporal features and uses stochastic variational deep-kernel learning for friction estimation and uncertainty evaluation. The repository provides a spatio-temporal CNN feature extractor plus sparse variational GP (`gpytorch`).

**Fidelity:** methodology-level common-sensor reimplementation.

## Literature-only references

Some important methods should be discussed but not placed in the direct table when their inputs/model parameters cannot be faithfully recovered from the open benchmark:

- Du et al. 2023 multimodal LiRA-related vehicle-dynamics + vision method, DOI `10.1177/03611981231165029`.
- Wang et al. 2025 EKFNet/model-based learning framework, DOI `10.1109/TVT.2024.3464524`.

Their published numbers are not copied into `metrics.csv`.

## Why generic Ridge/RF/XGBoost/MLP are absent

They can be useful software sanity checks, but the user requested paper-backed scientific baselines only. They are therefore not part of the paper preset and not named as competing methods in the result table.

## Machine-readable provenance

Each paper run writes:

```text
results/lira_paper/baseline_manifest.csv
results/lira_paper/literature_only.json
```

The manifest contains title, authors, year, venue, DOI, method family and reproduction fidelity for every direct baseline.
