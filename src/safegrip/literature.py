from __future__ import annotations

"""Literature-backed baseline registry.

The paper preset is intentionally restricted to methods that correspond to a
published tire/road-friction estimator.  The code records the fidelity of the
reimplementation because open datasets do not expose exactly the same sensor
channels as the source papers.  Reported paper numbers are NEVER mixed with our
cross-model test table unless the protocol is exactly reproduced.
"""

LITERATURE_BASELINES = {
    "todorovic2022_cnn": {
        "title": "Neural Network Based Model for Friction Potential Estimation under Longitudinal and Lateral Excitations",
        "authors": "Smiljana Todorovic; Andreas Wagner; Sven Müller; Jens Neubeck",
        "year": 2022,
        "venue": "Journal of Physics: Conference Series",
        "doi": "10.1088/1742-6596/2234/1/012005",
        "family": "temporal_cnn",
        "fidelity": "architecture-faithful adaptation; common LiRA inputs and scalar target",
        "source_constraints": "100-sample input; Conv1D 128/128/256 + MaxPool; Dense 400; scalar-output adaptation",
        "runnable": True,
    },
    "lampe2023_lstm": {
        "title": "Neural Network based Tire-Road Friction Estimation Using Experimental Data",
        "authors": "Nicolas Lampe; Karl-Philipp Kortmann; Clemens Westerkamp",
        "year": 2023,
        "venue": "IFAC-PapersOnLine",
        "doi": "10.1016/j.ifacol.2023.12.056",
        "family": "lstm",
        # The paper's selected LSTM uses two recurrent layers and a dense layer,
        # each with 256 neurons. We reproduce that architecture in paper mode.
        "fidelity": "architecture/training-faithful adaptation; common available sensor subset",
        "source_constraints": "2xLSTM(256)+Dense(256,tanh); Adam lr=1e-3; batch=64; 500 epochs; L2=1e-4; train-only MinMax; orthogonal recurrent/Glorot input-dense init",
        "runnable": True,
    },
    "lampe2023_gru": {
        "title": "Neural Network based Tire-Road Friction Estimation Using Experimental Data",
        "authors": "Nicolas Lampe; Karl-Philipp Kortmann; Clemens Westerkamp",
        "year": 2023,
        "venue": "IFAC-PapersOnLine",
        "doi": "10.1016/j.ifacol.2023.12.056",
        "family": "gru",
        # The paper's selected GRU uses two recurrent layers with 256 units.
        "fidelity": "architecture/training-faithful adaptation; common available sensor subset",
        "source_constraints": "2xGRU(256); Adam lr=1e-3; batch=64; 500 epochs; L2=1e-4; train-only MinMax; orthogonal recurrent/Glorot input-dense init",
        "runnable": True,
    },
    "schaefke2023_transformer": {
        "title": "Transformer Neural Networks for Maximum Friction Coefficient Estimation of Tire-Road Contact using Onboard Vehicle Sensors",
        "authors": "Hendrik Schäfke; Nicolas Lampe; Karl-Philipp Kortmann",
        "year": 2023,
        "venue": "IEEE Conference on Decision and Control",
        "doi": "10.1109/CDC49753.2023.10384175",
        "family": "transformer",
        "fidelity": "methodology-level adapted Transformer; exact source architecture not claimed",
        "source_constraints": "onboard-sensor Transformer family; unknown public architectural details selected only on validation",
        "runnable": True,
    },
    "chen2025_svdkl": {
        "title": "Data-Driven Tire-Road Friction Estimation for Electric-Wheel Vehicle With Data Category Selection and Uncertainty Evaluation",
        "authors": "Liang Chen; Zhaobo Qin; Yougang Bian; Manjiang Hu; Xiaoyan Peng",
        "year": 2025,
        "venue": "IEEE Transactions on Industrial Electronics",
        "doi": "10.1109/TIE.2024.3440510",
        "family": "spatiotemporal_cnn_sparse_variational_gp",
        "fidelity": "methodology-level adapted SV-DKL; source data-category-selection stage not reproduced",
        "source_constraints": "spatio-temporal feature learning + stochastic variational DKL + uncertainty; electric-wheel data-category selection unavailable under common LiRA inputs",
        "runnable": True,
    },
}

# Papers used only as contextual/literature references because the public inputs,
# hardware or exact protocol are not available in our open-data benchmark.
LITERATURE_ONLY = {
    "du2023_multimodal": {
        "title": "Pavement Friction Evaluation Based on Vehicle Dynamics and Vision Data Using a Multi-Feature Fusion Network",
        "doi": "10.1177/03611981231165029",
        "note": "LiRA-related multimodal vehicle-dynamics + vision study; do not mix reported R2 with our sensor-only direct table.",
    },
    "wang2025_ekfnet": {
        "title": "Fundamental Estimation for Tire Road Friction Coefficient: A Model-Based Learning Framework",
        "doi": "10.1109/TVT.2024.3464524",
        "note": "EKFNet/model-based learning; original state/vehicle setup is not directly reproducible from LiRA alone.",
    },
}

PAPER_BASELINES = tuple(k for k, v in LITERATURE_BASELINES.items() if v["runnable"])
QUICK_BASELINES = ("lampe2023_gru", "schaefke2023_transformer")


def validate_paper_baselines(names) -> None:
    missing = [n for n in names if n not in LITERATURE_BASELINES]
    if missing:
        raise ValueError(
            "Paper mode only accepts literature-backed baselines. Missing citation metadata for: "
            + ", ".join(missing)
        )
    no_doi = [n for n in names if not LITERATURE_BASELINES[n].get("doi")]
    if no_doi:
        raise ValueError("Baseline registry entries without DOI: " + ", ".join(no_doi))
