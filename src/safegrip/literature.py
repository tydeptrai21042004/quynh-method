from __future__ import annotations

"""Paper-backed comparator registry for the closed D1--D4 benchmark.

The registry separates *paper provenance* from *local implementation status*.
A comparator may be accepted as a benchmark target even when this repository
expects the authors' official implementation or a dedicated reproduction
wrapper rather than silently substituting a generic neural network.
"""

LITERATURE_BASELINES = {
    "du2023_inceptiontime": {
        "dataset": "lira_cd",
        "display_name": "Du2023 Dynamics-InceptionTime",
        "title": "Pavement Friction Evaluation Based on Vehicle Dynamics and Vision Data Using a Multi-Feature Fusion Network",
        "authors": "Zhao Du; Asmus Skar; Matteo Pettinari; Xingyi Zhu",
        "year": 2023,
        "doi": "10.1177/03611981231165029",
        "family": "inception_time",
        "fidelity": "paper-structure dynamics-only common-input adaptation",
        "exact_dataset": True,
        "runnable": True,
        "metrics": ("r2", "rmse", "mae"),
    },
    "levenberg2023_stft": {
        "dataset": "lira_cd",
        "display_name": "Levenberg2023 vibration/STFT",
        "title": "Estimating the Tire-Pavement Grip Potential From Vehicle Vibrations",
        "authors": "Eyal Levenberg",
        "year": 2023,
        "doi": "10.1177/03611981231152249",
        "family": "vibration_stft_linear",
        "fidelity": "paper-supported low-rate adaptation; not a source-frequency reproduction",
        "exact_dataset": True,
        "runnable": True,
        "metrics": ("r2", "rmse", "mae"),
        "limitation": "LiRA prepared signals are low-rate, so the paper's 70--125 Hz band is not claimed reproduced.",
    },
    "mendoza2019_fuzzy": {
        "dataset": "uc3m_tire",
        "display_name": "Mendoza-Petit2019 Fuzzy/Pacejka",
        "title": "A Strain-Based Method to Estimate Tire Parameters for Intelligent Tires under Complex Maneuvering Operations",
        "authors": "M. F. Mendoza-Petit; D. Garcia-Pozuelo; V. Diaz; O. Olatunbosun",
        "year": 2019,
        "doi": "10.3390/s19132973",
        "family": "fuzzy_tire_estimator",
        "fidelity": "paper-structured local fuzzy reproduction; exact original MATLAB FIS parameters are not claimed",
        "exact_dataset": True,
        "runnable": True,
        "implementation_kind": "paper_structured_local_reproduction",
        "supported_targets": ("force_x", "force_y", "force_z", "slip_angle"),
        "metrics": ("normalized_error", "rmse"),
    },
    "yunta2018_fuzzy_lfc": {
        "dataset": "uc3m_tire",
        "display_name": "Yunta2018 Fuzzy LFC",
        "title": "A Strain-Based Method to Detect Tires' Loss of Grip and Estimate Lateral Friction Coefficient from Experimental Data by Fuzzy Logic for Intelligent Tire Development",
        "authors": "J. Yunta et al.",
        "year": 2018,
        "doi": "10.3390/s18020490",
        "family": "fuzzy_lateral_friction",
        "fidelity": "paper-structured local fuzzy reproduction; related experimental lineage, not exact U6ICRX archive reproduction",
        "exact_dataset": False,
        "runnable": True,
        "implementation_kind": "paper_structured_local_reproduction",
        "supported_targets": ("force_y", "force_z", "slip_angle"),
        "metrics": ("relative_error",),
    },
    "chrosniak2024_ddm": {
        "dataset": "deep_dynamics_iac",
        "display_name": "Chrosniak2024 Deep Dynamics (DDM)",
        "title": "Deep Dynamics: Vehicle Dynamics Modeling With a Physics-Constrained Neural Network for Autonomous Racing",
        "authors": "John Chrosniak; Jingyun Ning; Madhur Behl",
        "year": 2024,
        "doi": "10.1109/LRA.2024.3388847",
        "family": "deep_dynamics",
        "fidelity": "local Physics-Guard + single-track/Pacejka structural reproduction; official upstream implementation remains preferred for source-faithful runs",
        "exact_dataset": True,
        "runnable": True,
        "implementation_kind": "paper_structured_local_reproduction",
        "supported_targets": ("velocity_x", "velocity_y", "yaw_rate"),
        "official_repo": "https://github.com/linklab-uva/deep-dynamics",
        "metrics": ("rmse_vx", "rmse_vy", "rmse_yaw", "ade", "fde"),
    },
    "fang_yu2025_fthd": {
        "dataset": "deep_dynamics_iac",
        "display_name": "FangYu2025 FTHD/EKF-FTHD",
        "title": "Fine-tuning hybrid dynamics with physics-informed neural networks for vehicle dynamics estimation",
        "authors": "Fang; Yu et al.",
        "year": 2025,
        "doi": "10.1007/s41315-025-00452-4",
        "family": "fthd_ekf_fthd",
        "fidelity": "local DDM-derived hybrid physics/data fine-tuning reproduction; official upstream implementation remains preferred for source-faithful runs",
        "exact_dataset": True,
        "runnable": True,
        "implementation_kind": "paper_structured_local_reproduction",
        "supported_targets": ("velocity_x", "velocity_y", "yaw_rate"),
        "official_repo": "https://github.com/Binghamton-ACSR-Lab/FTHD",
        "metrics": ("rmse_vx", "rmse_vy", "rmse_yaw", "max_error"),
    },
    "onyekpe2021_qgru": {
        "dataset": "io_vnbd",
        "display_name": "Onyekpe2021 QGRU/GRU",
        "title": "A Quaternion Gated Recurrent Unit Neural Network for Sensor Fusion",
        "authors": "U. Onyekpe; V. Palade; S. Kanarachos",
        "year": 2021,
        "doi": "10.3390/info12030117",
        "family": "qgru",
        "fidelity": "local quaternion-GRU reproduction using Hamilton-product recurrent maps; exact original training weights are not claimed",
        "exact_dataset": True,
        "runnable": True,
        "implementation_kind": "paper_structured_local_reproduction",
        "supported_targets": ("displacement", "orientation"),
        "metrics": ("crse_displacement", "crse_orientation"),
    },
    "wang2023_transformer": {
        "dataset": "io_vnbd",
        "display_name": "Wang2023 Transformer/WhONet/LSTM",
        "title": "Wheel Odometry with Deep Learning-Based Error Prediction Model for Vehicle Localization",
        "authors": "Wang et al.",
        "year": 2023,
        "doi": "10.3390/app13095588",
        "family": "transformer_wheel_odometry",
        "fidelity": "local Transformer wheel-odometry error-prediction reproduction; exact original training weights are not claimed",
        "exact_dataset": True,
        "runnable": True,
        "implementation_kind": "paper_structured_local_reproduction",
        "supported_targets": ("displacement", "orientation"),
        "metrics": ("crse_displacement", "crse_orientation"),
    },
}

DATASET_BASELINES = {
    "lira_cd": ("du2023_inceptiontime", "levenberg2023_stft"),
    "uc3m_tire": ("mendoza2019_fuzzy", "yunta2018_fuzzy_lfc"),
    "deep_dynamics_iac": ("chrosniak2024_ddm", "fang_yu2025_fthd"),
    "io_vnbd": ("onyekpe2021_qgru", "wang2023_transformer"),
}
PAPER_BASELINES = tuple(name for ds in DATASET_BASELINES.values() for name in ds)

# Compatibility for the legacy LiRA-only benchmark engine.  This is deliberately
# not the public paper registry and contains only the two selected D1 baselines.
LEGACY_EXECUTABLE_BASELINES = DATASET_BASELINES["lira_cd"]
QUICK_BASELINES = LEGACY_EXECUTABLE_BASELINES
LITERATURE_ONLY = {name: meta for name, meta in LITERATURE_BASELINES.items() if not meta["runnable"]}


def baselines_for_dataset(dataset: str) -> tuple[str, ...]:
    key = str(dataset).strip().lower()
    if key not in DATASET_BASELINES:
        raise ValueError(f"unknown paper dataset: {dataset}. Choices: {', '.join(DATASET_BASELINES)}")
    return DATASET_BASELINES[key]


def validate_paper_baselines(names, dataset: str | None = None, *, require_runnable: bool = False) -> None:
    names = list(names)
    unknown = [n for n in names if n not in LITERATURE_BASELINES]
    if unknown:
        raise ValueError("Unknown/non-paper baseline(s): " + ", ".join(unknown))
    if dataset is not None:
        allowed = set(baselines_for_dataset(dataset))
        wrong = [n for n in names if n not in allowed]
        if wrong:
            raise ValueError(
                f"Baseline(s) {', '.join(wrong)} are not allowed for dataset {dataset}; "
                f"allowed: {', '.join(sorted(allowed))}"
            )
    invalid = [n for n in names if not LITERATURE_BASELINES[n].get("doi")]
    if invalid:
        raise ValueError("Baseline registry entries without DOI: " + ", ".join(invalid))
    if require_runnable:
        missing = [n for n in names if not LITERATURE_BASELINES[n].get("runnable", False)]
        if missing:
            raise ValueError(
                "Baseline provenance is registered, but local reproduction is not implemented for: "
                + ", ".join(missing)
            )


def validate_source_settings(names) -> None:
    """Only locally runnable D1 adaptations have source-setting support here."""
    validate_paper_baselines(names, require_runnable=True)
