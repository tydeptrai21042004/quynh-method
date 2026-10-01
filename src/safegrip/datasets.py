from __future__ import annotations

"""Closed paper benchmark registry for Universal SafeGrip.

Only the four datasets used by the final research protocol are public/runnable
through the registry.  Older exploratory datasets remain outside this registry
so they cannot accidentally enter paper tables or command-line sweeps.
"""

DATASET_REGISTRY = {
    "lira_cd": {
        "id": "D1",
        "title": "LiRA-CD platoon friction test",
        "task": "road_friction_regression",
        "targets": ("friction",),
        "dataset_doi": "10.11583/DTU.23096600.v1",
        "license": "CC BY 4.0",
        "download_kind": "figshare",
        "input_modalities": ("vehicle_dynamics", "imu", "can", "gps"),
        "allowed_baselines": ("du2023_inceptiontime", "todorovic2022_cnn", "lampe2023_gru", "levenberg2023_stft"),
    },
    "uc3m_tire": {
        "id": "D2",
        "title": "UC3M/Birmingham strain-based intelligent tire data",
        "task": "tire_mechanics",
        "targets": ("force_x", "force_y", "force_z", "slip_angle"),
        "dataset_doi": "10.21950/U6ICRX",
        "license": "see e-cienciaDatos record",
        "download_kind": "dataverse",
        "input_modalities": ("tire_strain",),
        "allowed_baselines": ("mendoza2019_fuzzy", "yunta2018_fuzzy_lfc"),
        "provenance_note": (
            "Mendoza2019 is the exact dataset-generating paper. Yunta2018 is a "
            "related UC3M/Birmingham experimental-lineage comparator and is not "
            "claimed to use the exact U6ICRX deposited files."
        ),
    },
    "deep_dynamics_iac": {
        "id": "D3",
        "title": "Deep Dynamics Indy Autonomous Challenge real-vehicle data",
        "task": "vehicle_dynamics_state_prediction",
        "targets": ("velocity_x", "velocity_y", "yaw_rate"),
        "dataset_doi": None,
        "paper_doi": "10.1109/LRA.2024.3388847",
        "license": "upstream repository GPL-3.0; verify bundled data terms",
        "download_kind": "github",
        "input_modalities": ("vehicle_state", "steering", "drivetrain_control"),
        "allowed_baselines": ("chrosniak2024_ddm", "fang_yu2025_fthd"),
    },
    "io_vnbd": {
        "id": "D4",
        "title": "IO-VNBD inertial and odometry vehicle navigation benchmark",
        "task": "vehicle_localization",
        "targets": ("displacement",),
        "dataset_doi": "10.1016/j.dib.2021.106885",
        "license": "open dataset; see upstream repository/article",
        "download_kind": "github",
        "input_modalities": ("ins", "wheel_odometry", "vehicle_ego_motion"),
        "allowed_baselines": ("onyekpe2021_qgru", "onyekpe2021_whonet"),
    },
}

PAPER_DATASETS = tuple(DATASET_REGISTRY)
PRIMARY_FRICTION_DATASETS = ("lira_cd",)
AUXILIARY_REAL_DATASETS: tuple[str, ...] = ()


def validate_dataset(name: str) -> str:
    key = str(name).strip().lower()
    if key not in DATASET_REGISTRY:
        raise ValueError(f"unknown paper dataset: {name}. Choices: {', '.join(PAPER_DATASETS)}")
    return key
