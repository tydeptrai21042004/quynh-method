from __future__ import annotations

"""Verified paper-supported comparators for SafeGrip-PFR-ECR.

The registry is authoritative: only these four names may enter the primary
baseline comparison.  ``fidelity`` deliberately distinguishes a source-paper
reproduction from a common-input/common-target adaptation.  We do not claim
bit-exact reproduction where the publication or the common LiRA sampling
policy does not support it.
"""

LITERATURE_BASELINES = {
    "du2023_inceptiontime": {
        "display_name": "Du2023 Dynamics-InceptionTime",
        "title": "Pavement Friction Evaluation Based on Vehicle Dynamics and Vision Data Using a Multi-Feature Fusion Network",
        "authors": "Zhao Du; Asmus Skar; Matteo Pettinari; Xingyi Zhu",
        "year": 2023,
        "venue": "Transportation Research Record",
        "doi": "10.1177/03611981231165029",
        "family": "inception_time",
        "task": "maximum tire-road/pavement friction estimation from vehicle dynamics; dynamics branch used here",
        "fidelity": "paper-structure dynamics-only common-input adaptation",
        "source_constraints": (
            "paper describes an InceptionTime dynamics branch with six residual blocks of three Inception modules, "
            "multi-scale Conv1D kernels 40/20/10, 1x1 bottleneck/max-pool branches, global-average pooling and a final dense head; "
            "source training uses SGD lr=1e-3, momentum=0.98, early stopping patience 20, and validation-driven LR reduction"
        ),
        "common_benchmark_note": (
            "vision is intentionally excluded so all primary methods receive vehicle-signal inputs only; "
            "the code follows the paper text as six residual blocks of three Inception modules (18 modules total); the 50 m source sampling/feature policy is adapted to the locked common LiRA input protocol"
        ),
        "source_settings_available": True,
        "paper_verified": True,
        "runnable": True,
    },
    "todorovic2022_cnn": {
        "display_name": "Todorovic2022 CNN adaptation",
        "title": "Neural Network Based Model for Friction Potential Estimation under Longitudinal and Lateral Excitations",
        "authors": "Smiljana Todorovic; Andreas Wagner; Sven Mueller; Jens Neubeck",
        "year": 2022,
        "venue": "Journal of Physics: Conference Series",
        "doi": "10.1088/1742-6596/2234/1/012005",
        "family": "temporal_cnn",
        "task": "tire-road friction-potential estimation under longitudinal and lateral excitation using experimental 4WD vehicle data",
        "fidelity": "paper-supported CNN/input-policy common-target adaptation; exact published layer topology is not claimed",
        "source_constraints": (
            "the paper supports a one-dimensional CNN friction-potential regressor using longitudinal/lateral acceleration, vehicle velocity, "
            "wheel speeds, tire slip angle, brake/engine torque and steering angle; its source input uses the previous 99 time steps "
            "(about 3 s) and predicts separate longitudinal/lateral friction potentials; exact table values that cannot be read reliably "
            "from the accessible text are not invented by this repository"
        ),
        "common_benchmark_note": (
            "the benchmark CNN preserves the one-dimensional temporal-CNN method family and approximately 3 s physical horizon, "
            "but adapts the source two-output friction-potential target to the common scalar LiRA friction target and common sensor subset; "
            "it must not be described as a bit-exact architecture reproduction"
        ),
        "source_settings_available": False,
        "paper_verified": True,
        "runnable": True,
    },
    "lampe2023_gru": {
        "display_name": "Lampe2023 GRU",
        "title": "Neural Network based Tire-Road Friction Estimation Using Experimental Data",
        "authors": "Nicolas Lampe; Karl-Philipp Kortmann; Clemens Westerkamp",
        "year": 2023,
        "venue": "IFAC-PapersOnLine",
        "doi": "10.1016/j.ifacol.2023.12.056",
        "family": "gru",
        "task": "maximum tire-road friction coefficient estimation from serial onboard vehicle sensors",
        "fidelity": "architecture-faithful GRU common-input adaptation with source training settings available",
        "source_constraints": (
            "2 stacked GRU layers with 256 units; source training uses Adam lr=1e-3, batch 64, 500 epochs, "
            "L2=1e-4, orthogonal recurrent-weight initialization and Glorot input/dense initialization"
        ),
        "common_benchmark_note": "uses the common LiRA sensor subset, split and target rather than the source test-vehicle dataset",
        "source_settings_available": True,
        "paper_verified": True,
        "runnable": True,
    },
    "levenberg2023_stft": {
        "display_name": "Levenberg2023 vibration/STFT adaptation",
        "title": "Estimating the Tire-Pavement Grip Potential From Vehicle Vibrations",
        "authors": "Eyal Levenberg",
        "year": 2023,
        "venue": "Transportation Research Record",
        "doi": "10.1177/03611981231152249",
        "family": "vibration_stft_linear",
        "task": "tire-pavement grip-potential estimation from transverse vehicle-vibration spectra",
        "fidelity": "paper-supported method-structure low-rate adaptation; not a source-frequency reproduction",
        "source_constraints": (
            "source method uses high-rate transverse acceleration, short-time Fourier spectral amplitudes, smoothing, "
            "and a positive linear relation to grip potential"
        ),
        "common_benchmark_note": (
            "LiRA is resampled to 20 Hz, so the source 70-125 Hz high-frequency vibration band cannot be reproduced; "
            "the implementation uses training-only frequency selection below the common Nyquist limit and is reported as an adaptation"
        ),
        "source_settings_available": False,
        "paper_verified": True,
        "runnable": True,
    },
}

PAPER_BASELINES = (
    "du2023_inceptiontime",
    "todorovic2022_cnn",
    "lampe2023_gru",
    "levenberg2023_stft",
)
QUICK_BASELINES = PAPER_BASELINES
LITERATURE_ONLY = {}


def validate_paper_baselines(names) -> None:
    names = list(names)
    unknown = [n for n in names if n not in LITERATURE_BASELINES]
    if unknown:
        raise ValueError(
            "Only verified paper-supported friction baselines are allowed: "
            + ", ".join(PAPER_BASELINES)
            + ". Unknown/non-primary: "
            + ", ".join(unknown)
        )
    invalid = [n for n in names if not LITERATURE_BASELINES[n].get("paper_verified") or not LITERATURE_BASELINES[n].get("doi")]
    if invalid:
        raise ValueError("Unverified baseline registry entries: " + ", ".join(invalid))


def validate_source_settings(names) -> None:
    """Reject a source-settings claim where the paper cannot be reproduced here."""
    validate_paper_baselines(names)
    unsupported = [n for n in names if not LITERATURE_BASELINES[n].get("source_settings_available", False)]
    if unsupported:
        raise ValueError(
            "Source-settings protocol is unavailable for: " + ", ".join(unsupported)
            + ". Use protocol=controlled and report these methods as adaptations."
        )
