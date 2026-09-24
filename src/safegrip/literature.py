from __future__ import annotations

"""Primary paper-supported comparators for SafeGrip-PFR-ECR.

Only methods addressing tire/road or pavement friction/grip estimation are
exposed as primary baselines.  Every implementation is evaluated on the same
LiRA target and locked endpoints as SafeGrip-PFR-ECR.  The fidelity field is
explicit because common-input benchmarking necessarily changes some source
paper inputs/protocols.
"""

LITERATURE_BASELINES = {
    "du2023_inceptiontime": {
        "display_name": "Du2023 Dynamics-InceptionTime",
        "title": "Pavement Friction Evaluation Based on Vehicle Dynamics and Vision Data Using a Multi-Feature Fusion Network",
        "authors": "Yang Du et al.",
        "year": 2023,
        "venue": "Transportation Research Record",
        "doi": "10.1177/03611981231165029",
        "family": "inception_time",
        "task": "maximum tire-road/pavement friction estimation from vehicle dynamics; dynamics branch used here",
        "fidelity": "architecture-faithful dynamics-only common-input adaptation",
        "source_constraints": (
            "InceptionTime vehicle-dynamics branch with six Inception modules grouped into two residual blocks; "
            "multi-scale temporal convolutions, residual connections and global-average pooling; source training "
            "uses SGD lr=1e-3, momentum=0.98, validation early stopping and LR reduction"
        ),
        "common_benchmark_note": "vision is intentionally excluded so every primary method receives the same vehicle-signal inputs",
        "runnable": True,
    },
    "todorovic2022_cnn": {
        "display_name": "Todorovic2022 CNN",
        "title": "Neural Network Based Model for Friction Potential Estimation under Longitudinal and Lateral Excitations",
        "authors": "Smiljana Todorovic; Andreas Wagner; Sven Müller; Jens Neubeck",
        "year": 2022,
        "venue": "Journal of Physics: Conference Series",
        "doi": "10.1088/1742-6596/2234/1/012005",
        "family": "temporal_cnn",
        "task": "tire-road friction-potential estimation from longitudinal/lateral vehicle excitation",
        "fidelity": "architecture-faithful common-target adaptation",
        "source_constraints": "100-sample input; Conv1D 128/128/256 + MaxPool stages; Dense 400; final scalar-output adaptation",
        "common_benchmark_note": "source friction-potential output is mapped to the common continuous LiRA friction target",
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
        "fidelity": "architecture- and reported-training-setting common-input adaptation",
        "source_constraints": "2xGRU(256); Adam lr=1e-3; batch=64; 500 epochs; L2=1e-4; train-only MinMax; orthogonal recurrent/Glorot input initialization",
        "common_benchmark_note": "uses the common LiRA sensor subset, split and target rather than the source vehicle dataset",
        "runnable": True,
    },
    "levenberg2023_stft": {
        "display_name": "Levenberg2023 vibration/STFT",
        "title": "Estimating the Tire-Pavement Grip Potential From Vehicle Vibrations",
        "authors": "Eyal Levenberg et al.",
        "year": 2023,
        "venue": "Transportation Research Record",
        "doi": "10.1177/03611981231152249",
        "family": "vibration_stft_linear",
        "task": "tire-pavement grip/friction-potential estimation from transverse vehicle vibration spectra",
        "fidelity": "method-structure common-input low-rate STFT adaptation; not a source-faithful frequency reproduction",
        "source_constraints": (
            "source method uses high-rate transverse acceleration (about 250 Hz or higher), short-time Fourier analysis, "
            "relative spectral amplitude in dB, smoothing, and a positive linear mapping to grip potential"
        ),
        "common_benchmark_note": (
            "the common LiRA benchmark is resampled to 20 Hz, so the source 70-125 Hz vibration band is unavailable; "
            "frequency selection is performed on training data only below the common Nyquist limit and smoothing across locked endpoints is omitted to avoid leakage"
        ),
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
    missing = [n for n in names if n not in LITERATURE_BASELINES]
    if missing:
        raise ValueError(
            "SafeGrip-PFR-ECR accepts only the registered friction-estimation paper baselines: "
            + ", ".join(PAPER_BASELINES)
            + ". Unknown: "
            + ", ".join(missing)
        )
    no_doi = [n for n in names if not LITERATURE_BASELINES[n].get("doi")]
    if no_doi:
        raise ValueError("Baseline registry entries without DOI: " + ", ".join(no_doi))
