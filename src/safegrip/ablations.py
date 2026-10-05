from __future__ import annotations

from dataclasses import dataclass, replace


# NPI-v3 is the only active proposal.  Every other entry below is a controlled
# ablation of that proposal, not a competing proposal revision.
PROPOSAL_ABLATION = "npi_v3"


@dataclass(frozen=True)
class UniversalAblationConfig:
    # NPI coordinate. Supported values are implemented by
    # SemanticInnovationNormalizer in universal/innovation.py.
    innovation_mode: str = "npi_v3"
    localization_reference: str = "curvature"

    # Shared representation/backbone controls.
    physical_metadata: bool = True
    unit_metadata: bool = True
    sampling_rate_metadata: bool = True
    time_metadata: bool = True
    spectrum: bool = True
    aggregation: str = "latent_attention"
    sensor_dropout: float = 0.2
    learned_scale: bool = True
    dataset_id_conditioning: bool = False
    channel_id_embedding: bool = False


_BASE = UniversalAblationConfig()

# Primary paper ablations: isolate the mathematical NPI-v3 contribution first,
# then test the two strongest representation choices.
PRIMARY_ABLATIONS: tuple[str, ...] = (
    "direct_normalized",
    "reference_only",
    "old_physical_innovation",
    "npi_no_center",
    "npi_no_scale",
    "mean_pool",
    "channel_id_only",
    "npi_v3",
)

# Secondary/supplementary controls.  The path-length control differs from the
# proposal only for the displacement/localization target (D4).
SUPPLEMENTARY_ABLATIONS: tuple[str, ...] = (
    "no_physical_metadata",
    "no_spectrum",
    "no_sensor_dropout",
    "dataset_id_conditioning",
    "path_length_reference",
    "no_learned_scale",
)


ABLATIONS: dict[str, UniversalAblationConfig] = {
    # Only active proposal.
    "npi_v3": _BASE,

    # Core NPI-v3 mathematical controls.
    # Direct: same decoder and robust target coordinate, but never use P_q(X).
    "direct_normalized": replace(_BASE, innovation_mode="direct_normalized"),
    # Physics alone: use P_q(X) where observable and no learned correction.
    "reference_only": replace(_BASE, innovation_mode="reference_only"),
    # Previous proposal coordinate y_hat = P_q(X) + R_theta(X,q).
    "old_physical_innovation": replace(_BASE, innovation_mode="old_physical_innovation"),
    # Decompose robust residual normalization into location and scale effects.
    "npi_no_center": replace(_BASE, innovation_mode="npi_no_center"),
    "npi_no_scale": replace(_BASE, innovation_mode="npi_no_scale"),

    # Representation/backbone controls that directly support the paper claims.
    "mean_pool": replace(_BASE, aggregation="mean_pool"),
    "channel_id_only": replace(
        _BASE,
        physical_metadata=False,
        unit_metadata=False,
        sampling_rate_metadata=False,
        time_metadata=False,
        channel_id_embedding=True,
    ),
    "no_physical_metadata": replace(_BASE, physical_metadata=False),
    "no_spectrum": replace(_BASE, spectrum=False),
    "no_sensor_dropout": replace(_BASE, sensor_dropout=0.0),
    "dataset_id_conditioning": replace(_BASE, dataset_id_conditioning=True),
    "path_length_reference": replace(_BASE, localization_reference="path_length"),
    "no_learned_scale": replace(_BASE, learned_scale=False),
}


def get_ablation(name: str) -> UniversalAblationConfig:
    key = str(name).strip().lower()
    if key not in ABLATIONS:
        raise ValueError(f"unknown ablation: {name}. Choices: {', '.join(ABLATIONS)}")
    return ABLATIONS[key]
