from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class UniversalAblationConfig:
    physical_metadata: bool = True
    unit_metadata: bool = True
    sampling_rate_metadata: bool = True
    time_metadata: bool = True
    spectrum: bool = True
    aggregation: str = "latent_attention"
    sensor_dropout: float = 0.2
    mechanics_queries: bool = True
    physics_consistency: bool = True
    friction_inequality: bool = True
    learned_scale: bool = True
    dataset_id_conditioning: bool = False
    channel_id_embedding: bool = False


ABLATIONS: dict[str, UniversalAblationConfig] = {
    "full": UniversalAblationConfig(),
    "no_physical_metadata": replace(UniversalAblationConfig(), physical_metadata=False),
    "no_unit_metadata": replace(UniversalAblationConfig(), unit_metadata=False),
    "no_sampling_rate_metadata": replace(UniversalAblationConfig(), sampling_rate_metadata=False),
    "no_time_metadata": replace(UniversalAblationConfig(), time_metadata=False),
    "no_spectrum": replace(UniversalAblationConfig(), spectrum=False),
    "mean_pool": replace(UniversalAblationConfig(), aggregation="mean_pool"),
    "no_sensor_dropout": replace(UniversalAblationConfig(), sensor_dropout=0.0),
    "no_mechanics_queries": replace(UniversalAblationConfig(), mechanics_queries=False),
    "no_physics_consistency": replace(UniversalAblationConfig(), physics_consistency=False),
    "no_friction_inequality": replace(UniversalAblationConfig(), friction_inequality=False),
    "no_learned_scale": replace(UniversalAblationConfig(), learned_scale=False),
    "dataset_id_conditioning": replace(UniversalAblationConfig(), dataset_id_conditioning=True),
    "channel_id_only": replace(
        UniversalAblationConfig(),
        physical_metadata=False,
        unit_metadata=False,
        sampling_rate_metadata=False,
        time_metadata=False,
        channel_id_embedding=True,
    ),
}


def get_ablation(name: str) -> UniversalAblationConfig:
    key = str(name).strip().lower()
    if key not in ABLATIONS:
        raise ValueError(f"unknown ablation: {name}. Choices: {', '.join(ABLATIONS)}")
    return ABLATIONS[key]
