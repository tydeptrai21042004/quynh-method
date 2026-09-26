from __future__ import annotations

"""Canonical physical ontology for Universal SafeGrip.

The ontology is intentionally dataset-agnostic.  Dataset loaders translate raw
column names into these canonical semantic labels; predictive models never see
raw dataset identifiers.
"""

from dataclasses import dataclass

UNKNOWN = "unknown"
SCALAR = "scalar"

QUANTITIES = (
    UNKNOWN,
    "acceleration",
    "angular_rate",
    "velocity",
    "wheel_speed",
    "steering_angle",
    "torque",
    "force",
    "strain",
    "pressure",
    "slip_ratio",
    "slip_angle",
    "temperature",
    "friction",
    "grip_margin",
    "utilization",
)

AXES = (
    UNKNOWN,
    SCALAR,
    "longitudinal",
    "lateral",
    "vertical",
    "yaw",
    "roll",
    "pitch",
)

LOCATIONS = (
    UNKNOWN,
    "vehicle_body",
    "front_left",
    "front_right",
    "rear_left",
    "rear_right",
    "tire",
    "road",
    "global",
)

UNIT_CLASSES = (
    UNKNOWN,
    "dimensionless",
    "acceleration",
    "angular_rate",
    "velocity",
    "force",
    "torque",
    "pressure",
    "strain",
    "temperature",
    "angle",
)

TARGET_TYPES = (
    UNKNOWN,
    "instantaneous",
    "road_friction",
    "peak_friction",
    "force_component",
    "grip_margin",
    "utilization",
    "scale",
)


def _index(values: tuple[str, ...]) -> dict[str, int]:
    return {name: i for i, name in enumerate(values)}


QUANTITY_TO_ID = _index(QUANTITIES)
AXIS_TO_ID = _index(AXES)
LOCATION_TO_ID = _index(LOCATIONS)
UNIT_CLASS_TO_ID = _index(UNIT_CLASSES)
TARGET_TYPE_TO_ID = _index(TARGET_TYPES)


def canonical_name(value: str | None) -> str:
    if value is None:
        return UNKNOWN
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


def ontology_id(value: str | None, mapping: dict[str, int]) -> int:
    return mapping.get(canonical_name(value), mapping[UNKNOWN])


@dataclass(frozen=True)
class OntologySizes:
    quantities: int = len(QUANTITIES)
    axes: int = len(AXES)
    locations: int = len(LOCATIONS)
    unit_classes: int = len(UNIT_CLASSES)
    target_types: int = len(TARGET_TYPES)
