from __future__ import annotations

from dataclasses import dataclass

from .ontology import QUANTITY_TO_ID, AXIS_TO_ID, LOCATION_TO_ID, TARGET_TYPE_TO_ID, ontology_id


@dataclass(frozen=True)
class PhysicalQuery:
    quantity: str
    axis: str | None = "scalar"
    location: str | None = "global"
    target_type: str | None = "instantaneous"
    name: str | None = None

    def ids(self) -> tuple[int, int, int, int]:
        return (
            ontology_id(self.quantity, QUANTITY_TO_ID),
            ontology_id(self.axis, AXIS_TO_ID),
            ontology_id(self.location, LOCATION_TO_ID),
            ontology_id(self.target_type, TARGET_TYPE_TO_ID),
        )


FRICTION_QUERY = PhysicalQuery("friction", "scalar", "road", "road_friction", "friction")
FORCE_X_QUERY = PhysicalQuery("force", "longitudinal", "tire", "force_component", "force_x")
FORCE_Y_QUERY = PhysicalQuery("force", "lateral", "tire", "force_component", "force_y")
FORCE_Z_QUERY = PhysicalQuery("force", "vertical", "tire", "force_component", "force_z")
SLIP_ANGLE_QUERY = PhysicalQuery("slip_angle", "scalar", "tire", "slip_angle", "slip_angle")
VELOCITY_X_QUERY = PhysicalQuery("velocity", "longitudinal", "vehicle_body", "state_component", "velocity_x")
VELOCITY_Y_QUERY = PhysicalQuery("velocity", "lateral", "vehicle_body", "state_component", "velocity_y")
YAW_RATE_QUERY = PhysicalQuery("angular_rate", "yaw", "vehicle_body", "state_component", "yaw_rate")
DISPLACEMENT_QUERY = PhysicalQuery("displacement", "scalar", "global", "localization", "displacement")
ORIENTATION_QUERY = PhysicalQuery("orientation", "yaw", "global", "localization", "orientation")
UTILIZATION_QUERY = PhysicalQuery("utilization", "scalar", "tire", "utilization", "utilization")
GRIP_MARGIN_QUERY = PhysicalQuery("grip_margin", "scalar", "tire", "grip_margin", "grip_margin")
SCALE_QUERY = PhysicalQuery("friction", "scalar", "road", "scale", "scale")

DATASET_QUERIES: dict[str, tuple[PhysicalQuery, ...]] = {
    "lira_cd": (FRICTION_QUERY,),
    "uc3m_tire": (FORCE_X_QUERY, FORCE_Y_QUERY, FORCE_Z_QUERY, SLIP_ANGLE_QUERY),
    "deep_dynamics_iac": (VELOCITY_X_QUERY, VELOCITY_Y_QUERY, YAW_RATE_QUERY),
    "io_vnbd": (DISPLACEMENT_QUERY, ORIENTATION_QUERY),
}


def queries_for_dataset(dataset: str) -> tuple[PhysicalQuery, ...]:
    key = str(dataset).strip().lower()
    if key not in DATASET_QUERIES:
        raise ValueError(f"unknown paper dataset: {dataset}. Choices: {', '.join(DATASET_QUERIES)}")
    return DATASET_QUERIES[key]
