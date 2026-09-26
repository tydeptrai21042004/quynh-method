from __future__ import annotations

from dataclasses import dataclass

from .ontology import (
    QUANTITY_TO_ID,
    AXIS_TO_ID,
    LOCATION_TO_ID,
    TARGET_TYPE_TO_ID,
    ontology_id,
)


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
FORCE_X_QUERY = PhysicalQuery("force", "longitudinal", "vehicle_body", "force_component", "force_x")
FORCE_Y_QUERY = PhysicalQuery("force", "lateral", "vehicle_body", "force_component", "force_y")
FORCE_Z_QUERY = PhysicalQuery("force", "vertical", "vehicle_body", "force_component", "force_z")
UTILIZATION_QUERY = PhysicalQuery("utilization", "scalar", "vehicle_body", "utilization", "utilization")
GRIP_MARGIN_QUERY = PhysicalQuery("grip_margin", "scalar", "tire", "grip_margin", "grip_margin")
SCALE_QUERY = PhysicalQuery("friction", "scalar", "road", "scale", "scale")
