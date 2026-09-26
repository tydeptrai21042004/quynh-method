from __future__ import annotations

import pandas as pd

from .base import ChannelSpec, TargetSpec
from .frame import dataframe_to_sensor_record


def prepared_friction_frame_to_record(frame: pd.DataFrame, *, sequence_id: str = "prepared-friction", source_domain: str = "friction"):
    """Bridge a legacy prepared friction frame into the universal schema.

    This adapter deliberately consumes canonical prepared columns rather than
    duplicating raw-dataset parsing logic from safegrip.data.
    """
    mapping = []
    candidates = {
        "speed": ChannelSpec("speed", "velocity", "m/s", "scalar", "vehicle_body"),
        "ax": ChannelSpec("ax", "acceleration", "m/s^2", "longitudinal", "vehicle_body"),
        "ay": ChannelSpec("ay", "acceleration", "m/s^2", "lateral", "vehicle_body"),
        "yaw_rate": ChannelSpec("yaw_rate", "angular_rate", "rad/s", "yaw", "vehicle_body"),
        "steering": ChannelSpec("steering", "steering_angle", "rad", "scalar", "vehicle_body"),
    }
    for name, spec in candidates.items():
        if name in frame.columns:
            mapping.append(spec)
    if not mapping:
        raise ValueError("prepared frame has no recognized universal sensor columns")
    # Legacy prepared friction data are normally resampled before this bridge.
    for i, spec in enumerate(mapping):
        mapping[i] = ChannelSpec(spec.column, spec.quantity, spec.unit, spec.axis, spec.location, 20.0)
    targets = [TargetSpec("mu_ref", "friction")] if "mu_ref" in frame.columns else []
    return dataframe_to_sensor_record(
        frame,
        mapping,
        targets=targets,
        sequence_id=sequence_id,
        source_domain=source_domain,
    )
