from __future__ import annotations

import numpy as np
import pandas as pd

from safegrip.universal.schema import SensorChannel, SensorMeta, SensorRecord, TargetValue
from ._paper_common import find_column


def io_vnbd_frame_to_record(
    frame: pd.DataFrame,
    *,
    sequence_id: str = "io-vnbd",
    sample_rate_hz: float = 10.0,
) -> SensorRecord:
    """D4 adapter preserving wheel channels as angular quantities when angular."""
    if frame.empty:
        raise ValueError("frame cannot be empty")
    time_col = find_column(frame, ("time", "timestamp", "time_s", "seconds"))
    t = (pd.to_numeric(frame[time_col], errors="coerce").to_numpy(float) if time_col is not None
         else np.arange(len(frame), dtype=float) / float(sample_rate_hz))

    specs = [
        (("ax", "acc_x", "acceleration_x", "longitudinal_acceleration"), "acceleration", "m/s^2", "longitudinal"),
        (("ay", "acc_y", "acceleration_y", "lateral_acceleration"), "acceleration", "m/s^2", "lateral"),
        (("yaw_rate", "gyro_z", "angular_velocity_z"), "angular_rate", "rad/s", "yaw"),
        (("speed", "vehicle_speed", "velocity"), "velocity", "m/s", "scalar"),
        (("wheel_speed_fl", "front_left_wheel_speed", "fl_wheel_speed"), "wheel_speed", "rad/s", "scalar"),
        (("wheel_speed_fr", "front_right_wheel_speed", "fr_wheel_speed"), "wheel_speed", "rad/s", "scalar"),
        (("wheel_speed_rl", "rear_left_wheel_speed", "rl_wheel_speed"), "wheel_speed", "rad/s", "scalar"),
        (("wheel_speed_rr", "rear_right_wheel_speed", "rr_wheel_speed"), "wheel_speed", "rad/s", "scalar"),
    ]
    locations = ["vehicle_body", "vehicle_body", "vehicle_body", "vehicle_body", "front_left", "front_right", "rear_left", "rear_right"]
    channels = []
    for (aliases, quantity, unit, axis), location in zip(specs, locations):
        col = find_column(frame, aliases)
        if col is None:
            continue
        v = pd.to_numeric(frame[col], errors="coerce").to_numpy(float)
        mask = np.isfinite(t) & np.isfinite(v)
        if np.any(mask):
            channels.append(SensorChannel(v[mask], t[mask], SensorMeta(quantity, unit, axis, location, sample_rate_hz)))
    if not channels:
        raise ValueError("no supported IO-VNBD ego-motion channels resolved")

    targets = {}
    # Only consume explicit paper-task/error targets.  Do not derive displacement
    # from GPS inside the adapter because that would silently define a new task.
    for name, aliases in {
        "displacement": ("displacement_error", "position_error", "displacement", "crse_displacement"),
        "orientation": ("orientation_error", "yaw_error", "orientation", "crse_orientation"),
    }.items():
        col = find_column(frame, aliases)
        if col is None:
            continue
        vals = pd.to_numeric(frame[col], errors="coerce").to_numpy(float)
        vals = vals[np.isfinite(vals)]
        if len(vals):
            targets[name] = TargetValue(name, float(np.mean(vals)))
    return SensorRecord(channels, targets, sequence_id=sequence_id, source_domain="io_vnbd")
