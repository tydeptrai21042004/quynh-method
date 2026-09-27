from __future__ import annotations

import numpy as np
import pandas as pd

from safegrip.universal.schema import SensorChannel, SensorMeta, SensorRecord, TargetValue
from ._paper_common import find_column


def deep_dynamics_iac_frame_to_record(
    frame: pd.DataFrame,
    *,
    sequence_id: str = "deep-dynamics-iac",
    sample_rate_hz: float = 25.0,
) -> SensorRecord:
    """One-step D3 record: history is input, final state is the prediction target."""
    if len(frame) < 2:
        raise ValueError("D3 record requires at least two rows for history -> next-state prediction")
    time_col = find_column(frame, ("time", "timestamp", "t", "time_s"))
    if time_col is not None:
        t_all = pd.to_numeric(frame[time_col], errors="coerce").to_numpy(float)
    else:
        t_all = np.arange(len(frame), dtype=float) / float(sample_rate_hz)
    hist = frame.iloc[:-1]
    t = t_all[:-1]

    specs = [
        (("vx", "v_x", "velocity_x", "longitudinal_velocity"), "velocity", "m/s", "longitudinal"),
        (("vy", "v_y", "velocity_y", "lateral_velocity"), "velocity", "m/s", "lateral"),
        (("omega", "yaw_rate", "yawrate", "r"), "angular_rate", "rad/s", "yaw"),
        (("delta", "steering", "steering_angle"), "steering_angle", "rad", "scalar"),
        (("torque", "drive_torque", "t"), "torque", "N*m", "scalar"),
    ]
    channels = []
    for aliases, quantity, unit, axis in specs:
        col = find_column(frame, aliases)
        if col is None or col == time_col:
            continue
        v = pd.to_numeric(hist[col], errors="coerce").to_numpy(float)
        mask = np.isfinite(t) & np.isfinite(v)
        if np.any(mask):
            channels.append(SensorChannel(v[mask], t[mask], SensorMeta(quantity, unit, axis, "vehicle_body", sample_rate_hz)))
    if not channels:
        raise ValueError("no supported IAC state/control channels resolved")

    targets = {}
    for name, aliases in {
        "velocity_x": ("vx", "v_x", "velocity_x", "longitudinal_velocity"),
        "velocity_y": ("vy", "v_y", "velocity_y", "lateral_velocity"),
        "yaw_rate": ("omega", "yaw_rate", "yawrate", "r"),
    }.items():
        col = find_column(frame, aliases)
        if col is not None:
            value = pd.to_numeric(frame[col], errors="coerce").to_numpy(float)[-1]
            if np.isfinite(value):
                targets[name] = TargetValue(name, float(value))
    return SensorRecord(channels, targets, sequence_id=sequence_id, source_domain="deep_dynamics_iac")
