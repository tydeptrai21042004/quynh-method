from __future__ import annotations

import numpy as np
import pandas as pd

from safegrip.universal.schema import SensorChannel, SensorMeta, SensorRecord, TargetValue
from ._paper_common import find_column

STRAIN_ALIASES = (
    ("strain_1", "microstrain_1", "strain1", "gauge_1", "sensor_1"),
    ("strain_2", "microstrain_2", "strain2", "gauge_2", "sensor_2"),
    ("strain_3", "microstrain_3", "strain3", "gauge_3", "sensor_3"),
)


def uc3m_tire_frame_to_record(
    frame: pd.DataFrame,
    *,
    sequence_id: str = "uc3m-tire",
    sample_rate_hz: float | None = None,
    strain_unit: str = "microstrain",
    slip_angle_deg: float | None = None,
) -> SensorRecord:
    """Create a D2 record without synthesizing unavailable force targets."""
    if frame.empty:
        raise ValueError("frame cannot be empty")
    time_col = find_column(frame, ("time", "timestamp", "time_s", "seconds"))
    if time_col is not None:
        t = pd.to_numeric(frame[time_col], errors="coerce").to_numpy(float)
    elif sample_rate_hz and sample_rate_hz > 0:
        t = np.arange(len(frame), dtype=float) / float(sample_rate_hz)
    else:
        # U6ICRX workbooks can be steady-state sequences without an explicit
        # clock.  Index time is permitted only as ordering, with 1 Hz metadata.
        sample_rate_hz = 1.0
        t = np.arange(len(frame), dtype=float)

    channels = []
    used = set()
    for i, aliases in enumerate(STRAIN_ALIASES):
        col = find_column(frame, aliases)
        if col is None or col in used:
            continue
        used.add(col)
        v = pd.to_numeric(frame[col], errors="coerce").to_numpy(float)
        mask = np.isfinite(t) & np.isfinite(v)
        if np.any(mask):
            channels.append(SensorChannel(v[mask], t[mask], SensorMeta("strain", strain_unit, f"scalar", "tire", sample_rate_hz)))
    # Fallback: accept explicitly named strain columns only; never arbitrary numerics.
    if not channels:
        for col in frame.columns:
            if "strain" not in str(col).lower():
                continue
            v = pd.to_numeric(frame[col], errors="coerce").to_numpy(float)
            mask = np.isfinite(t) & np.isfinite(v)
            if np.any(mask):
                channels.append(SensorChannel(v[mask], t[mask], SensorMeta("strain", strain_unit, "scalar", "tire", sample_rate_hz)))
    if not channels:
        raise ValueError("no strain channels were resolved from UC3M frame")

    targets: dict[str, TargetValue] = {}
    target_aliases = {
        "force_x": ("fx", "force_x", "longitudinal_force"),
        "force_y": ("fy", "force_y", "lateral_force"),
        "force_z": ("fz", "force_z", "vertical_force", "normal_force"),
        "slip_angle": ("slip_angle", "alpha"),
    }
    for name, aliases in target_aliases.items():
        col = find_column(frame, aliases)
        if col is None:
            continue
        vals = pd.to_numeric(frame[col], errors="coerce").to_numpy(float)
        vals = vals[np.isfinite(vals)]
        if len(vals):
            value = float(np.mean(vals))
            if name == "slip_angle":
                value = float(np.deg2rad(value))
            targets[name] = TargetValue(name, value)
    if "slip_angle" not in targets and slip_angle_deg is not None:
        targets["slip_angle"] = TargetValue("slip_angle", float(np.deg2rad(slip_angle_deg)))
    return SensorRecord(channels, targets, sequence_id=sequence_id, source_domain="uc3m_tire")
