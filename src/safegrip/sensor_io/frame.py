from __future__ import annotations

import numpy as np
import pandas as pd

from safegrip.universal.schema import SensorMeta, SensorChannel, SensorRecord, TargetValue
from .base import ChannelSpec, TargetSpec


def dataframe_to_sensor_record(
    frame: pd.DataFrame,
    channels: list[ChannelSpec] | tuple[ChannelSpec, ...],
    *,
    targets: list[TargetSpec] | tuple[TargetSpec, ...] = (),
    time_column: str | None = None,
    sequence_id: str = "",
    source_domain: str | None = None,
) -> SensorRecord:
    if frame.empty:
        raise ValueError("frame cannot be empty")
    if time_column is not None:
        if time_column not in frame:
            raise KeyError(time_column)
        t = pd.to_numeric(frame[time_column], errors="coerce").to_numpy(float)
    else:
        # When no timestamps exist, each channel must provide a sample rate.
        rates = [c.sample_rate_hz for c in channels if c.sample_rate_hz]
        if not rates:
            raise ValueError("time_column or channel sample_rate_hz is required")
        base_rate = float(rates[0])
        t = np.arange(len(frame), dtype=float) / base_rate

    out_channels = []
    for spec in channels:
        if spec.column not in frame:
            raise KeyError(spec.column)
        v = pd.to_numeric(frame[spec.column], errors="coerce").to_numpy(float)
        mask = np.isfinite(t) & np.isfinite(v)
        if not np.any(mask):
            continue
        out_channels.append(SensorChannel(
            v[mask], t[mask],
            SensorMeta(
                quantity=spec.quantity,
                unit=spec.unit,
                axis=spec.axis,
                location=spec.location,
                sample_rate_hz=spec.sample_rate_hz,
            ),
        ))
    target_values: dict[str, TargetValue] = {}
    for spec in targets:
        if spec.column not in frame:
            continue
        vals = pd.to_numeric(frame[spec.column], errors="coerce").to_numpy(float)
        vals = vals[np.isfinite(vals)]
        if len(vals):
            # Record-level default: mean target. Window-level datasets should
            # create one SensorRecord per target-aligned window instead.
            target_values[spec.name] = TargetValue(spec.name, float(np.mean(vals)), True)
    return SensorRecord(out_channels, target_values, sequence_id, source_domain=source_domain)
