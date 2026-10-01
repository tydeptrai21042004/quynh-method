from __future__ import annotations

import numpy as np
import pandas as pd

from safegrip.universal.schema import SensorChannel, SensorMeta, SensorRecord, TargetValue
from ._paper_common import find_column

# Public U6ICRX sheets are documented as:
# A description, B time [s], C e1y, D e2y, E e3x.
# Aliases cover both descriptive Excel headers and short symbols.
STRAIN_ALIASES = (
    ("e1y", "measurement_channel_1_lateral_microstrains_e1y", "lateral_microstrains_e1y", "strain_1", "microstrain_1"),
    ("e2y", "measurement_channel_2_lateral_microstrains_e2y", "lateral_microstrains_e2y", "strain_2", "microstrain_2"),
    ("e3x", "measurement_channel_3_longitudinal_microstrains_e3x", "longitudinal_microstrains_e3x", "strain_3", "microstrain_3"),
)
TIME_ALIASES = ("time", "time_s", "time_in_seconds", "seconds")


def _numeric_fraction(series: pd.Series) -> float:
    if len(series) == 0:
        return 0.0
    return float(pd.to_numeric(series, errors="coerce").notna().mean())


def _resolve_uc3m_columns(frame: pd.DataFrame) -> tuple[str | None, tuple[str, ...]]:
    """Resolve the documented U6ICRX table schema without arbitrary numerics."""
    time_col = find_column(frame, TIME_ALIASES)
    strain_cols = tuple(c for aliases in STRAIN_ALIASES if (c := find_column(frame, aliases)) is not None)
    if len(strain_cols) == 3:
        return time_col, strain_cols

    # The official workbooks occasionally load with generic/merged headers.
    # The deposit documentation fixes the semantic column positions A--E, so a
    # positional fallback is valid only when B--E are predominantly numeric.
    cols = list(frame.columns)
    if len(cols) >= 5:
        positional_time = cols[1]
        positional_strain = tuple(cols[2:5])
        if _numeric_fraction(frame[positional_time]) >= 0.8 and all(
            _numeric_fraction(frame[c]) >= 0.8 for c in positional_strain
        ):
            return time_col or positional_time, positional_strain
    return time_col, strain_cols


def uc3m_tire_frame_to_record(
    frame: pd.DataFrame,
    *,
    sequence_id: str = "uc3m-tire",
    sample_rate_hz: float | None = None,
    strain_unit: str = "microstrain",
    slip_angle_deg: float | None = None,
) -> SensorRecord:
    """Convert one real U6ICRX test table into the universal sensor schema.

    The public deposit supplies the three strain waveforms and an experiment
    slip-angle condition. No force label is synthesized from the strain data.
    """
    if frame.empty:
        raise ValueError("frame cannot be empty")

    time_col, strain_cols = _resolve_uc3m_columns(frame)
    if time_col is not None:
        t = pd.to_numeric(frame[time_col], errors="coerce").to_numpy(float)
    elif sample_rate_hz and sample_rate_hz > 0:
        t = np.arange(len(frame), dtype=float) / float(sample_rate_hz)
    else:
        sample_rate_hz = 1.0
        t = np.arange(len(frame), dtype=float)

    if len(strain_cols) != 3:
        raise ValueError("could not resolve the documented e1y/e2y/e3x UC3M strain channels")

    axes = ("lateral", "lateral", "longitudinal")
    channels = []
    for col, axis in zip(strain_cols, axes):
        v = pd.to_numeric(frame[col], errors="coerce").to_numpy(float)
        mask = np.isfinite(t) & np.isfinite(v)
        if np.any(mask):
            channels.append(
                SensorChannel(v[mask], t[mask], SensorMeta("strain", strain_unit, axis, "tire", sample_rate_hz))
            )
    if len(channels) != 3:
        raise ValueError("UC3M table did not contain three finite strain channels")

    targets: dict[str, TargetValue] = {}
    slip_col = find_column(frame, ("slip_angle", "alpha", "slip_angle_deg"))
    if slip_col is not None:
        vals = pd.to_numeric(frame[slip_col], errors="coerce").to_numpy(float)
        vals = vals[np.isfinite(vals)]
        if len(vals):
            targets["slip_angle"] = TargetValue("slip_angle", float(np.deg2rad(np.mean(vals))))
    if "slip_angle" not in targets and slip_angle_deg is not None and np.isfinite(slip_angle_deg):
        targets["slip_angle"] = TargetValue("slip_angle", float(np.deg2rad(slip_angle_deg)))

    return SensorRecord(channels, targets, sequence_id=sequence_id, source_domain="uc3m_tire")
