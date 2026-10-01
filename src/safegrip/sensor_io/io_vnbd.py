from __future__ import annotations

import numpy as np
import pandas as pd

from safegrip.universal.schema import SensorChannel, SensorMeta, SensorRecord, TargetValue
from ._paper_common import find_column


# Column order documented by the IO-VNBD data paper for the synchronized
# vehicle stream. Keeping this schema here makes raw-file parsing declarative
# rather than scattering numeric indices through experiment scripts.
IO_VNBD_VEHICLE_COLUMNS = (
    "satellites", "time_s", "latitude_deg", "longitude_deg",
    "gps_velocity_kmh", "heading_deg", "height_km", "vertical_velocity_kmh",
    "sample_period_s", "steering_deg", "wheel_speed_fl_rad_s",
    "wheel_speed_fr_rad_s", "wheel_speed_rl_rad_s", "wheel_speed_rr_rad_s",
    "yaw_rate_deg_s", "vehicle_speed_kmh", "accel_long_g", "accel_lat_g",
    "engine_speed_rpm", "engine_torque_nm", "brake_pressure_bar",
    "gear", "fuel_rate", "engine_temp_c", "outside_temp_c",
    "brake_pedal", "clutch_pedal", "accelerator_pedal_aux", "accelerator_pedal",
)


def normalize_io_vnbd_vehicle_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a numeric frame with the documented 29-column vehicle schema.

    Official synchronized files may be headerless.  If semantic column names
    already exist they are retained; otherwise the first 29 columns are mapped
    positionally according to the public data-paper schema.  Non-numeric rows
    (for example an embedded header) are removed after coercion.
    """
    if frame.empty:
        raise ValueError("frame cannot be empty")

    known = {
        "time_s", "time", "timestamp", "latitude_deg", "longitude_deg",
        "vehicle_speed_kmh", "vehicle_speed", "speed",
        "wheel_speed_fl_rad_s", "wheel_speed_fr_rad_s", "wheel_speed_rl_rad_s",
        "wheel_speed_rr_rad_s", "wheel_speed_fl", "wheel_speed_fr",
        "wheel_speed_rl", "wheel_speed_rr", "yaw_rate_deg_s", "yaw_rate",
        "accel_long_g", "accel_lat_g", "ax", "ay", "steering_deg", "steering",
    }
    lowered = {str(c).strip().lower() for c in frame.columns}
    if known.intersection(lowered):
        renamed = frame.copy()
        renamed.columns = [str(c).strip().lower() for c in renamed.columns]
    else:
        if frame.shape[1] < len(IO_VNBD_VEHICLE_COLUMNS):
            raise ValueError(
                f"IO-VNBD vehicle frame needs at least {len(IO_VNBD_VEHICLE_COLUMNS)} columns; "
                f"got {frame.shape[1]}"
            )
        renamed = frame.iloc[:, : len(IO_VNBD_VEHICLE_COLUMNS)].copy()
        renamed.columns = IO_VNBD_VEHICLE_COLUMNS

    numeric = renamed.apply(pd.to_numeric, errors="coerce")
    # Remove rows that contain no numeric measurements at all (for example an
    # embedded textual header) without requiring GPS columns for generic adapter
    # use.  The real-data runner separately requires finite GPS for targets.
    numeric = numeric.loc[numeric.notna().any(axis=1)].reset_index(drop=True)
    if numeric.empty:
        raise ValueError("IO-VNBD frame contains no numeric vehicle rows")
    return numeric


def io_vnbd_frame_to_record(
    frame: pd.DataFrame,
    *,
    sequence_id: str = "io-vnbd",
    sample_rate_hz: float = 10.0,
    displacement_m: float | None = None,
) -> SensorRecord:
    """Convert one real synchronized IO-VNBD vehicle window to the shared schema.

    Inputs remain measured vehicle channels.  A displacement target is accepted
    only when supplied from the real GPS trajectory by the caller; it is never
    synthesized from wheel speeds or a simulator.
    """
    frame = normalize_io_vnbd_vehicle_frame(frame)
    time_col = find_column(frame, ("time_s", "time", "timestamp", "seconds"))
    t = (
        pd.to_numeric(frame[time_col], errors="coerce").to_numpy(float)
        if time_col is not None
        else np.arange(len(frame), dtype=float) / float(sample_rate_hz)
    )
    # Make each window local in time, which avoids large absolute timestamps in
    # token time embeddings while preserving the measured sampling intervals.
    finite_t = t[np.isfinite(t)]
    if len(finite_t):
        t = t - float(finite_t[0])

    specs = (
        (("wheel_speed_fl_rad_s", "wheel_speed_fl", "front_left_wheel_speed"), "wheel_speed", "rad/s", "scalar", "front_left"),
        (("wheel_speed_fr_rad_s", "wheel_speed_fr", "front_right_wheel_speed"), "wheel_speed", "rad/s", "scalar", "front_right"),
        (("wheel_speed_rl_rad_s", "wheel_speed_rl", "rear_left_wheel_speed"), "wheel_speed", "rad/s", "scalar", "rear_left"),
        (("wheel_speed_rr_rad_s", "wheel_speed_rr", "rear_right_wheel_speed"), "wheel_speed", "rad/s", "scalar", "rear_right"),
        (("vehicle_speed_kmh", "vehicle_speed", "speed"), "velocity", "km/h", "scalar", "vehicle_body"),
        (("yaw_rate_deg_s", "yaw_rate", "gyro_z"), "angular_rate", "deg/s", "yaw", "vehicle_body"),
        (("accel_long_g", "ax", "acceleration_x"), "acceleration", "g", "longitudinal", "vehicle_body"),
        (("accel_lat_g", "ay", "acceleration_y"), "acceleration", "g", "lateral", "vehicle_body"),
        (("steering_deg", "steering", "steering_angle"), "steering_angle", "deg", "scalar", "vehicle_body"),
    )

    channels = []
    for aliases, quantity, unit, axis, location in specs:
        col = find_column(frame, aliases)
        if col is None:
            continue
        v = pd.to_numeric(frame[col], errors="coerce").to_numpy(float)
        mask = np.isfinite(t) & np.isfinite(v)
        if np.any(mask):
            channels.append(
                SensorChannel(v[mask], t[mask], SensorMeta(quantity, unit, axis, location, sample_rate_hz))
            )
    if not channels:
        raise ValueError("no supported IO-VNBD ego-motion channels resolved")

    targets = {}
    if displacement_m is not None and np.isfinite(displacement_m):
        targets["displacement"] = TargetValue("displacement", float(displacement_m))

    return SensorRecord(channels, targets, sequence_id=sequence_id, source_domain="io_vnbd")
