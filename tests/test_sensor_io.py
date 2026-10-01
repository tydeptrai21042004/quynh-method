import numpy as np
import pandas as pd

from safegrip.sensor_io import ChannelSpec, TargetSpec, dataframe_to_sensor_record, prepared_friction_frame_to_record


def test_generic_dataframe_mapping():
    df = pd.DataFrame({"t": [0.0, 0.1, 0.2], "Ay": [1.0, 2.0, 3.0], "mu": [0.8, 0.8, 0.8]})
    r = dataframe_to_sensor_record(
        df,
        [ChannelSpec("Ay", "acceleration", "m/s^2", "lateral", "vehicle_body", 10.0)],
        targets=[TargetSpec("mu", "friction")],
        time_column="t",
    )
    assert len(r.channels) == 1
    assert np.isclose(r.targets["friction"].value, 0.8)


def test_legacy_prepared_bridge():
    df = pd.DataFrame({"speed": [10.0, 11.0], "ax": [0.1, 0.2], "ay": [0.3, 0.2], "mu_ref": [0.7, 0.8]})
    r = prepared_friction_frame_to_record(df)
    assert len(r.channels) == 3
    assert "friction" in r.targets


def test_uc3m_public_u6icrx_positional_schema_and_real_slip_condition():
    from safegrip.sensor_io.uc3m_tire import uc3m_tire_frame_to_record

    n = 12
    frame = pd.DataFrame({
        "description": ["sample"] * n,
        "generic_time": np.linspace(0.0, 0.11, n),
        "generic_c": np.linspace(100.0, 120.0, n),
        "generic_d": np.linspace(200.0, 220.0, n),
        "generic_e": np.linspace(300.0, 320.0, n),
    })
    record = uc3m_tire_frame_to_record(frame, slip_angle_deg=6.0)
    assert len(record.channels) == 3
    assert {c.meta.quantity for c in record.channels} == {"strain"}
    assert tuple(c.meta.axis for c in record.channels) == ("lateral", "lateral", "longitudinal")
    assert set(record.targets) == {"slip_angle"}
    assert np.isclose(record.targets["slip_angle"].value, np.deg2rad(6.0))


def test_io_vnbd_documented_29_column_schema_and_real_displacement_target():
    from safegrip.sensor_io.io_vnbd import (
        IO_VNBD_VEHICLE_COLUMNS,
        io_vnbd_frame_to_record,
        normalize_io_vnbd_vehicle_frame,
    )

    n = 11
    raw = np.zeros((n, len(IO_VNBD_VEHICLE_COLUMNS)), dtype=float)
    raw[:, 1] = np.arange(n) / 10.0
    raw[:, 2] = 52.0
    raw[:, 3] = 4.0 + np.arange(n) * 1e-5
    raw[:, 10:14] = 20.0
    raw[:, 14] = 1.5
    raw[:, 15] = 36.0
    raw[:, 16] = 0.1
    raw[:, 17] = 0.05
    frame = normalize_io_vnbd_vehicle_frame(pd.DataFrame(raw))
    record = io_vnbd_frame_to_record(frame, displacement_m=9.5)
    assert "displacement" in record.targets
    assert np.isclose(record.targets["displacement"].value, 9.5)
    quantities = [c.meta.quantity for c in record.channels]
    assert quantities.count("wheel_speed") == 4
    assert "velocity" in quantities
    assert "angular_rate" in quantities
    assert quantities.count("acceleration") == 2
    speed = next(c for c in record.channels if c.meta.quantity == "velocity")
    assert np.allclose(speed.values, 10.0)  # 36 km/h -> canonical 10 m/s
