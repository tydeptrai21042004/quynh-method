import numpy as np
import pytest

from safegrip.universal.schema import SensorMeta, SensorChannel, SensorRecord


def test_unit_canonicalization_force_and_speed():
    force = SensorChannel(np.array([1.0, 2.0]), np.array([0.0, 1.0]), SensorMeta("force", "kN", "longitudinal", "front_left"))
    assert np.allclose(force.values, [1000.0, 2000.0])
    assert force.meta.canonical_unit == "N"
    speed = SensorChannel(np.array([36.0]), np.array([0.0]), SensorMeta("velocity", "km/h"))
    assert np.allclose(speed.values, [10.0])


def test_sensor_record_requires_channel_and_sorted_time():
    with pytest.raises(ValueError):
        SensorRecord([])
    with pytest.raises(ValueError):
        SensorChannel(np.ones(2), np.array([1.0, 0.0]), SensorMeta("acceleration", "m/s^2"))


def test_numeric_context_only_record_is_allowed():
    from safegrip.universal.schema import ContextValue
    r = SensorRecord([], context=[ContextValue(36.0, SensorMeta("velocity", "km/h", "scalar", "vehicle_body"))])
    assert np.isclose(r.context[0].value, 10.0)
