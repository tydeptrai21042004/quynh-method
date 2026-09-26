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
