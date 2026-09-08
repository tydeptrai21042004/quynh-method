from pathlib import Path
import pandas as pd
from safegrip.data import canonical_friction, canonical_vehicle

def test_aliases():
    f=pd.DataFrame({"µ_V [-]":[.5],"µ_H [-]":[.6],"F_vertikal_V [N]":[1000],"Lat":[1],"Lon":[2]})
    z=canonical_friction(f)
    assert abs(z.mu_ref.iloc[0]-.55)<1e-9
    v=pd.DataFrame({"Vehicle Speed":[10],"Longitudinal Acceleration":[1],"Lateral Acceleration":[2],"Yaw Rate":[.1]})
    c=canonical_vehicle(v)
    assert {"speed","ax","ay","yaw_rate"}.issubset(c.columns)


def test_lira_like_alignment_core():
    from safegrip.data import canonical_vehicle, canonical_friction, spatial_align
    car=pd.DataFrame({
        "Latitude":[55.0,55.00001],"Longitude":[12.0,12.00001],"Vehicle Speed":[10,11],
        "Longitudinal Acceleration":[.2,.3],"Lateral Acceleration":[.1,.2],"Yaw Rate":[.01,.02]
    })
    ref=pd.DataFrame({
        "Lat":[55.0,55.00001],"Lon":[12.0,12.00001],"µ_V [-]":[.5,.6],"µ_H [-]":[.52,.62]
    })
    z=spatial_align(canonical_vehicle(car),canonical_friction(ref),max_m=5)
    assert len(z)==2 and "mu_ref" in z and abs(z.mu_ref.iloc[0]-.51)<1e-9


def test_vehicle_units_from_column_names():
    v=pd.DataFrame({"Vehicle Speed [km/h]":[36.0],"Longitudinal Acceleration [g]":[1.0],"Lateral Acceleration [m/s2]":[2.0]})
    c=canonical_vehicle(v)
    assert abs(c.speed.iloc[0]-10.0)<1e-9
    assert abs(c.ax.iloc[0]-9.80665)<1e-6
    assert abs(c.ay.iloc[0]-2.0)<1e-9


def test_official_lira_task_streams_are_synchronised_before_alignment(tmp_path):
    """Regression for the Kaggle failure on separate task_7505 sensor files."""
    import numpy as np
    import yaml
    from safegrip.data import assemble_lira_task_streams, prepare_lira

    raw = tmp_path / "raw"
    out = tmp_path / "processed"
    raw.mkdir()

    # GPS is low-rate while CAN signals are higher-rate and live in separate
    # files, matching the structure of the public platoon-test download.
    tgps = np.arange(0.0, 20.1, 1.0)
    lat_gps = 55.0 + tgps * 1e-5
    lon_gps = np.full_like(tgps, 12.0)
    pd.DataFrame({"timestamp": 1_000_000.0 + tgps, "lat": lat_gps, "lon": lon_gps}).to_csv(
        raw / "task_7505_gps_raw.txt", index=False
    )

    ts = np.arange(0.0, 20.0, 0.1)
    for name, value in {
        "speed": np.full_like(ts, 36.0),       # official unit: km/h
        "acc_lon": np.full_like(ts, 0.20),
        "acc_trans": np.full_like(ts, 0.10),
        "acc_yaw": np.full_like(ts, 0.01),
    }.items():
        pd.DataFrame({"timestamp": 1_000_000.0 + ts, "value": value}).to_csv(
            raw / f"task_7505_{name}.txt", index=False
        )

    # High-resolution VIAFRIK trace along the same route.
    tr = np.arange(0.0, 20.0, 0.1)
    pd.DataFrame({
        "Lat": 55.0 + tr * 1e-5,
        "Lon": np.full_like(tr, 12.0),
        "µ_V [-]": np.full_like(tr, 0.55),
        "µ_H [-]": np.full_like(tr, 0.57),
    }).to_csv(raw / "m3_custom_fric_hh.csv", index=False)

    cfg = yaml.safe_load((Path(__file__).parents[1] / "configs" / "default.yaml").read_text())
    cfg["lira"]["resample_hz"] = 10.0
    cfg["lira"]["heading_tolerance_deg"] = 60.0

    task_files = sorted(raw.glob("task_7505*.txt"))
    assembled, report = assemble_lira_task_streams(task_files, cfg)
    assert {"time", "lat", "lon", "speed", "ax", "ay"}.issubset(assembled.columns)
    assert len(assembled) > 100
    assert abs(float(assembled.speed.median()) - 10.0) < 1e-6
    assert report["rows_after_sync"] == len(assembled)

    result = prepare_lira(raw, out, cfg)
    z = pd.read_csv(result)
    assert len(z) > 50
    assert {"mu_ref", "physics_lower_raw", "split", "sample_uid"}.issubset(z.columns)
    assert {"train", "calibration", "validation", "test"}.issubset(set(z.split))
    assert (out / "lira_stream_assembly_report.json").exists()
