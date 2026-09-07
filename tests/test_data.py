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
