import numpy as np
import pandas as pd

from safegrip.benchmark import windows_for_split
from safegrip.data import impute_features_by_partition, parse_lira_context, spatial_align


def test_windows_never_bridge_trip_boundaries():
    rows=[]
    for trip,value in (("a",1.0),("b",100.0)):
        for i in range(5):
            rows.append({"trip_id":trip,"sample_uid":f"{trip}:{i}","split":"train","time":i,
                         "x":value+i*0.0,"mu_ref":.5,"physics_lower_raw":.1})
    df=pd.DataFrame(rows)
    X,y,lo,ids=windows_for_split(df,["x"],"train",L=4,stride=1,eval_start=3)
    assert len(X)==4
    for w in X:
        assert np.all(w[:,0] < 10) or np.all(w[:,0] > 90)
    assert set(x.split(":")[0] for x in ids)=={"a","b"}


def test_imputation_cannot_cross_split_boundary():
    df=pd.DataFrame({
        "trip_id":["t"]*4,
        "split":["train","train","validation","validation"],
        "x":[1.0,np.nan,100.0,np.nan],
    })
    z=impute_features_by_partition(df,["x"],limit=None)
    assert z.loc[1,"x"]==1.0
    assert z.loc[3,"x"]==100.0


def test_route_context_only_uses_explicit_tokens(tmp_path):
    p=tmp_path/"CPH1"/"HH"/"task_7505_01.txt"
    c=parse_lira_context(p,tmp_path)
    assert c["route_id"]=="CPH1" and c["direction"]=="HH"
    q=parse_lira_context(tmp_path/"misc"/"trip.txt",tmp_path)
    assert q["route_id"]=="unknown" and q["direction"]=="unknown"


def test_spatial_alignment_reference_indices_are_monotone():
    car=pd.DataFrame({"lat":[55,55.00001,55.00002],"lon":[12,12,12]})
    ref=pd.DataFrame({"lat":[55,55.00001,55.00002],"lon":[12,12,12],"mu_ref":[.4,.5,.6]})
    z=spatial_align(car,ref,max_m=5,heading_tolerance_deg=60,enforce_monotonic=True)
    assert len(z)==3
    assert np.all(np.diff(z.match_ref_index.to_numpy())>=0)
