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


def test_windows_never_bridge_discontinuous_segments_and_use_window_max_bound():
    rows=[]
    for seg,base in (("s1",1.0),("s2",100.0)):
        for i in range(6):
            rows.append({
                "trip_id":"t","segment_id":seg,"sample_uid":f"{seg}:{i}","split":"test","time":i,
                "x":base,"mu_ref":.6,"physics_lower_raw":[.1,.2,.3,.15,.25,.4][i],
            })
    df=pd.DataFrame(rows)
    X,y,lo,ids=windows_for_split(df,["x"],"test",L=3,stride=1,eval_start=2,physics_window=3)
    assert len(X)==8
    for w in X:
        assert np.all(w[:,0] < 10) or np.all(w[:,0] > 90)
    # First endpoint in each segment uses max(.1,.2,.3)=.3.
    assert np.isclose(lo[0],.3)
    assert np.isclose(lo[4],.3)


def test_segment_builder_breaks_reference_trace_and_large_time_gaps():
    from safegrip.data import add_lira_trajectory_segments
    df=pd.DataFrame({
        "trip_id":["t"]*6,
        "ref_source_file":["a.csv"]*4+["b.csv"]*2,
        "time":[0.0,.1,5.0,5.1,5.2,5.3],
    })
    z=add_lira_trajectory_segments(df,{"lira":{"segment_gap_s":1.0}})
    assert z.trajectory_id.nunique()==2
    assert z.segment_id.nunique()==3


def test_relative_pair_indices_never_cross_segment():
    from safegrip.benchmark import _previous_pair_indices
    ids=np.asarray(["segA:train:0","segA:train:1","segB:train:2","segB:train:3"],dtype=str)
    prev,valid=_previous_pair_indices(ids,1)
    assert valid.tolist()==[False,True,False,True]
    assert prev.tolist()==[0,0,2,2]
