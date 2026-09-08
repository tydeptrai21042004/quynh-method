import numpy as np
import pandas as pd

from safegrip.features import add_safegrip_features


def test_feature_engineering_is_label_free_and_bounded_excitation():
    n=12
    df=pd.DataFrame({
        "time":np.arange(n)/10,
        "split":["train"]*n,
        "segment_id":["a"]*n,
        "ax":np.linspace(0,1,n),"ay":np.linspace(0,-.5,n),
        "wheel_fl":500+np.arange(n),"wheel_fr":501+np.arange(n),
        "wheel_rl":499+np.arange(n),"wheel_rr":500+np.arange(n),
        "torque":np.linspace(0,200,n),"mu_ref":np.linspace(.3,.8,n),
    })
    a=add_safegrip_features(df,{"vehicle":{"gravity":9.81}})
    b=add_safegrip_features(df.assign(mu_ref=np.linspace(1.0,.1,n)),{"vehicle":{"gravity":9.81}})
    cols=[c for c in a if c.startswith("sg_")]
    assert cols
    assert np.allclose(a[cols],b[cols])
    assert ((a.sg_excitation_score>=0)&(a.sg_excitation_score<=1)).all()


def test_jerk_does_not_cross_split_boundary():
    df=pd.DataFrame({
        "time":[0,1,0,1],"split":["train","train","test","test"],"segment_id":["s","s","s","s"],
        "ax":[0,1,100,101],"ay":[0,0,0,0],"torque":[0,0,0,0],
    })
    z=add_safegrip_features(df,{"vehicle":{"gravity":9.81}})
    assert z.loc[2,"sg_jerk_x"]==0.0


def test_bundle_uses_disjoint_calibration_roles(tmp_path):
    import yaml
    from safegrip.data import make_synthetic
    from safegrip.benchmark import make_bundle
    from pathlib import Path
    cfg=yaml.safe_load((Path(__file__).resolve().parents[1]/"configs/kaggle_small.yaml").read_text())
    cfg["benchmark"]["common_warmup_samples"]=16
    cfg["sequence_length"]=8
    cfg["physics"]["window_samples"]=8
    cfg["stride"]=8
    csv=make_synthetic(tmp_path/"syn",n=1600,seed=3)
    b=make_bundle(csv,cfg,sequence_length=8,eval_start=15,proposal_features=True)
    assert b.proposal_features
    assert "sg_excitation_score" in b.features
    assert not np.any(b.lower_cal_mask & b.uq_cal_mask)
    assert b.lower_cal_mask.any() and b.uq_cal_mask.any()
    assert np.all((b.et>=0)&(b.et<=1))
