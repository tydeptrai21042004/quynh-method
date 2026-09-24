import numpy as np
import pandas as pd

from safegrip.features import add_safegrip_features


def _write_sequence_fixture(out_dir, n=1600):
    """Deterministic unit-test fixture with the canonical real-data schema.

    This is not exposed as a dataset or benchmark source; it only exercises
    windowing/calibration invariants without downloading external data in tests.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    i=np.arange(n,dtype=float)
    speed=15.0+2.0*np.sin(i/50.0)
    ax=0.25*np.sin(i/17.0)
    ay=0.18*np.cos(i/23.0)
    mu=0.65+0.05*np.sin(i/200.0)
    lower=np.minimum(mu, np.hypot(ax,ay)/9.81)
    q=i/n
    split=np.where(q<.60,"train",np.where(q<.72,"calibration",np.where(q<.84,"validation","test")))
    df=pd.DataFrame({
        "time":i/20.0,"speed":speed,"ax":ax,"ay":ay,
        "yaw_rate":ay/np.maximum(speed,1.0),"steer":0.01*ay,
        "wheel_fl":500+i*0.01,"wheel_fr":501+i*0.01,
        "wheel_rl":499+i*0.01,"wheel_rr":500+i*0.01,
        "torque":50+5*np.sin(i/31.0),"mu_ref":mu,
        "physics_lower_raw":lower,"split":split,
        "trip_id":"fixture_real_schema","segment_id":"fixture_real_schema",
    })
    df["sample_uid"]=[f"fixture_real_schema:{sp}:{j}" for j,sp in enumerate(split)]
    path=out_dir/"real_schema_fixture.csv"
    df.to_csv(path,index=False)
    return path


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
    from safegrip.benchmark import make_bundle
    from pathlib import Path
    cfg=yaml.safe_load((Path(__file__).resolve().parents[1]/"configs/kaggle_small.yaml").read_text())
    cfg["benchmark"]["common_warmup_samples"]=16
    cfg["sequence_length"]=8
    cfg["physics"]["window_samples"]=8
    cfg["stride"]=8
    csv=_write_sequence_fixture(tmp_path/"fixture1",n=1600)
    b=make_bundle(csv,cfg,sequence_length=8,eval_start=15,proposal_features=True)
    assert b.proposal_features
    assert "sg_excitation_score" in b.features
    assert not np.any(b.lower_cal_mask & b.uq_cal_mask)
    assert b.lower_cal_mask.any() and b.uq_cal_mask.any()
    assert np.all((b.et>=0)&(b.et<=1))


def test_excitation_peak_memory_is_causal_and_resets_at_boundary():
    df=pd.DataFrame({
        "time":[0,1,2,0,1],
        "split":["train","train","train","test","test"],
        "segment_id":["a","a","a","b","b"],
        "ax":[0,9.81,0,0,0],"ay":[0,0,0,0,0],"torque":[0,0,0,0,0],
    })
    z=add_safegrip_features(df,{"vehicle":{"gravity":9.81},"feature_engineering":{"excitation_half_life_samples":2}})
    assert z.loc[1,"sg_excitation_score"]>=z.loc[1,"sg_excitation_instant"]-1e-12
    assert z.loc[2,"sg_excitation_score"]>z.loc[2,"sg_excitation_instant"]
    assert z.loc[3,"sg_excitation_score"]==z.loc[3,"sg_excitation_instant"]


def test_bundle_preserves_excitation_score_in_physical_scale(tmp_path):
    import yaml
    from safegrip.benchmark import make_bundle
    from pathlib import Path
    cfg=yaml.safe_load((Path(__file__).resolve().parents[1]/"configs/kaggle_small.yaml").read_text())
    cfg["benchmark"]["common_warmup_samples"]=16
    cfg["sequence_length"]=8; cfg["physics"]["window_samples"]=8; cfg["stride"]=8
    csv=_write_sequence_fixture(tmp_path/"fixture2",n=1600)
    b=make_bundle(csv,cfg,sequence_length=8,eval_start=15,proposal_features=True)
    idx=b.features.index("sg_excitation_score")
    assert np.all((b.Xtr[:,:,idx]>=0)&(b.Xtr[:,:,idx]<=1))
    assert np.allclose(b.Xtr[:,-1,idx],b.etr)


def test_pfr_bundle_adds_label_free_mechanics_and_excitation_channels(tmp_path):
    import yaml
    from safegrip.benchmark import make_bundle
    from pathlib import Path

    cfg = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs/kaggle_small.yaml").read_text())
    cfg["benchmark"]["common_warmup_samples"] = 16
    cfg["stride"] = 8
    cfg.setdefault("physics", {})["window_samples"] = 8
    csv = _write_sequence_fixture(tmp_path / "fixture_pfr", n=1600)
    b = make_bundle(csv, cfg, sequence_length=8, eval_start=15, feature_mode="pfr")
    assert "pfr_mechanics_lower" in b.features
    assert "pfr_excitation" in b.features
    ei = b.features.index("pfr_excitation")
    assert np.all((b.Xtr[:, :, ei] >= 0.0) & (b.Xtr[:, :, ei] <= 1.0))
    assert np.allclose(b.Xtr[:, -1, ei], b.etr)
