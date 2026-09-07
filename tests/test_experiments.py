from pathlib import Path
import pandas as pd

from safegrip.experiments import run_force_validation


def test_force_validation_robust_bound_is_not_above_nominal(tmp_path):
    src=tmp_path/"forces.csv"
    pd.DataFrame({"fx":[1000,2000],"fy":[0,1000],"fz":[5000,6000]}).to_csv(src,index=False)
    out=tmp_path/"out"
    m=run_force_validation(src,out,{"vehicle":{}},eps_t=100,eps_z=100)
    assert float(m.robust_not_above_nominal_rate.iloc[0])==1.0
    assert (out/"force_validation_samples.csv").exists()
