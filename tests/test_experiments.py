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


def test_statistics_use_per_seed_hierarchical_predictions_and_ignore_diagnostics(tmp_path):
    import numpy as np
    from safegrip.experiments import run_statistical_comparison
    results=tmp_path/"results"; results.mkdir()
    rows=[]
    # Two seeds, two independent segments, four endpoints per segment.
    for seed in (0,1):
        for seg in ("A","B"):
            for i in range(4):
                endpoint=f"{seg}:0:{i}"
                y=0.5+0.01*i
                rows.append({"endpoint_id":endpoint,"y_true":y,"model":"safegrip","seed":seed,"prediction":y+0.01})
                rows.append({"endpoint_id":endpoint,"y_true":y,"model":"baseline","seed":seed,"prediction":y+0.03})
    pd.DataFrame(rows).to_csv(results/"predictions_by_seed.csv",index=False)
    # Ensemble file intentionally contains a diagnostic column that must never be
    # treated as a model by the new statistics path.
    pd.DataFrame({"endpoint_id":[f"A:0:{i}" for i in range(4)],"y_true":[.5]*4,
                  "safegrip":[.5]*4,"safegrip_acceptance":[.9]*4}).to_csv(results/"predictions.csv",index=False)
    out=run_statistical_comparison(results,results/"statistics",bootstrap=100,seed=1)
    assert list(out.comparator)==["baseline"]
    assert int(out.n_seeds.iloc[0])==2
    assert int(out.n_segments_min.iloc[0])==2
    assert float(out.delta_rmse_proposal_minus_comparator.iloc[0])<0
