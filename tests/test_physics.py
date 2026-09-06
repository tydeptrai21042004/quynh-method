import numpy as np
from safegrip.physics import robust_force_utilization_lower, project_numpy, conformal_lower_correction, apply_lower_correction

def test_robust_bound_reduces_with_uncertainty():
    a=robust_force_utilization_lower([3000],[4000],[10000],0,0)[0]
    b=robust_force_utilization_lower([3000],[4000],[10000],500,500)[0]
    assert abs(a-.5)<1e-9 and b<a

def test_projection_dominance_when_truth_in_interval():
    rng=np.random.default_rng(0)
    for _ in range(1000):
        lo=rng.uniform(0,.8); hi=rng.uniform(max(lo,.9),1.3); y=rng.uniform(lo,hi); x=rng.uniform(-.5,2)
        p=project_numpy(x,lo,hi)
        assert abs(p-y) <= abs(x-y)+1e-12

def test_conformal_only_relaxes_physical_lower():
    lo=np.array([.6,.7,.8]); y=np.array([.5,.65,.75])
    q=conformal_lower_correction(lo,y,.1)
    assert q>=0
    assert np.all(apply_lower_correction(lo,q)<=lo+1e-12)
