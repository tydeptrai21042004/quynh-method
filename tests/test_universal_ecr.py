import numpy as np

from safegrip.universal.ecr import TargetDomainCalibrator


def test_target_domain_calibrator_is_separate_by_domain():
    c = TargetDomainCalibrator(alpha=0.2)
    point = np.array([0.8, 0.9, 0.7, 0.85])
    truth = np.array([0.75, 0.8, 0.68, 0.8])
    scale = np.full(4, 0.1)
    c.fit("friction", "a", point, truth, scale)
    assert ("friction", "a") in c.corrections
    lo = c.statistical_lower("friction", "a", point, scale)
    assert np.all(lo <= point + 1e-12)
