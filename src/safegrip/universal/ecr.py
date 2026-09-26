from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from safegrip.pfr import conformal_safe_correction, statistical_safe_lower, fuse_safe_lower


@dataclass
class TargetDomainCalibrator:
    """Target/domain conditional wrapper around the existing one-sided ECR core.

    The predictive model stays dataset-independent.  Calibration keys describe
    the deployment target/distribution and are intentionally outside the neural
    forward path.
    """

    alpha: float = 0.05
    corrections: dict[tuple[str, str], float] = field(default_factory=dict)

    def fit(self, target: str, domain: str, point, truth, scale) -> float:
        q = conformal_safe_correction(point, truth, scale, alpha=self.alpha)
        self.corrections[(str(target), str(domain))] = float(q)
        return float(q)

    def statistical_lower(self, target: str, domain: str, point, scale):
        key = (str(target), str(domain))
        if key not in self.corrections:
            raise KeyError(f"calibrator not fit for target/domain {key}")
        return statistical_safe_lower(point, scale, self.corrections[key])

    def fuse(self, target: str, domain: str, point, scale, mechanics_lower, mu_upper):
        stat = self.statistical_lower(target, domain, point, scale)
        return fuse_safe_lower(np.asarray(mechanics_lower), stat, mu_upper)
