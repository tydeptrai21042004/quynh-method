from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from safegrip.pfr import conformal_safe_correction, statistical_safe_lower, fuse_safe_lower


@dataclass
class TargetDomainCalibrator:
    """Target/domain conditional one-sided ECR calibration.

    ``min_calibration_size`` is explicit and enforced.  A missing or undersized
    calibration split is an error, never a zero-width correction.
    """

    alpha: float = 0.05
    min_calibration_size: int = 2
    corrections: dict[tuple[str, str], float] = field(default_factory=dict)
    sample_counts: dict[tuple[str, str], int] = field(default_factory=dict)

    def fit(self, target: str, domain: str, point, truth, scale) -> float:
        point_arr = np.asarray(point, dtype=float)
        truth_arr = np.asarray(truth, dtype=float)
        scale_arr = np.asarray(scale, dtype=float)
        try:
            point_arr, truth_arr, scale_arr = np.broadcast_arrays(point_arr, truth_arr, scale_arr)
        except ValueError as exc:
            raise ValueError("point, truth and scale must be broadcast compatible") from exc
        n = int(np.sum(np.isfinite(point_arr) & np.isfinite(truth_arr) & np.isfinite(scale_arr)))
        if n < int(self.min_calibration_size):
            raise ValueError(f"insufficient calibration samples for {target}/{domain}: {n} < {self.min_calibration_size}")
        q = conformal_safe_correction(
            point_arr, truth_arr, scale_arr, alpha=self.alpha,
            min_calibration_size=self.min_calibration_size,
        )
        key = (str(target), str(domain))
        self.corrections[key] = float(q)
        self.sample_counts[key] = n
        return float(q)

    def statistical_lower(self, target: str, domain: str, point, scale):
        key = (str(target), str(domain))
        if key not in self.corrections:
            raise KeyError(f"calibrator not fit for target/domain {key}")
        return statistical_safe_lower(point, scale, self.corrections[key])

    def fuse(self, target: str, domain: str, point, scale, mechanics_lower, mu_upper):
        stat = self.statistical_lower(target, domain, point, scale)
        return fuse_safe_lower(np.asarray(mechanics_lower), stat, mu_upper)
