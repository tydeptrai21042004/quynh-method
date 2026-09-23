from __future__ import annotations

"""Mathematical core for SafeGrip-PFR-ECR.

The active method uses a learned point estimate plus a positive scale, a
normalized one-sided conformal statistical lower estimate, and an independently
calibrated mechanics lower estimate.  Their maximum is the controller-facing
safe lower friction value.

Legacy projection helpers remain in this module so earlier PFR experiments and
tests are still reproducible; the active benchmark no longer uses broad
feasible-set clipping as its defining contribution.
"""

from dataclasses import dataclass

import numpy as np

from .physics import project_numpy


@dataclass(frozen=True)
class ProjectionAudit:
    """Pointwise diagnostics for the projection theorem."""

    prediction: np.ndarray
    covered: np.ndarray
    distance_to_set: np.ndarray
    raw_squared_error: np.ndarray
    projected_squared_error: np.ndarray
    squared_error_gain: np.ndarray
    theorem_margin: np.ndarray
    theorem_holds: np.ndarray


def residual_target(mu: np.ndarray, mechanics_lower_raw: np.ndarray) -> np.ndarray:
    """Signed train target ``mu - L0``.

    The target is intentionally *not* clipped.  This keeps learning independent
    of calibration and allows the network to compensate when the nominal
    mechanics model is imperfect on a training example.
    """

    mu = np.asarray(mu, dtype=float)
    lower = np.asarray(mechanics_lower_raw, dtype=float)
    if mu.shape != lower.shape:
        raise ValueError("mu and mechanics_lower_raw must have identical shapes")
    return mu - lower


def compose_raw_prediction(
    mechanics_lower_raw: np.ndarray,
    residual_prediction: np.ndarray,
) -> np.ndarray:
    """Compose the unconstrained residual estimator ``L0 + r_theta``."""

    lower = np.asarray(mechanics_lower_raw, dtype=float)
    residual = np.asarray(residual_prediction, dtype=float)
    if lower.shape != residual.shape:
        raise ValueError("mechanics lower and residual prediction must have identical shapes")
    return lower + residual


def pfr_project(
    raw_prediction: np.ndarray,
    calibrated_lower: np.ndarray,
    mu_upper: float | np.ndarray,
) -> np.ndarray:
    """Project a raw estimate onto the calibrated physical interval."""

    raw = np.asarray(raw_prediction, dtype=float)
    lower = np.asarray(calibrated_lower, dtype=float)
    upper = np.asarray(mu_upper, dtype=float)
    try:
        raw, lower, upper = np.broadcast_arrays(raw, lower, upper)
    except ValueError as exc:
        raise ValueError("raw_prediction, calibrated_lower and mu_upper are not broadcast compatible") from exc
    if np.any(lower > upper):
        raise ValueError("calibrated lower bound cannot exceed upper bound")
    return project_numpy(raw, lower, upper)


def pfr_from_residual(
    mechanics_lower_raw: np.ndarray,
    residual_prediction: np.ndarray,
    calibrated_lower: np.ndarray,
    mu_upper: float | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(raw_prediction, projected_prediction)`` for SafeGrip-PFR."""

    raw = compose_raw_prediction(mechanics_lower_raw, residual_prediction)
    return raw, pfr_project(raw, calibrated_lower, mu_upper)


def distance_to_interval(
    value: np.ndarray,
    lower: np.ndarray,
    upper: float | np.ndarray,
) -> np.ndarray:
    """Euclidean distance from scalar values to closed intervals."""

    x = np.asarray(value, dtype=float)
    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)
    try:
        x, lo, hi = np.broadcast_arrays(x, lo, hi)
    except ValueError as exc:
        raise ValueError("value, lower and upper are not broadcast compatible") from exc
    if np.any(lo > hi):
        raise ValueError("lower cannot exceed upper")
    return np.maximum(np.maximum(lo - x, x - hi), 0.0)


def projection_theorem_audit(
    y_true: np.ndarray,
    raw_prediction: np.ndarray,
    calibrated_lower: np.ndarray,
    mu_upper: float | np.ndarray,
    *,
    atol: float = 1e-10,
) -> ProjectionAudit:
    """Evaluate the exact pointwise SafeGrip-PFR projection inequality.

    On the coverage event ``y_true in [lower, upper]`` the theorem requires

        raw_sq_error - projected_sq_error - distance_to_set**2 >= 0.

    ``theorem_holds`` is reported only as a numerical implementation audit; the
    mathematical result follows from orthogonal projection onto a closed convex
    interval.
    """

    y = np.asarray(y_true, dtype=float)
    raw = np.asarray(raw_prediction, dtype=float)
    lo = np.asarray(calibrated_lower, dtype=float)
    hi = np.asarray(mu_upper, dtype=float)
    try:
        y, raw, lo, hi = np.broadcast_arrays(y, raw, lo, hi)
    except ValueError as exc:
        raise ValueError("audit arrays are not broadcast compatible") from exc
    if np.any(lo > hi):
        raise ValueError("calibrated lower bound cannot exceed upper bound")

    pred = pfr_project(raw, lo, hi)
    dist = distance_to_interval(raw, lo, hi)
    raw_sq = np.square(raw - y)
    proj_sq = np.square(pred - y)
    gain = raw_sq - proj_sq
    margin = gain - np.square(dist)
    covered = (y >= lo - atol) & (y <= hi + atol)
    holds = (~covered) | (margin >= -abs(float(atol)))

    return ProjectionAudit(
        prediction=pred,
        covered=covered,
        distance_to_set=dist,
        raw_squared_error=raw_sq,
        projected_squared_error=proj_sq,
        squared_error_gain=gain,
        theorem_margin=margin,
        theorem_holds=holds,
    )


def unsafe_overestimate_improvement_audit(
    y_true: np.ndarray,
    raw_prediction: np.ndarray,
    projected_prediction: np.ndarray,
    calibrated_lower: np.ndarray,
    mu_upper: float | np.ndarray,
    *,
    atol: float = 1e-10,
) -> np.ndarray:
    """Pointwise no-worsening audit for positive overestimation on covered samples."""

    y = np.asarray(y_true, dtype=float)
    raw = np.asarray(raw_prediction, dtype=float)
    pred = np.asarray(projected_prediction, dtype=float)
    lo = np.asarray(calibrated_lower, dtype=float)
    hi = np.asarray(mu_upper, dtype=float)
    y, raw, pred, lo, hi = np.broadcast_arrays(y, raw, pred, lo, hi)
    covered = (y >= lo - atol) & (y <= hi + atol)
    raw_over = np.maximum(raw - y, 0.0)
    pred_over = np.maximum(pred - y, 0.0)
    return (~covered) | (pred_over <= raw_over + abs(float(atol)))


@dataclass(frozen=True)
class SafetyFusionAudit:
    """Diagnostics for the revised mechanics/statistical safe lower estimate."""

    statistical_lower: np.ndarray
    fused_safe: np.ndarray
    physics_covered: np.ndarray
    statistical_covered: np.ndarray
    joint_component_covered: np.ndarray
    fused_covered: np.ndarray
    fusion_logic_holds: np.ndarray


def _finite_sample_higher_quantile(scores: np.ndarray, alpha: float) -> float:
    """Finite-sample split-conformal higher quantile for non-empty scores."""

    scores = np.asarray(scores, dtype=float)
    scores = scores[np.isfinite(scores)]
    if len(scores) == 0:
        return 0.0
    alpha = float(alpha)
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    level = min(1.0, np.ceil((len(scores) + 1) * (1.0 - alpha)) / len(scores))
    try:
        return float(np.quantile(scores, level, method="higher"))
    except TypeError:  # NumPy < 1.22
        return float(np.quantile(scores, level, interpolation="higher"))


def conformal_safe_correction(
    point_cal: np.ndarray,
    y_cal: np.ndarray,
    scale_cal: np.ndarray,
    *,
    alpha: float = 0.025,
    scale_floor: float = 1e-6,
) -> float:
    """One-sided normalized conformal correction for safe friction use.

    Scores are ``(mu_point - mu_true) / sigma``.  With a finite-sample higher
    quantile ``q`` the lower operational estimate ``mu_point - q*sigma`` has
    marginal one-sided split-conformal coverage under exchangeability.  The
    quantile is clipped below at zero so the safety correction never increases
    the point estimate; this can only make the estimate more conservative.
    """

    point = np.asarray(point_cal, dtype=float)
    y = np.asarray(y_cal, dtype=float)
    scale = np.maximum(np.asarray(scale_cal, dtype=float), float(scale_floor))
    try:
        point, y, scale = np.broadcast_arrays(point, y, scale)
    except ValueError as exc:
        raise ValueError("point_cal, y_cal and scale_cal must be broadcast compatible") from exc
    mask = np.isfinite(point) & np.isfinite(y) & np.isfinite(scale)
    scores = (point[mask] - y[mask]) / scale[mask]
    return max(0.0, _finite_sample_higher_quantile(scores, float(alpha)))


def statistical_safe_lower(
    point_prediction: np.ndarray,
    scale_prediction: np.ndarray,
    q_safe: float,
) -> np.ndarray:
    """Return the one-sided learned lower estimate ``mu_point - q*sigma``."""

    point = np.asarray(point_prediction, dtype=float)
    scale = np.maximum(np.asarray(scale_prediction, dtype=float), 1e-8)
    try:
        point, scale = np.broadcast_arrays(point, scale)
    except ValueError as exc:
        raise ValueError("point_prediction and scale_prediction must be broadcast compatible") from exc
    return point - max(0.0, float(q_safe)) * scale


def fuse_safe_lower(
    mechanics_lower: np.ndarray,
    statistical_lower: np.ndarray,
    mu_upper: float | np.ndarray,
) -> np.ndarray:
    """Fuse two lower estimates by taking the stronger one.

    If both component lower estimates are valid for a sample, their maximum is
    also a valid lower estimate.  Clipping to physical support ``[0,U]`` only
    decreases values above ``U`` and uses the physical assumption ``mu >= 0``.
    """

    mech = np.asarray(mechanics_lower, dtype=float)
    stat = np.asarray(statistical_lower, dtype=float)
    upper = np.asarray(mu_upper, dtype=float)
    try:
        mech, stat, upper = np.broadcast_arrays(mech, stat, upper)
    except ValueError as exc:
        raise ValueError("mechanics_lower, statistical_lower and mu_upper are not broadcast compatible") from exc
    if np.any(upper < 0):
        raise ValueError("mu_upper must be non-negative")
    fused = np.maximum(mech, stat)
    return np.minimum(np.maximum(fused, 0.0), upper)


def safety_fusion_audit(
    y_true: np.ndarray,
    mechanics_lower: np.ndarray,
    point_prediction: np.ndarray,
    scale_prediction: np.ndarray,
    q_safe: float,
    mu_upper: float | np.ndarray,
    *,
    atol: float = 1e-10,
) -> SafetyFusionAudit:
    """Audit the deterministic fusion logic behind the joint coverage claim."""

    y = np.asarray(y_true, dtype=float)
    mech = np.asarray(mechanics_lower, dtype=float)
    stat = statistical_safe_lower(point_prediction, scale_prediction, q_safe)
    upper = np.asarray(mu_upper, dtype=float)
    try:
        y, mech, stat, upper = np.broadcast_arrays(y, mech, stat, upper)
    except ValueError as exc:
        raise ValueError("safety audit arrays are not broadcast compatible") from exc
    fused = fuse_safe_lower(mech, stat, upper)
    physics_covered = mech <= y + abs(float(atol))
    statistical_covered = stat <= y + abs(float(atol))
    joint = physics_covered & statistical_covered
    fused_covered = fused <= y + abs(float(atol))
    # This is a deterministic algebraic audit, not an empirical proof of the
    # marginal conformal coverage assumptions.
    logic_holds = (~joint) | fused_covered
    return SafetyFusionAudit(
        statistical_lower=stat,
        fused_safe=fused,
        physics_covered=physics_covered,
        statistical_covered=statistical_covered,
        joint_component_covered=joint,
        fused_covered=fused_covered,
        fusion_logic_holds=logic_holds,
    )
