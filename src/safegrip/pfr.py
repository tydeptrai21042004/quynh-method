from __future__ import annotations

"""Mathematical core of SafeGrip-PFR.

SafeGrip-PFR separates the estimator into three objects:

1. a mechanics predictor ``L0(x)`` that captures a known structured component;
2. a learned signed residual ``r_theta(x)`` trained only on the training split;
3. a calibrated feasible interval ``[L_alpha(x), U]`` used only at inference.

The raw estimate is

    mu_tilde = L0 + r_theta,

and the final estimate is the Euclidean projection

    mu_hat = Pi_[L_alpha,U](mu_tilde).

For every sample whose true friction belongs to the feasible interval, projection
satisfies the pointwise Pythagorean inequality

    |mu_hat-mu*|^2 <= |mu_tilde-mu*|^2 - dist(mu_tilde,[L_alpha,U])^2.

The implementation below contains only deterministic algebra.  Statistical
coverage of ``L_alpha`` is handled by ``physics.conformal_lower_correction``.
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
