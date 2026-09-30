from __future__ import annotations

import numpy as np
import torch

from .conformal import block_max_scores, finite_sample_higher_quantile


def robust_force_utilization_lower(fx, fy, fz, eps_t=0.0, eps_z=0.0):
    """Conservative friction-utilization lower bound under bounded force errors.

    If ``||e_t|| <= eps_t`` and ``|e_z| <= eps_z``, then the measured force
    utilization implies

        mu >= max(0, ||Fhat_t|| - eps_t) / (|Fhat_z| + eps_z).

    The statement is conditional on the stated bounded-error assumptions; this
    function does not claim that the bounds themselves are known exactly.
    """
    fx = np.asarray(fx, float)
    fy = np.asarray(fy, float)
    fz = np.asarray(fz, float)
    num = np.maximum(0.0, np.hypot(fx, fy) - abs(float(eps_t)))
    den = np.maximum(np.abs(fz) + abs(float(eps_z)), 1e-6)
    return num / den


def vehicle_level_lower_bound(
    ax,
    ay,
    speed,
    *,
    mass=1500.0,
    g=9.81,
    crr=0.012,
    rho_air=1.225,
    cdA=0.66,
    accel_error=0.15,
    external_force_margin=250.0,
    vertical_force_margin=250.0,
):
    """Conditional conservative whole-vehicle lower grip bound.

    v0.7 uses a vector force balance instead of subtracting nominal road loads
    from the *magnitude* of inertial acceleration.  For forward speed, the
    nominal longitudinal tire-force demand is

        F_x,tire ~= m a_x + F_drag + F_rr,
        F_y,tire ~= m a_y.

    If the horizontal acceleration-vector error is bounded by ``accel_error``
    and the remaining unmodelled horizontal forces are bounded in norm by
    ``external_force_margin``, reverse triangle inequality gives

        ||F_tire|| >= max(0,
            ||[m a_x + F_drag + F_rr, m a_y]||
            - m*eps_a - eps_F).

    Dividing by the conservative upper normal-load surrogate ``m g + eps_z``
    yields a conditional lower bound on available friction.  The validity claim
    remains conditional on a level-road/forward-motion approximation and on the
    configured uncertainty margins dominating omitted effects such as grade,
    mass error and sensor bias.  No arbitrary hard clipping is applied here so
    unit/schema failures remain visible to the preprocessing audit.
    """
    ax = np.asarray(ax, float)
    ay = np.asarray(ay, float)
    speed = np.asarray(speed, float)
    mass = float(mass)
    g = float(g)
    v = np.nan_to_num(speed, nan=0.0)
    drag = 0.5 * float(rho_air) * float(cdA) * np.square(v)
    rolling = abs(float(crr)) * mass * g
    # Direction is relevant only for the nominal longitudinal road load.  LiRA
    # uses positive forward speed; zero speed receives no signed road-load term.
    direction = np.sign(v)
    fx_nom = mass * np.nan_to_num(ax, nan=0.0) + direction * (drag + rolling)
    fy_nom = mass * np.nan_to_num(ay, nan=0.0)
    nominal_demand = np.hypot(fx_nom, fy_nom)
    uncertainty = mass * abs(float(accel_error)) + abs(float(external_force_margin))
    tang = np.maximum(0.0, nominal_demand - uncertainty)
    fz_up = mass * g + abs(float(vertical_force_margin))
    return np.maximum(0.0, tang / max(fz_up, 1e-6))


def identified_interval(lower, mu_upper=1.3):
    """Return the admissible interval [lower, mu_upper] under A1: mu <= mu_upper."""
    upper = float(mu_upper)
    lo = np.clip(np.asarray(lower, float), 0.0, upper)
    hi = np.full_like(lo, upper)
    return lo, hi


def window_identified_lower(lower, window: int):
    """Sliding-window maximum used as the lower endpoint for a fixed context."""
    x = np.asarray(lower, float)
    w = max(1, int(window))
    if w == 1:
        return x.copy()
    out = np.empty_like(x)
    for i in range(len(x)):
        out[i] = np.nanmax(x[max(0, i - w + 1) : i + 1])
    return out


def nested_identified_lower(lower):
    """Lower endpoints for nested windows W_1 subset W_2 ... .

    Because each endpoint is the maximum over all observations seen so far,
    this sequence is non-decreasing. It mirrors the theorem used in the paper
    without confusing it with a sliding window whose oldest sample can leave.
    """
    x = np.asarray(lower, float)
    if len(x) == 0:
        return x.copy()
    return np.maximum.accumulate(np.nan_to_num(x, nan=-np.inf))


def project_numpy(mu, lower, upper):
    return np.minimum(np.maximum(np.asarray(mu), np.asarray(lower)), np.asarray(upper))


def project_torch(mu, lower, upper):
    return torch.minimum(torch.maximum(mu, lower), upper)


def project_interval_numpy(low, high, lower, upper):
    """Project both endpoints of an interval onto a closed admissible interval."""
    low = np.asarray(low)
    high = np.asarray(high)
    a = np.minimum(low, high)
    b = np.maximum(low, high)
    plo = project_numpy(a, lower, upper)
    phi = project_numpy(b, lower, upper)
    return np.minimum(plo, phi), np.maximum(plo, phi)


def project_interval_torch(low, high, lower, upper):
    a = torch.minimum(low, high)
    b = torch.maximum(low, high)
    plo = project_torch(a, lower, upper)
    phi = project_torch(b, lower, upper)
    return torch.minimum(plo, phi), torch.maximum(plo, phi)


def gaussian_interval(mean, sigma, z=1.96):
    mean = np.asarray(mean, float)
    sigma = np.maximum(np.asarray(sigma, float), 1e-8)
    half = float(z) * sigma
    return mean - half, mean + half


def physics_truncated_gaussian_interval(mean, sigma, lower, upper, z=1.96):
    raw_low, raw_high = gaussian_interval(mean, sigma, z=z)
    return project_interval_numpy(raw_low, raw_high, lower, upper)


def conformal_lower_correction(lower_cal, y_cal, alpha=0.05, *, endpoint_ids=None, block_size=1):
    """One-sided split-conformal relaxation for a mechanics-derived anchor.

    Scores are ``s_i = anchor_i - y_i``. The trailing-window maximum is an
    *anchor*, not an unconditional deterministic lower bound for the current
    friction unless a local temporal-persistence assumption is supplied. The
    finite-sample correction therefore calibrates observed anchor violations.

    When ``endpoint_ids`` are provided, one worst-case score is retained per
    consecutive block within each trajectory segment. This mitigates the
    pseudo-replication caused by overlapping windows but does not establish
    arbitrary-dependence validity.
    """
    lower = np.asarray(lower_cal, float)
    y = np.asarray(y_cal, float)
    if lower.shape != y.shape:
        raise ValueError("lower_cal and y_cal must have identical shapes")
    mask = np.isfinite(lower) & np.isfinite(y)
    scores = lower[mask] - y[mask]
    if endpoint_ids is not None:
        ids = np.asarray(endpoint_ids, dtype=str)
        if ids.shape != lower.shape:
            raise ValueError("endpoint_ids must match lower_cal")
        scores = block_max_scores(scores, ids[mask], block_size)
    if len(scores) == 0:
        return 0.0
    return max(0.0, finite_sample_higher_quantile(scores, float(alpha)))


def apply_lower_correction(lower, q):
    return np.maximum(0.0, np.asarray(lower, float) - float(q))
