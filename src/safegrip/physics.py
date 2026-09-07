from __future__ import annotations

import numpy as np
import torch


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

    The implementation uses

        ||sum F_tire|| >= max(0, m (||a_xy|| - eps_a) - F_external^max)

    and the upper normal-load surrogate ``m g + vertical_force_margin``. Rolling
    resistance, aerodynamic drag and an explicit external-force margin are
    subtracted from the tangential demand. Therefore the returned value is a
    *conditional* lower bound when the configured uncertainty margins dominate
    omitted effects (grade, mass error, sensor bias, etc.).
    """
    ax = np.asarray(ax, float)
    ay = np.asarray(ay, float)
    speed = np.asarray(speed, float)
    mass = float(mass)
    g = float(g)
    inertial = np.maximum(0.0, mass * (np.hypot(ax, ay) - abs(float(accel_error))))
    drag = 0.5 * float(rho_air) * float(cdA) * np.square(np.nan_to_num(speed, nan=0.0))
    ext = np.abs(drag) + abs(float(crr)) * mass * g + abs(float(external_force_margin))
    tang = np.maximum(0.0, inertial - ext)
    fz_up = mass * g + abs(float(vertical_force_margin))
    return np.clip(tang / max(fz_up, 1e-6), 0.0, 2.0)


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


def conformal_lower_correction(lower_cal, y_cal, alpha=0.05):
    """One-sided split-conformal relaxation for a physics-derived lower bound.

    Scores are ``s_i = lower_i - y_i``. The finite-sample higher quantile is
    used, and ``q`` is clipped below at zero so calibration can only relax the
    deterministic mechanics lower endpoint. Statistical coverage still relies
    on the usual calibration/exchangeability assumptions and is not a physical
    guarantee under arbitrary domain shift.
    """
    lower = np.asarray(lower_cal, float)
    y = np.asarray(y_cal, float)
    mask = np.isfinite(lower) & np.isfinite(y)
    lower = lower[mask]
    y = y[mask]
    scores = lower - y
    n = len(scores)
    if n == 0:
        return 0.0
    level = min(1.0, np.ceil((n + 1) * (1 - float(alpha))) / n)
    try:
        q = float(np.quantile(scores, level, method="higher"))
    except TypeError:  # NumPy < 1.22 compatibility
        q = float(np.quantile(scores, level, interpolation="higher"))
    return max(0.0, q)


def apply_lower_correction(lower, q):
    return np.maximum(0.0, np.asarray(lower, float) - float(q))
