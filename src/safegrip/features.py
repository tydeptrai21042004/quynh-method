from __future__ import annotations

import numpy as np
import pandas as pd

ENGINEERED_PREFIX = "sg_"


def _safe_group_derivative(df: pd.DataFrame, value: str) -> pd.Series:
    """Finite-difference derivative without crossing split/trajectory boundaries."""
    out = pd.Series(np.zeros(len(df), dtype=float), index=df.index)
    if value not in df or "time" not in df:
        return out
    group_cols = [c for c in ("segment_id", "split") if c in df]
    if not group_cols:
        group_cols = [c for c in ("trip_id", "split") if c in df]
    groups = df.groupby(group_cols, sort=False, dropna=False) if group_cols else [("all", df)]
    for _, g in groups:
        g = g.sort_values("time", kind="stable") if g["time"].notna().any() else g.sort_index(kind="stable")
        v = pd.to_numeric(g[value], errors="coerce").to_numpy(float)
        t = pd.to_numeric(g["time"], errors="coerce").to_numpy(float)
        dv = np.diff(v, prepend=np.nan)
        dt = np.diff(t, prepend=np.nan)
        deriv = np.divide(dv, dt, out=np.zeros_like(dv), where=np.isfinite(dt) & (dt > 1e-6))
        deriv[~np.isfinite(deriv)] = 0.0
        out.loc[g.index] = deriv
    return out


def add_safegrip_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Add label-free, boundary-safe dynamics features for the SafeGrip proposal only.

    Literature baselines continue to receive the original prepared sensor channels.
    All scales below are fixed configuration constants; no train/validation/test
    statistics or friction labels are used to construct these features.
    """
    z = df.copy()
    fcfg = cfg.get("feature_engineering", {})
    g = float(cfg.get("vehicle", {}).get("gravity", 9.81))
    eps = 1e-6

    ax = pd.to_numeric(z.get("ax", 0.0), errors="coerce").fillna(0.0) if isinstance(z.get("ax", 0.0), pd.Series) else pd.Series(0.0, index=z.index)
    ay = pd.to_numeric(z.get("ay", 0.0), errors="coerce").fillna(0.0) if isinstance(z.get("ay", 0.0), pd.Series) else pd.Series(0.0, index=z.index)
    acc_mag = np.hypot(ax.to_numpy(float), ay.to_numpy(float))
    z["sg_accel_mag"] = acc_mag
    z["sg_accel_util"] = acc_mag / max(g, eps)

    jx = _safe_group_derivative(z, "ax")
    jy = _safe_group_derivative(z, "ay")
    z["sg_jerk_x"] = jx
    z["sg_jerk_y"] = jy
    z["sg_jerk_mag"] = np.hypot(jx.to_numpy(float), jy.to_numpy(float))

    wheel_cols = [c for c in ("wheel_fl", "wheel_fr", "wheel_rl", "wheel_rr") if c in z]
    if len(wheel_cols) >= 2:
        wheels = z[wheel_cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        wmean = np.nanmean(wheels, axis=1)
        denom = np.maximum(np.abs(wmean), 1.0)
        z["sg_wheel_spread"] = (np.nanmax(wheels, axis=1) - np.nanmin(wheels, axis=1)) / denom
        if all(c in z for c in ("wheel_fl", "wheel_fr", "wheel_rl", "wheel_rr")):
            front = 0.5 * (pd.to_numeric(z.wheel_fl, errors="coerce").to_numpy(float) + pd.to_numeric(z.wheel_fr, errors="coerce").to_numpy(float))
            rear = 0.5 * (pd.to_numeric(z.wheel_rl, errors="coerce").to_numpy(float) + pd.to_numeric(z.wheel_rr, errors="coerce").to_numpy(float))
            left = 0.5 * (pd.to_numeric(z.wheel_fl, errors="coerce").to_numpy(float) + pd.to_numeric(z.wheel_rl, errors="coerce").to_numpy(float))
            right = 0.5 * (pd.to_numeric(z.wheel_fr, errors="coerce").to_numpy(float) + pd.to_numeric(z.wheel_rr, errors="coerce").to_numpy(float))
            z["sg_wheel_front_rear"] = (front - rear) / denom
            z["sg_wheel_left_right"] = (left - right) / denom
    else:
        z["sg_wheel_spread"] = 0.0
        z["sg_wheel_front_rear"] = 0.0
        z["sg_wheel_left_right"] = 0.0

    pressure_cols = [c for c in ("pressure_fl", "pressure_fr", "pressure_rl", "pressure_rr") if c in z]
    if len(pressure_cols) >= 2:
        prs = z[pressure_cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        pmean = np.nanmean(prs, axis=1)
        z["sg_pressure_spread"] = (np.nanmax(prs, axis=1) - np.nanmin(prs, axis=1)) / np.maximum(np.abs(pmean), 1.0)
    else:
        z["sg_pressure_spread"] = 0.0

    torque = pd.to_numeric(z["torque"], errors="coerce").fillna(0.0).to_numpy(float) if "torque" in z else np.zeros(len(z), dtype=float)
    torque_ref = max(float(fcfg.get("torque_ref_nm", 500.0)), eps)
    z["sg_torque_util"] = np.abs(torque) / torque_ref

    # Excitation is deliberately label-free. It is a bounded proxy for how much
    # instantaneous dynamics can reveal about the available tire-road friction.
    acc_ref = max(float(fcfg.get("accel_ref_g", 0.25)), eps)
    wheel_ref = max(float(fcfg.get("wheel_spread_ref", 0.03)), eps)
    jerk_ref = max(float(fcfg.get("jerk_ref_ms3", 5.0)), eps)
    acc_term = np.clip(z["sg_accel_util"].to_numpy(float) / acc_ref, 0.0, 1.0)
    wheel_term = np.clip(np.abs(z["sg_wheel_spread"].to_numpy(float)) / wheel_ref, 0.0, 1.0)
    torque_term = np.clip(z["sg_torque_util"].to_numpy(float), 0.0, 1.0)
    jerk_term = np.clip(z["sg_jerk_mag"].to_numpy(float) / jerk_ref, 0.0, 1.0)
    weights = np.asarray([
        float(fcfg.get("weight_accel", 0.45)),
        float(fcfg.get("weight_wheel", 0.25)),
        float(fcfg.get("weight_torque", 0.15)),
        float(fcfg.get("weight_jerk", 0.15)),
    ], dtype=float)
    if not np.isfinite(weights).all() or weights.sum() <= 0:
        weights = np.asarray([0.45, 0.25, 0.15, 0.15], dtype=float)
    weights = weights / weights.sum()
    score = weights[0] * acc_term + weights[1] * wheel_term + weights[2] * torque_term + weights[3] * jerk_term
    z["sg_excitation_score"] = np.clip(score, 0.0, 1.0)

    engineered = [c for c in z.columns if c.startswith(ENGINEERED_PREFIX)]
    z[engineered] = z[engineered].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return z
