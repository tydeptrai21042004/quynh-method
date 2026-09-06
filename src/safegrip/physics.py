from __future__ import annotations
import numpy as np
import torch


def robust_force_utilization_lower(fx, fy, fz, eps_t=0.0, eps_z=0.0):
    """Guaranteed lower friction-utilization bound under bounded force errors.

    If ||e_t|| <= eps_t and |e_z| <= eps_z, then
    mu >= max(0, ||Fhat_t||-eps_t)/(Fhat_z+eps_z).
    """
    fx=np.asarray(fx,float); fy=np.asarray(fy,float); fz=np.asarray(fz,float)
    num=np.maximum(0.0, np.hypot(fx,fy)-eps_t)
    den=np.maximum(np.abs(fz)+eps_z,1e-6)
    return num/den


def vehicle_level_lower_bound(ax, ay, speed, *, mass=1500.0, g=9.81, crr=0.012,
                              rho_air=1.225, cdA=0.66, accel_error=0.15,
                              external_force_margin=250.0, vertical_force_margin=250.0):
    """Conservative whole-vehicle lower grip bound from production signals.

    From ||sum F_tire|| >= max(0, m||a_xy|| - ||F_external||), and
    sum Fz <= mg + vertical margin. Aero/rolling/grade/model uncertainty is subtracted
    as an external-force upper bound, making the result deliberately conservative.
    """
    ax=np.asarray(ax,float); ay=np.asarray(ay,float); speed=np.asarray(speed,float)
    inertial=np.maximum(0.0, mass*(np.hypot(ax,ay)-abs(accel_error)))
    drag=0.5*rho_air*cdA*np.square(np.nan_to_num(speed,nan=0.0))
    ext=np.abs(drag)+crr*mass*g+abs(external_force_margin)
    tang=np.maximum(0.0,inertial-ext)
    fz_up=mass*g+abs(vertical_force_margin)
    return np.clip(tang/max(fz_up,1e-6),0.0,2.0)


def identified_interval(lower, mu_upper=1.3):
    lo=np.clip(np.asarray(lower,float),0.0,mu_upper)
    hi=np.full_like(lo,float(mu_upper))
    return lo,hi


def project_numpy(mu, lower, upper):
    return np.minimum(np.maximum(np.asarray(mu),np.asarray(lower)),np.asarray(upper))


def project_torch(mu, lower, upper):
    return torch.minimum(torch.maximum(mu,lower),upper)


def conformal_lower_correction(lower_cal, y_cal, alpha=0.05):
    """One-sided split-conformal correction for lower bounds.

    Scores s_i = lower_i - y_i. q is the finite-sample (1-alpha) quantile.
    Returned q>=0 only relaxes the physical lower bound; it never tightens it.
    """
    lower=np.asarray(lower_cal,float); y=np.asarray(y_cal,float)
    scores=lower-y
    n=len(scores)
    if n==0: return 0.0
    level=min(1.0, np.ceil((n+1)*(1-alpha))/n)
    try: q=float(np.quantile(scores,level,method="higher"))
    except TypeError: q=float(np.quantile(scores,level,interpolation="higher"))
    return max(0.0,q)


def apply_lower_correction(lower, q):
    return np.maximum(0.0,np.asarray(lower,float)-float(q))
