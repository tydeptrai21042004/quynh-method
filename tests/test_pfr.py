import numpy as np

from safegrip.pfr import (
    compose_raw_prediction,
    conformal_safe_correction,
    distance_to_interval,
    fuse_safe_lower,
    pfr_from_residual,
    pfr_project,
    projection_theorem_audit,
    residual_target,
    safety_fusion_audit,
    statistical_safe_lower,
    unsafe_overestimate_improvement_audit,
)
from safegrip.physics import apply_lower_correction, conformal_lower_correction


def test_residual_parameterization_is_exact_reparameterization_of_friction_error():
    y = np.array([0.3, 0.7, 1.0])
    lower0 = np.array([0.1, 0.4, 0.8])
    residual_true = residual_target(y, lower0)
    residual_pred = np.array([0.25, 0.20, 0.15])
    mu_raw = compose_raw_prediction(lower0, residual_pred)
    assert np.allclose((residual_pred - residual_true) ** 2, (mu_raw - y) ** 2)


def test_pfr_projection_is_equivalent_to_clipping_residual_against_feasible_interval():
    lower0 = np.array([0.40, 0.55, 0.70])
    residual = np.array([-0.25, 0.20, 1.10])
    lower_alpha = np.array([0.30, 0.50, 0.65])
    raw, pred = pfr_from_residual(lower0, residual, lower_alpha, 1.3)
    assert np.allclose(raw, lower0 + residual)
    assert np.allclose(pred, np.clip(raw, lower_alpha, 1.3))


def test_projection_theorem_no_harm_on_covered_samples():
    rng = np.random.default_rng(7)
    lo = rng.uniform(0.0, 0.8, size=1000)
    hi = np.full(1000, 1.3)
    y = rng.uniform(lo, hi)
    raw = rng.uniform(-0.4, 1.8, size=1000)
    audit = projection_theorem_audit(y, raw, lo, hi)
    assert audit.covered.all()
    assert audit.theorem_holds.all()
    assert np.all(audit.projected_squared_error <= audit.raw_squared_error + 1e-12)
    assert np.all(audit.squared_error_gain + 1e-12 >= audit.distance_to_set ** 2)


def test_projection_is_strictly_better_when_raw_is_outside_and_truth_is_not_boundary_match():
    y = np.array([0.6, 0.9])
    raw = np.array([0.1, 1.6])
    lo = np.array([0.4, 0.5])
    audit = projection_theorem_audit(y, raw, lo, 1.3)
    assert np.all(audit.squared_error_gain > 0)
    assert np.all(audit.distance_to_set > 0)


def test_distance_to_interval():
    x = np.array([0.1, 0.5, 1.6])
    lo = np.array([0.3, 0.3, 0.3])
    assert np.allclose(distance_to_interval(x, lo, 1.3), [0.2, 0.0, 0.3])


def test_conformal_relaxation_only_moves_lower_endpoint_downward():
    lower = np.array([0.50, 0.65, 0.80, 0.90])
    y = np.array([0.45, 0.70, 0.75, 0.95])
    q = conformal_lower_correction(lower, y, alpha=0.25)
    corrected = apply_lower_correction(lower, q)
    assert q >= 0.0
    assert np.all(corrected <= lower + 1e-12)


def test_projection_cannot_increase_positive_overestimate_on_covered_samples():
    y = np.array([0.5, 0.8, 1.0])
    lo = np.array([0.2, 0.4, 0.7])
    raw = np.array([1.6, 0.95, 0.2])
    pred = pfr_project(raw, lo, 1.3)
    ok = unsafe_overestimate_improvement_audit(y, raw, pred, lo, 1.3)
    assert ok.all()


def test_normalized_conformal_safe_correction_is_nonnegative_and_conservative():
    point = np.array([0.70, 0.72, 0.74, 0.76, 0.78])
    y = np.array([0.71, 0.70, 0.77, 0.75, 0.80])
    scale = np.full(5, 0.02)
    q = conformal_safe_correction(point, y, scale, alpha=0.2)
    lower = statistical_safe_lower(point, scale, q)
    assert q >= 0.0
    assert np.all(lower <= point + 1e-12)


def test_fused_safe_lower_is_maximum_of_valid_component_lower_bounds():
    y = np.array([0.6, 0.8, 1.0])
    mechanics = np.array([0.4, 0.5, 0.7])
    statistical = np.array([0.55, 0.75, 0.95])
    fused = fuse_safe_lower(mechanics, statistical, 1.3)
    assert np.allclose(fused, statistical)
    assert np.all(fused <= y)


def test_safety_fusion_logic_holds_whenever_both_components_are_covered():
    y = np.array([0.6, 0.8, 1.0, 0.7])
    mechanics = np.array([0.4, 0.5, 0.7, 0.65])
    point = np.array([0.63, 0.85, 1.08, 0.75])
    scale = np.array([0.03, 0.04, 0.05, 0.03])
    audit = safety_fusion_audit(y, mechanics, point, scale, q_safe=2.0, mu_upper=1.3)
    assert audit.fusion_logic_holds.all()
    assert np.all(audit.fused_safe[audit.joint_component_covered] <= y[audit.joint_component_covered] + 1e-12)
