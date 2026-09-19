import numpy as np

from safegrip.pntr import (
    regularized_local_choice,
    local_identifiability,
    strong_convexity_error_bound,
)


def test_pntr_local_choice_keeps_exact_anchor_as_fallback():
    grid = np.array([0.2, 0.4, 0.6, 0.8], dtype=float)
    energy = np.array([[2.0, 0.9, 0.7, 0.1]], dtype=float)
    # Candidate 0.8 has much lower response energy, but it is outside tau=0.1.
    out = regularized_local_choice(
        energy, grid, np.array([0.5]), np.array([0.0]), 1.3,
        trust_radius=0.1, anchor_lambda=0.5,
        anchor_response_energy=np.array([0.6]),
    )
    assert np.isclose(out.mu_hat[0], 0.5)
    assert not out.corrected[0]
    assert out.regularized_objective_selected[0] <= out.regularized_objective_anchor[0] + 1e-12


def test_pntr_correction_is_trust_region_bounded_and_objective_safe():
    grid = np.arange(0.0, 1.01, 0.05)
    anchor = np.array([0.50, 0.70])
    # Quadratic physical energies with nearby minima.
    energy = np.stack([(grid - 0.55) ** 2, (grid - 0.65) ** 2], axis=0)
    base_e = np.array([(0.50 - 0.55) ** 2, (0.70 - 0.65) ** 2])
    out = regularized_local_choice(
        energy, grid, anchor, np.zeros(2), 1.3,
        trust_radius=0.10, anchor_lambda=0.01,
        anchor_response_energy=base_e,
    )
    assert np.all(np.abs(out.correction) <= 0.100001)
    assert np.all(out.regularized_objective_selected <= out.regularized_objective_anchor + 1e-12)


def test_local_identifiability_matches_linear_map():
    # Phi(mu) = [2mu, -mu], repeated over H=3. Difference between plus/minus
    # with d=.1 is 2d*[2,-1], whose RMS divided by 2d is sqrt((4+1)/2).
    d = 0.1
    minus = np.zeros((2, 3, 2), dtype=float)
    plus = np.tile(np.array([4*d, -2*d]), (2, 3, 1))
    ident = local_identifiability(minus, plus, d)
    assert np.allclose(ident, np.sqrt(2.5))


def test_strong_convexity_bound_improves_anchor_when_physics_is_informative():
    b = strong_convexity_error_bound(anchor_error=0.10, gradient_mismatch=0.01, curvature=4.0, anchor_lambda=1.0)
    assert np.isclose(b, 0.022)
    assert b < 0.10


def test_physics_slack_anchor_cannot_go_below_mechanics_bound():
    import torch
    from safegrip.pntr_benchmark import AnchorModel, predict_anchor

    class Dummy(torch.nn.Module):
        def forward(self, x):
            # negative standardized slack -> ReLU must force zero slack
            return torch.full((x.shape[0],), -10.0, dtype=x.dtype, device=x.device)

    anchor = AnchorModel(model=Dummy(), target_mean=0.1, target_std=0.01,
                         best_val_mse=0.0, target_mode="physics_slack")
    X = np.zeros((3, 4, 2), dtype=np.float32)
    lower = np.array([0.25, 0.50, 0.90], dtype=np.float32)
    pred = predict_anchor(anchor, X, lower, mu_upper=1.3)
    assert np.all(pred >= lower - 1e-7)
    assert np.allclose(pred, lower)


def test_physics_slack_anchor_respects_upper_bound():
    import torch
    from safegrip.pntr_benchmark import AnchorModel, predict_anchor

    class Dummy(torch.nn.Module):
        def forward(self, x):
            return torch.full((x.shape[0],), 10.0, dtype=x.dtype, device=x.device)

    anchor = AnchorModel(model=Dummy(), target_mean=1.0, target_std=0.1,
                         best_val_mse=0.0, target_mode="physics_slack")
    X = np.zeros((2, 4, 2), dtype=np.float32)
    lower = np.array([0.4, 0.8], dtype=np.float32)
    pred = predict_anchor(anchor, X, lower, mu_upper=1.3)
    assert np.all(pred <= 1.3 + 1e-7)
    assert np.all(pred >= lower - 1e-7)
