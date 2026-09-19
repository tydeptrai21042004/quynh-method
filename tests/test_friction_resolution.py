import numpy as np

from safegrip.friction_resolution import (
    make_mu_grid,
    estimate_mu,
    separation_margin,
    resolution_certificate,
    select_horizon,
    ResolutionResult,
    verify_grid_recovery_theorem,
)


def test_linear_response_separation_matches_closed_form():
    grid = make_mu_grid(0.0, 1.0, 0.1)
    # Phi(mu)=[2mu, -mu] repeated over one transition.
    sig = np.stack([np.array([2*m, -m], np.float32) for m in grid], axis=0)[None, :, None, :]
    sep = separation_margin(sig, grid, [0.2, 0.5])[0]
    # Because the grid is exact, minimum spacing >= delta equals delta here.
    assert np.isclose(sep[0], np.sqrt(5.0)*0.2, rtol=1e-5, atol=1e-6)
    assert np.isclose(sep[1], np.sqrt(5.0)*0.5, rtol=1e-5, atol=1e-6)


def test_separation_is_monotone_in_requested_resolution():
    rng = np.random.default_rng(4)
    grid = make_mu_grid(0.0, 1.0, 0.1)
    sig = rng.normal(size=(3, len(grid), 4, 2)).astype(np.float32)
    sep = separation_margin(sig, grid, [0.1, 0.2, 0.4, 0.6])
    assert np.all(np.diff(sep, axis=1) >= -1e-6)


def test_resolution_certificate_includes_half_grid_step():
    sep = np.array([[0.1, 0.4, 0.9], [0.05, 0.10, 0.15]], dtype=float)
    dc, ec, ok = resolution_certificate(sep, [0.1, 0.2, 0.3], residual_radius=0.15, grid_step=0.02)
    assert ok.tolist() == [True, False]
    assert np.isclose(dc[0], 0.2)
    assert np.isclose(ec[0], 0.21)
    assert np.isinf(dc[1]) and np.isinf(ec[1])


def test_grid_recovery_theorem_has_no_violation_on_exact_map():
    grid = make_mu_grid(0.0, 1.0, 0.1)
    true = np.array([0.2, 0.5, 0.8], dtype=np.float32)
    # Two-dimensional exact map Phi(mu) = [mu, 2mu].
    sig0 = np.stack([np.array([m, 2*m], np.float32) for m in grid], axis=0)[:, None, :]
    sig = np.repeat(sig0[None, ...], len(true), axis=0)
    obs = np.stack([np.array([[m, 2*m]], np.float32) for m in true], axis=0)
    audit = verify_grid_recovery_theorem(obs, sig, grid, true, delta=0.2, residual_radius=1e-6)
    assert audit["premise"].all()
    assert not audit["violation"].any()


def test_horizon_selection_prefers_smallest_finite_certificate_and_fallback():
    n = 3
    def r(h, cert):
        cert=np.asarray(cert,np.float32)
        return ResolutionResult(h,np.full(n,h/100,dtype=np.float32),np.zeros(n),np.zeros((n,1)),cert,cert,np.isfinite(cert))
    a=r(4,[0.2,np.inf,np.inf]); b=r(8,[0.1,0.3,np.inf]); c=r(16,[0.15,0.2,np.inf])
    mu,h,ec,ok=select_horizon([a,b,c],fallback_horizon=16)
    assert h.tolist()==[8,16,16]
    assert ok.tolist()==[True,True,False]
    assert np.isclose(mu[0],0.08) and np.isclose(mu[1],0.16)


def test_estimate_mu_recovers_grid_point():
    grid=make_mu_grid(0,1,0.1)
    sig=np.stack([np.array([[m,m*m]],np.float32) for m in grid],axis=0)[None,...]
    observed=np.array([[[0.4,0.16]]],np.float32)
    mu,obj=estimate_mu(observed,sig,grid)
    assert np.isclose(mu[0],0.4)
    assert obj[0] < 1e-10
