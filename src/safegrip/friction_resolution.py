from __future__ import annotations

"""Finite-window friction resolution certification (SafeGrip-FRC).

The primary estimator is deliberately small and theorem-aligned:

1. A causal response model predicts a standardized vehicle-response innovation
   under a hypothetical friction coefficient ``mu``.
2. For a finite observation horizon H, friction is estimated by grid search on
   the residual objective J_H(mu).
3. A grid separation margin S_H(delta) measures the minimum distance between
   predicted response signatures for grid coefficients separated by at least
   ``delta``.
4. If S_H(delta) > 2 r_H and the true nearest-grid response residual is at most
   r_H, then any grid minimizer lies within delta of the nearest grid point.
   Therefore the error to the continuous true coefficient is bounded by
   ``delta + grid_step / 2`` for a uniform grid.
5. The adaptive estimator chooses the horizon with the smallest finite
   certificate; if none is certified, it falls back to the configured horizon.

This module contains the mathematical/numerical core only.  It does not apply
physical clipping, learned gates, posterior temperatures, or auxiliary heads.
"""

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import torch


@dataclass(frozen=True)
class ResolutionResult:
    """Result for one horizon over a batch of windows."""

    horizon: int
    mu_hat: np.ndarray
    objective_min: np.ndarray
    separation: np.ndarray  # [N, D]
    delta_certificate: np.ndarray
    error_certificate: np.ndarray
    certified: np.ndarray


def make_mu_grid(mu_min: float, mu_max: float, step: float) -> np.ndarray:
    """Create an inclusive, deterministic one-dimensional friction grid."""
    mu_min = float(mu_min)
    mu_max = float(mu_max)
    step = float(step)
    if not np.isfinite([mu_min, mu_max, step]).all() or step <= 0 or mu_max <= mu_min:
        raise ValueError("Require finite mu_min < mu_max and step > 0")
    count = int(np.floor((mu_max - mu_min) / step + 1e-12))
    grid = mu_min + step * np.arange(count + 1, dtype=np.float64)
    if grid[-1] < mu_max - 1e-10:
        grid = np.r_[grid, mu_max]
    else:
        grid[-1] = mu_max
    return grid.astype(np.float32)


def nearest_grid_values(mu: np.ndarray, mu_grid: np.ndarray) -> np.ndarray:
    """Return nearest candidate-grid friction values for each input value."""
    mu = np.asarray(mu, dtype=float).reshape(-1)
    grid = np.asarray(mu_grid, dtype=float).reshape(-1)
    if grid.size < 2:
        raise ValueError("mu_grid must contain at least two points")
    idx = np.abs(mu[:, None] - grid[None, :]).argmin(axis=1)
    return grid[idx].astype(np.float32)


def inverse_objective(observed: np.ndarray, signatures: np.ndarray) -> np.ndarray:
    """Squared residual objective for every friction candidate.

    Parameters
    ----------
    observed:
        Shape ``[N, H, R]``.
    signatures:
        Shape ``[N, M, H, R]``.

    Returns
    -------
    np.ndarray
        Objective array with shape ``[N, M]``.
    """
    y = np.asarray(observed, dtype=np.float64)
    s = np.asarray(signatures, dtype=np.float64)
    if y.ndim != 3 or s.ndim != 4 or s.shape[0] != y.shape[0] or s.shape[2:] != y.shape[1:]:
        raise ValueError("Expected observed [N,H,R] and signatures [N,M,H,R]")
    return np.sum((s - y[:, None, :, :]) ** 2, axis=(2, 3))


def estimate_mu(observed: np.ndarray, signatures: np.ndarray, mu_grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Grid residual minimizer and its minimum objective."""
    obj = inverse_objective(observed, signatures)
    grid = np.asarray(mu_grid, dtype=np.float32)
    if obj.shape[1] != len(grid):
        raise ValueError("Candidate dimension does not match mu_grid")
    idx = np.argmin(obj, axis=1)
    return grid[idx], obj[np.arange(len(idx)), idx]


def _delta_masks(mu_grid: np.ndarray, deltas: Sequence[float], *, device: torch.device) -> list[torch.Tensor]:
    grid = torch.as_tensor(np.asarray(mu_grid, dtype=np.float32), device=device)
    diff = torch.abs(grid[:, None] - grid[None, :])
    masks = []
    for delta in deltas:
        mask = diff >= float(delta) - 1e-7
        # Diagonal never belongs for strictly positive delta, but keep the
        # implementation safe if a caller requests zero.
        mask.fill_diagonal_(False)
        if not bool(mask.any()):
            raise ValueError(f"No grid pair is separated by delta={delta}")
        masks.append(mask)
    return masks


def separation_margin(
    signatures: np.ndarray | torch.Tensor,
    mu_grid: np.ndarray,
    deltas: Sequence[float],
    *,
    batch_size: int = 128,
) -> np.ndarray:
    """Compute finite-grid response separation margins.

    ``signatures`` has shape ``[N, M, H, R]``.  For each sample and requested
    resolution ``delta`` this returns

        min ||Phi(mu_i)-Phi(mu_j)||_2  over |mu_i-mu_j| >= delta.

    Batched ``torch.cdist`` avoids explicitly materializing response-difference
    tensors for every candidate pair.
    """
    if isinstance(signatures, np.ndarray):
        x = torch.from_numpy(np.asarray(signatures, dtype=np.float32))
    else:
        x = signatures.detach().to(dtype=torch.float32, device="cpu")
    if x.ndim != 4:
        raise ValueError("signatures must have shape [N,M,H,R]")
    n, m, h, r = x.shape
    if m != len(mu_grid):
        raise ValueError("Candidate dimension does not match mu_grid")
    deltas = [float(d) for d in deltas]
    masks = _delta_masks(mu_grid, deltas, device=torch.device("cpu"))
    out = torch.empty((n, len(deltas)), dtype=torch.float32)
    inf = torch.tensor(float("inf"), dtype=torch.float32)
    for start in range(0, n, max(1, int(batch_size))):
        flat = x[start:start + batch_size].reshape(-1, m, h * r)
        dist = torch.cdist(flat, flat, p=2)
        for di, mask in enumerate(masks):
            vals = torch.where(mask.unsqueeze(0), dist, inf)
            out[start:start + len(flat), di] = vals.amin(dim=(1, 2))
    return out.numpy()


def resolution_certificate(
    separation: np.ndarray,
    deltas: Sequence[float],
    residual_radius: float,
    grid_step: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert separation margins into a finite-resolution certificate.

    The returned ``delta_certificate`` is the smallest requested delta whose
    separation exceeds ``2 * residual_radius``.  ``error_certificate`` adds the
    nearest-grid approximation term ``grid_step/2``.  Rows with no valid
    certificate are returned as ``inf`` and ``certified=False``.
    """
    sep = np.asarray(separation, dtype=float)
    ds = np.asarray(list(deltas), dtype=float)
    if sep.ndim != 2 or sep.shape[1] != len(ds):
        raise ValueError("separation must have shape [N,len(deltas)]")
    if np.any(np.diff(ds) < 0):
        raise ValueError("deltas must be sorted nondecreasing")
    valid = sep > 2.0 * float(residual_radius)
    first = np.argmax(valid, axis=1)
    certified = valid.any(axis=1)
    delta_cert = np.full(len(sep), np.inf, dtype=np.float32)
    delta_cert[certified] = ds[first[certified]].astype(np.float32)
    error_cert = delta_cert.copy()
    error_cert[certified] += float(grid_step) / 2.0
    return delta_cert, error_cert, certified


def select_horizon(
    results: Sequence[ResolutionResult],
    *,
    fallback_horizon: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Select the horizon with the smallest finite error certificate.

    If no horizon is certified for a sample, ``fallback_horizon`` is used.
    Returns ``mu_hat, selected_horizon, selected_error_certificate, certified``.
    """
    if not results:
        raise ValueError("At least one horizon result is required")
    n = len(results[0].mu_hat)
    if any(len(r.mu_hat) != n for r in results):
        raise ValueError("All horizon results must refer to the same endpoints")
    hs = np.asarray([int(r.horizon) for r in results], dtype=int)
    cert = np.column_stack([r.error_certificate for r in results])
    finite = np.isfinite(cert)
    masked = np.where(finite, cert, np.inf)
    choice = np.argmin(masked, axis=1)
    has = finite.any(axis=1)
    if fallback_horizon is None:
        fallback_horizon = int(hs.max())
    if int(fallback_horizon) not in set(hs.tolist()):
        raise ValueError("fallback_horizon must be one of the evaluated horizons")
    fallback_index = int(np.flatnonzero(hs == int(fallback_horizon))[0])
    choice = np.where(has, choice, fallback_index)
    rows = np.arange(n)
    mu_hat = np.column_stack([r.mu_hat for r in results])[rows, choice]
    selected_h = hs[choice]
    selected_cert = cert[rows, choice]
    return mu_hat.astype(np.float32), selected_h.astype(np.int32), selected_cert.astype(np.float32), has


def theorem_condition_holds(separation: np.ndarray, residual_radius: float) -> np.ndarray:
    """Boolean theorem condition ``S(delta) > 2 r`` for audit tables."""
    return np.asarray(separation, dtype=float) > 2.0 * float(residual_radius)


def empirical_residual_radius(residual_norms: np.ndarray, quantile: float = 0.95) -> float:
    """Training/calibration-only empirical residual radius."""
    x = np.asarray(residual_norms, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        raise ValueError("No finite residual norms")
    q = float(quantile)
    if not 0 < q <= 1:
        raise ValueError("quantile must be in (0,1]")
    # 'higher' is conservative and deterministic for finite calibration sets.
    try:
        return float(np.quantile(x, q, method="higher"))
    except TypeError:  # numpy<1.22 compatibility
        return float(np.quantile(x, q, interpolation="higher"))


def verify_grid_recovery_theorem(
    observed: np.ndarray,
    signatures: np.ndarray,
    mu_grid: np.ndarray,
    true_mu: np.ndarray,
    delta: float,
    residual_radius: float,
) -> dict:
    """Numerically audit the finite-grid theorem for a batch.

    The premise uses the nearest-grid true coefficient, exactly matching the
    implemented estimator.  This helper is intended for tests and result audits,
    not for training or model selection.
    """
    grid = np.asarray(mu_grid, dtype=np.float32)
    nearest = nearest_grid_values(true_mu, grid)
    nearest_idx = np.abs(np.asarray(true_mu)[:, None] - grid[None, :]).argmin(axis=1)
    obj = inverse_objective(observed, signatures)
    residual = np.sqrt(obj[np.arange(len(obj)), nearest_idx])
    mu_hat, _ = estimate_mu(observed, signatures, grid)
    sep = separation_margin(signatures, grid, [delta])[:, 0]
    premise = (residual <= float(residual_radius) + 1e-8) & (sep > 2.0 * float(residual_radius))
    nearest_error = np.abs(mu_hat - nearest)
    conclusion = nearest_error < float(delta) + 1e-7
    return {
        "premise": premise,
        "conclusion": conclusion,
        "violation": premise & ~conclusion,
        "nearest_grid_error": nearest_error,
        "continuous_error": np.abs(mu_hat - np.asarray(true_mu, dtype=float)),
        "residual_norm": residual,
        "separation": sep,
    }
