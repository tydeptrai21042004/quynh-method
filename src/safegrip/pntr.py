from __future__ import annotations

"""Physics-Neural Trust-Region (PNTR) utilities.

The proposal keeps a strong direct neural friction estimate as an anchor and
uses a friction-conditioned response model only inside a local physically
admissible trust region.

The local objective is

    F(mu) = E_H(mu) + lambda * ((mu - mu0) / tau)^2,

where E_H is standardized response mismatch, mu0 is the physics-neural
anchor and tau is the maximum physical correction radius.

The exact anchor is always an admissible fallback.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class PNTRChoice:
    mu_hat: np.ndarray
    candidate_mu: np.ndarray
    corrected: np.ndarray

    response_energy_anchor: np.ndarray
    response_energy_selected: np.ndarray

    regularized_objective_anchor: np.ndarray
    regularized_objective_selected: np.ndarray

    correction: np.ndarray


def regularized_local_choice(
    response_energy: np.ndarray,
    mu_grid: np.ndarray,
    anchor_mu: np.ndarray,
    lower: np.ndarray,
    upper: float | np.ndarray,
    *,
    trust_radius: float,
    anchor_lambda: float,
    anchor_response_energy: np.ndarray,
    min_response_improvement: float = 0.0,
    atol: float = 1e-12,
) -> PNTRChoice:

    e = np.asarray(
        response_energy,
        dtype=float,
    )

    grid = np.asarray(
        mu_grid,
        dtype=float,
    )

    anchor = np.asarray(
        anchor_mu,
        dtype=float,
    ).reshape(-1)

    lo = np.asarray(
        lower,
        dtype=float,
    ).reshape(-1)

    hi = np.asarray(
        upper,
        dtype=float,
    )

    if hi.ndim == 0:
        hi = np.full_like(
            anchor,
            float(hi),
        )
    else:
        hi = hi.reshape(-1)

    base_e = np.asarray(
        anchor_response_energy,
        dtype=float,
    ).reshape(-1)

    if e.ndim != 2:
        raise ValueError(
            "response_energy must be 2-D"
        )

    if e.shape != (
        len(anchor),
        len(grid),
    ):
        raise ValueError(
            "response_energy must have shape [N, M]"
        )

    if (
        len(lo) != len(anchor)
        or len(hi) != len(anchor)
        or len(base_e) != len(anchor)
    ):
        raise ValueError(
            "endpoint arrays must have the same length"
        )

    tau = float(trust_radius)
    lam = float(anchor_lambda)

    if tau <= 0:
        raise ValueError(
            "trust_radius must be positive"
        )

    if lam < 0:
        raise ValueError(
            "anchor_lambda must be non-negative"
        )

    # --------------------------------------------------------
    # Candidate displacement from neural-physics anchor
    # --------------------------------------------------------

    delta = (
        grid[None, :]
        -
        anchor[:, None]
    )

    # --------------------------------------------------------
    # Mechanical + trust-region admissibility
    # --------------------------------------------------------

    admissible = (
        (
            grid[None, :]
            >=
            lo[:, None] - atol
        )
        &
        (
            grid[None, :]
            <=
            hi[:, None] + atol
        )
        &
        (
            np.abs(delta)
            <=
            tau + atol
        )
    )

    # --------------------------------------------------------
    # Neural-anchor penalty
    # --------------------------------------------------------

    penalty = (
        lam
        *
        np.square(
            delta / tau
        )
    )

    objective = np.where(
        admissible,
        e + penalty,
        np.inf,
    )

    idx = np.argmin(
        objective,
        axis=1,
    )

    rows = np.arange(
        len(anchor)
    )

    candidate = grid[idx]

    candidate_energy = e[
        rows,
        idx,
    ]

    candidate_objective = objective[
        rows,
        idx,
    ]

    # --------------------------------------------------------
    # Exact anchor is ALWAYS available as fallback.
    #
    # Accept physics only when:
    #
    #   1. regularized objective improves
    #   2. response energy also improves
    #
    # No target friction label is used here.
    # --------------------------------------------------------

    improve_objective = (
        candidate_objective
        <
        base_e - atol
    )

    improve_response = (
        candidate_energy
        <=
        base_e
        -
        float(min_response_improvement)
        +
        atol
    )

    accept = (
        np.isfinite(candidate_objective)
        &
        improve_objective
        &
        improve_response
    )

    selected = np.where(
        accept,
        candidate,
        anchor,
    )

    selected_energy = np.where(
        accept,
        candidate_energy,
        base_e,
    )

    selected_objective = np.where(
        accept,
        candidate_objective,
        base_e,
    )

    correction = (
        selected
        -
        anchor
    )

    return PNTRChoice(
        mu_hat=selected.astype(
            np.float32
        ),

        candidate_mu=candidate.astype(
            np.float32
        ),

        corrected=accept,

        response_energy_anchor=base_e.astype(
            np.float32
        ),

        response_energy_selected=selected_energy.astype(
            np.float32
        ),

        regularized_objective_anchor=base_e.astype(
            np.float32
        ),

        regularized_objective_selected=selected_objective.astype(
            np.float32
        ),

        correction=correction.astype(
            np.float32
        ),
    )


def local_identifiability(
    response_minus: np.ndarray,
    response_plus: np.ndarray,
    probe_delta: float,
    *,
    horizon: int | None = None,
) -> np.ndarray:

    a = np.asarray(
        response_minus,
        dtype=float,
    )

    b = np.asarray(
        response_plus,
        dtype=float,
    )

    if (
        a.shape != b.shape
        or a.ndim != 3
    ):
        raise ValueError(
            "responses must have shape [N,H,R]"
        )

    d = float(
        probe_delta
    )

    if d <= 0:
        raise ValueError(
            "probe_delta must be positive"
        )

    if horizon is not None:

        h = int(
            horizon
        )

        a = a[
            :,
            -h:,
            :
        ]

        b = b[
            :,
            -h:,
            :
        ]

    return (
        np.sqrt(
            np.mean(
                np.square(
                    b - a
                ),
                axis=(1, 2),
            )
        )
        /
        (
            2.0 * d
        )
    )


def strong_convexity_error_bound(
    anchor_error: np.ndarray | float,
    gradient_mismatch: np.ndarray | float,
    curvature: np.ndarray | float,
    anchor_lambda: float,
) -> np.ndarray:

    e0 = np.asarray(
        anchor_error,
        dtype=float,
    )

    eta = np.asarray(
        gradient_mismatch,
        dtype=float,
    )

    m = np.asarray(
        curvature,
        dtype=float,
    )

    lam = float(
        anchor_lambda
    )

    if (
        lam < 0
        or np.any(m < 0)
        or np.any(eta < 0)
        or np.any(e0 < 0)
    ):
        raise ValueError(
            "bound inputs must be non-negative"
        )

    return (
        eta
        +
        lam * e0
    ) / np.maximum(
        m + lam,
        1e-12,
    )