from __future__ import annotations

"""Shared finite-sample conformal utilities.

The active PFR-ECR path uses direct order-statistic selection rather than
mapping the desired rank back through ``numpy.quantile``.  This keeps the code
identical to the usual split-conformal finite-sample rank
``ceil((n + 1) * (1 - alpha))``.

``block_max_scores`` is a dependence-mitigation device for overlapping temporal
windows.  It does not by itself establish validity under arbitrary temporal
dependence; any population-coverage statement still requires an appropriate
exchangeability/independent-block assumption for the calibration and test
units being compared.
"""

import math
import numpy as np


def finite_sample_higher_quantile(values, alpha: float, *, min_samples: int = 1) -> float:
    """Return the split-conformal higher order statistic exactly.

    For ``n`` finite scores, this selects rank
    ``k = min(n, ceil((n + 1) * (1 - alpha)))`` using one-based ranks.
    """

    scores = np.asarray(values, dtype=float).reshape(-1)
    scores = scores[np.isfinite(scores)]
    min_samples = int(min_samples)
    if min_samples < 1:
        raise ValueError("min_samples must be >= 1")
    if len(scores) < min_samples:
        raise ValueError(f"insufficient finite calibration samples: {len(scores)} < {min_samples}")
    alpha = float(alpha)
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    n = len(scores)
    k = min(n, int(math.ceil((n + 1) * (1.0 - alpha))))
    return float(np.partition(scores, k - 1)[k - 1])


def _segment_key(endpoint_id: str) -> str:
    text = str(endpoint_id)
    parts = text.rsplit(":", 2)
    return parts[0] if len(parts) >= 3 else text


def block_max_scores(scores, endpoint_ids, block_size: int) -> np.ndarray:
    """Take one worst-case score per consecutive endpoint block per segment.

    This reduces pseudo-replication from strongly overlapping sequence windows.
    Blocks are formed only inside one trajectory/segment; no block crosses a
    segment boundary.  It is deliberately documented as dependence mitigation,
    not as a proof of arbitrary-dependence conformal validity.
    """

    scores = np.asarray(scores, dtype=float).reshape(-1)
    ids = np.asarray(endpoint_ids, dtype=str).reshape(-1)
    if len(scores) != len(ids):
        raise ValueError("scores and endpoint_ids must have the same length")
    block = max(1, int(block_size))
    out: list[float] = []
    start = 0
    while start < len(scores):
        key = _segment_key(ids[start])
        stop = start + 1
        while stop < len(scores) and _segment_key(ids[stop]) == key:
            stop += 1
        local = scores[start:stop]
        for i in range(0, len(local), block):
            finite = local[i:i + block]
            finite = finite[np.isfinite(finite)]
            if len(finite):
                out.append(float(np.max(finite)))
        start = stop
    return np.asarray(out, dtype=float)


def resolved_block_size(uq_cfg: dict | None, *, sequence_length: int, stride: int) -> int:
    """Resolve configured block size, deriving one from overlap when set to 0.

    ``block_size: 0`` means ``ceil(sequence_length / stride)`` endpoint scores,
    approximately one sequence length of endpoint overlap.
    """

    cfg = uq_cfg or {}
    configured = int(cfg.get("block_size", 0))
    if configured > 0:
        return configured
    return max(1, int(math.ceil(max(1, int(sequence_length)) / max(1, int(stride)))))
