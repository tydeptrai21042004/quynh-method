from __future__ import annotations

from collections import Counter
import torch
from torch.utils.data import WeightedRandomSampler


def multidataset_sample_weights(domain_ids, tau: float = 0.5) -> torch.Tensor:
    """Per-sample weights yielding P(domain=d) proportional to N_d**tau."""
    if not 0.0 <= tau <= 1.0:
        raise ValueError("tau must lie in [0,1]")
    ids = list(domain_ids)
    if not ids:
        raise ValueError("domain_ids cannot be empty")
    counts = Counter(ids)
    return torch.tensor([counts[d] ** (tau - 1.0) for d in ids], dtype=torch.double)


def make_multidataset_sampler(domain_ids, tau: float = 0.5, num_samples: int | None = None, replacement: bool = True):
    weights = multidataset_sample_weights(domain_ids, tau=tau)
    return WeightedRandomSampler(weights, int(num_samples or len(weights)), replacement=replacement)
