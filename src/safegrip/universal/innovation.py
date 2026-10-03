from __future__ import annotations

"""Semantic normalization for physical-reference innovation learning.

The normalizer contains *no trainable parameters*.  For each semantic query it
stores robust location/scale statistics estimated from training data only.
Two coordinates are retained:

* reference available: normalize the physical residual ``y - P_q(X)``;
* reference unavailable: normalize the direct target ``y``.

This makes channel-dropout and real missing-sensor cases mathematically
consistent: the decoder always predicts an O(1), dimensionless quantity rather
than being forced to alternate between a small residual and a large absolute
state in the same coordinate system.
"""

from dataclasses import dataclass
import math
import torch
from torch import nn

from .ontology import OntologySizes
from .reference import PhysicalReferenceOperator


@dataclass(frozen=True)
class RobustInnovationStats:
    center_ref: torch.Tensor
    scale_ref: torch.Tensor
    center_direct: torch.Tensor
    scale_direct: torch.Tensor


def _robust_center_scale(values: torch.Tensor, floor: float) -> tuple[float, float]:
    """Median/MAD location-scale with a standard-deviation fallback.

    MAD can be exactly zero for quantized/discrete targets.  In that case the
    population standard deviation is used before applying the numerical floor.
    """
    v = values.detach().float().reshape(-1)
    v = v[torch.isfinite(v)]
    if v.numel() == 0:
        return 0.0, 1.0
    center = torch.median(v)
    mad = torch.median(torch.abs(v - center))
    scale = 1.4826 * mad
    if not torch.isfinite(scale) or float(scale) < floor:
        std = v.std(unbiased=False) if v.numel() > 1 else v.new_tensor(0.0)
        if torch.isfinite(std):
            scale = torch.maximum(std, v.new_tensor(float(floor)))
        else:
            scale = v.new_tensor(float(floor))
    return float(center), float(scale)


class SemanticInnovationNormalizer(nn.Module):
    """Query-semantic, training-only normalization of physical innovations.

    The lookup table is indexed by the four ontology IDs composing a
    ``PhysicalQuery``.  It is therefore independent of dataset identity and
    adds no learned task head or dataset-specific branch.
    """

    def __init__(self, *, floor: float = 1e-3):
        super().__init__()
        sizes = OntologySizes()
        self._sizes = (sizes.quantities, sizes.axes, sizes.locations, sizes.target_types)
        n = math.prod(self._sizes)
        self.floor = float(floor)
        self.register_buffer("center_ref", torch.zeros(n, dtype=torch.float32))
        self.register_buffer("scale_ref", torch.ones(n, dtype=torch.float32))
        self.register_buffer("center_direct", torch.zeros(n, dtype=torch.float32))
        self.register_buffer("scale_direct", torch.ones(n, dtype=torch.float32))
        self.register_buffer("fitted_ref", torch.zeros(n, dtype=torch.bool))
        self.register_buffer("fitted_direct", torch.zeros(n, dtype=torch.bool))

    def _flat_index(self, query_ids: torch.Tensor) -> torch.Tensor:
        if query_ids.shape[-1] != 4:
            raise ValueError("query_ids must end in four ontology IDs")
        nq, na, nl, nt = self._sizes
        q, a, l, t = [query_ids[..., k].long() for k in range(4)]
        return (((q * na + a) * nl + l) * nt + t).long()

    @torch.no_grad()
    def fit(self, batches) -> "SemanticInnovationNormalizer":
        if not batches:
            raise ValueError("batches cannot be empty")
        reference_op = PhysicalReferenceOperator()
        by_key_direct: dict[int, list[torch.Tensor]] = {}
        by_key_ref: dict[int, list[torch.Tensor]] = {}

        for batch in batches:
            reference, observed = reference_op(batch.sensors, batch.queries)
            qids = torch.tensor([q.ids() for q in batch.queries], dtype=torch.long)
            keys = self._flat_index(qids)
            for j, key_t in enumerate(keys):
                key = int(key_t)
                valid = batch.target_mask[:, j].bool() & torch.isfinite(batch.target_values[:, j])
                if not torch.any(valid):
                    continue
                y = batch.target_values[valid, j].detach().cpu().float()
                by_key_direct.setdefault(key, []).append(y)

                obs_valid = valid & observed[:, j].bool() & torch.isfinite(reference[:, j])
                if torch.any(obs_valid):
                    residual = (
                        batch.target_values[obs_valid, j] - reference[obs_valid, j]
                    ).detach().cpu().float()
                    by_key_ref.setdefault(key, []).append(residual)

        # Direct statistics always provide a principled fallback for reference
        # dropout/missingness. Reference statistics are used only when a
        # physical reference is actually observed.
        for key, chunks in by_key_direct.items():
            c, s = _robust_center_scale(torch.cat(chunks), self.floor)
            self.center_direct[key] = c
            self.scale_direct[key] = s
            self.fitted_direct[key] = True

        for key, chunks in by_key_ref.items():
            c, s = _robust_center_scale(torch.cat(chunks), self.floor)
            self.center_ref[key] = c
            self.scale_ref[key] = s
            self.fitted_ref[key] = True

        return self

    def parameters_for(
        self,
        query_ids: torch.Tensor,
        reference_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return center/scale in the coordinate active for each sample/query."""
        keys = self._flat_index(query_ids)
        c_ref = self.center_ref[keys]
        s_ref = self.scale_ref[keys]
        c_dir = self.center_direct[keys]
        s_dir = self.scale_direct[keys]
        fit_ref = self.fitted_ref[keys]

        use_ref = reference_mask.bool() & fit_ref
        center = torch.where(use_ref, c_ref, c_dir)
        scale = torch.where(use_ref, s_ref, s_dir).clamp_min(self.floor)
        return center, scale

    def normalize_target(
        self,
        target: torch.Tensor,
        query_ids: torch.Tensor,
        physical_reference: torch.Tensor,
        reference_mask: torch.Tensor,
    ) -> torch.Tensor:
        center, scale = self.parameters_for(query_ids, reference_mask)
        base = torch.where(reference_mask.bool(), physical_reference, torch.zeros_like(physical_reference))
        return (target - base - center) / scale

    def reconstruct(
        self,
        normalized_innovation: torch.Tensor,
        query_ids: torch.Tensor,
        physical_reference: torch.Tensor,
        reference_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        center, scale = self.parameters_for(query_ids, reference_mask)
        base = torch.where(reference_mask.bool(), physical_reference, torch.zeros_like(physical_reference))
        point = base + center + scale * normalized_innovation
        return point, center, scale
