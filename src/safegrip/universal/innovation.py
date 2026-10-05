from __future__ import annotations

"""Normalized Physical Innovation (NPI-v3) and controlled ablation coordinates.

NPI-v3 is the only active proposal:

    y_hat = P_q(X) + c_q,ref + s_q,ref * R_theta(X, q),

when the semantic physical reference is observable, and

    y_hat = c_q,direct + s_q,direct * R_theta(X, q),

when it is not. Centers/scales are robust TRAINING-only statistics indexed by
semantic query IDs, never dataset identity, and are buffers rather than
trainable parameters.

The alternate ``mode`` values in this module exist only to provide controlled
paper ablations of NPI-v3. They do not define additional proposal methods.
"""

import math
import torch
from torch import nn

from .ontology import OntologySizes
from .reference import PhysicalReferenceOperator


NPI_MODES = {
    "npi_v3",
    "direct_normalized",
    "reference_only",
    "old_physical_innovation",
    "npi_no_center",
    "npi_no_scale",
}


def _robust_center_scale(values: torch.Tensor, floor: float) -> tuple[float, float]:
    """Median/MAD location-scale with a standard-deviation fallback."""
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
    """Training-only semantic coordinate used by NPI-v3 and its ablations."""

    def __init__(
        self,
        *,
        floor: float = 1e-3,
        mode: str = "npi_v3",
        localization_reference: str = "curvature",
    ):
        super().__init__()
        mode = str(mode).strip().lower()
        if mode not in NPI_MODES:
            raise ValueError(f"unknown NPI mode: {mode}. Choices: {', '.join(sorted(NPI_MODES))}")
        self.mode = mode
        self.localization_reference = str(localization_reference).strip().lower()
        # Validate the reference mode at construction time.
        PhysicalReferenceOperator(localization_mode=self.localization_reference)

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
        _, na, nl, nt = self._sizes
        q, a, l, t = [query_ids[..., k].long() for k in range(4)]
        return (((q * na + a) * nl + l) * nt + t).long()

    @torch.no_grad()
    def fit(self, batches) -> "SemanticInnovationNormalizer":
        if not batches:
            raise ValueError("batches cannot be empty")
        reference_op = PhysicalReferenceOperator(localization_mode=self.localization_reference)
        by_key_ref: dict[int, list[torch.Tensor]] = {}
        by_key_direct: dict[int, list[torch.Tensor]] = {}

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

    def _effective_reference_mask(self, reference_mask: torch.Tensor) -> torch.Tensor:
        if self.mode == "direct_normalized":
            return torch.zeros_like(reference_mask, dtype=torch.bool)
        return reference_mask.bool()

    def parameters_for(
        self,
        query_ids: torch.Tensor,
        reference_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the center/scale selected by the controlled coordinate."""
        keys = self._flat_index(query_ids)
        c_ref = self.center_ref[keys]
        s_ref = self.scale_ref[keys]
        c_dir = self.center_direct[keys]
        s_dir = self.scale_direct[keys]
        fit_ref = self.fitted_ref[keys]

        effective_ref = self._effective_reference_mask(reference_mask)
        use_ref_stats = effective_ref & fit_ref
        center = torch.where(use_ref_stats, c_ref, c_dir)
        scale = torch.where(use_ref_stats, s_ref, s_dir).clamp_min(self.floor)

        if self.mode in {"old_physical_innovation", "reference_only"}:
            center = torch.zeros_like(center)
            scale = torch.ones_like(scale)
        elif self.mode == "npi_no_center":
            center = torch.zeros_like(center)
        elif self.mode == "npi_no_scale":
            scale = torch.ones_like(scale)

        return center, scale

    def reference_base(
        self,
        physical_reference: torch.Tensor,
        reference_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return the physical base term selected by this ablation coordinate."""
        effective_ref = self._effective_reference_mask(reference_mask)
        return torch.where(
            effective_ref, physical_reference, torch.zeros_like(physical_reference)
        )

    def normalize_target(
        self,
        target: torch.Tensor,
        query_ids: torch.Tensor,
        physical_reference: torch.Tensor,
        reference_mask: torch.Tensor,
    ) -> torch.Tensor:
        center, scale = self.parameters_for(query_ids, reference_mask)
        base = self.reference_base(physical_reference, reference_mask)
        return (target - base - center) / scale

    def reconstruct(
        self,
        normalized_innovation: torch.Tensor,
        query_ids: torch.Tensor,
        physical_reference: torch.Tensor,
        reference_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        center, scale = self.parameters_for(query_ids, reference_mask)
        base = self.reference_base(physical_reference, reference_mask)
        if self.mode == "reference_only":
            point = base
        else:
            point = base + center + scale * normalized_innovation
        return point, center, scale
