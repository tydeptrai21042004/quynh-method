from __future__ import annotations

"""Semantic normalization for calibrated physical-reference innovation learning.

For each semantic query, training data define one affine calibration of the
existing physical reference, ``a_q P_q(X) + b_q``, followed by robust residual
location/scale normalization.  When the physical reference is unavailable, the
same decoder falls back to a robust direct-target coordinate.

All stored quantities are deterministic buffers estimated from training data;
there are no learned dataset-specific parameters or branches.
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


def _robust_affine(reference: torch.Tensor, target: torch.Tensor, floor: float) -> tuple[float, float]:
    """Small deterministic Huber-IRLS fit for ``target ~= a*reference + b``.

    The identity reference is the safe fallback when the reference does not vary
    enough to identify a slope.  No external fitting package is required.
    """
    x = reference.detach().double().reshape(-1)
    y = target.detach().double().reshape(-1)
    finite = torch.isfinite(x) & torch.isfinite(y)
    x, y = x[finite], y[finite]
    if x.numel() < 2:
        bias = torch.median(y - x) if x.numel() else y.new_tensor(0.0)
        return 1.0, float(bias)

    x_med = torch.median(x)
    y_med = torch.median(y)
    dx = x - x_med
    denom = torch.sum(dx * dx)
    if not torch.isfinite(denom) or float(denom) <= floor * floor:
        return 1.0, float(torch.median(y - x))

    a = torch.sum(dx * (y - y_med)) / denom
    b = y_med - a * x_med
    if not torch.isfinite(a) or not torch.isfinite(b):
        return 1.0, float(torch.median(y - x))

    # A few IRLS iterations make the calibration resistant to isolated target or
    # reference outliers without introducing a tunable model component.
    for _ in range(6):
        residual = y - (a * x + b)
        _, scale = _robust_center_scale(residual.float(), max(floor, 1e-8))
        c = max(1.345 * scale, floor)
        abs_r = torch.abs(residual)
        w = torch.where(abs_r <= c, torch.ones_like(abs_r), c / abs_r.clamp_min(floor))
        sw = torch.sum(w)
        if not torch.isfinite(sw) or float(sw) <= floor:
            break
        mx = torch.sum(w * x) / sw
        my = torch.sum(w * y) / sw
        xc = x - mx
        var = torch.sum(w * xc * xc)
        if not torch.isfinite(var) or float(var) <= floor * floor:
            break
        new_a = torch.sum(w * xc * (y - my)) / var
        new_b = my - new_a * mx
        if not torch.isfinite(new_a) or not torch.isfinite(new_b):
            break
        a, b = new_a, new_b

    return float(a), float(b)


class SemanticInnovationNormalizer(nn.Module):
    """Query-semantic calibrated physical innovation coordinate."""

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
        # Identity initialization preserves the exact v3 behavior before fit().
        self.register_buffer("reference_gain", torch.ones(n, dtype=torch.float32))
        self.register_buffer("reference_bias", torch.zeros(n, dtype=torch.float32))
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
        by_key_ref_x: dict[int, list[torch.Tensor]] = {}
        by_key_ref_y: dict[int, list[torch.Tensor]] = {}

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
                    by_key_ref_x.setdefault(key, []).append(reference[obs_valid, j].detach().cpu().float())
                    by_key_ref_y.setdefault(key, []).append(batch.target_values[obs_valid, j].detach().cpu().float())

        for key, chunks in by_key_direct.items():
            c, s = _robust_center_scale(torch.cat(chunks), self.floor)
            self.center_direct[key] = c
            self.scale_direct[key] = s
            self.fitted_direct[key] = True

        for key, x_chunks in by_key_ref_x.items():
            x = torch.cat(x_chunks)
            y = torch.cat(by_key_ref_y[key])
            gain, bias = _robust_affine(x, y, max(self.floor * 1e-3, 1e-8))
            residual = y - (gain * x + bias)
            c, s = _robust_center_scale(residual, self.floor)
            self.reference_gain[key] = gain
            self.reference_bias[key] = bias
            self.center_ref[key] = c
            self.scale_ref[key] = s
            self.fitted_ref[key] = True

        return self

    def parameters_for(
        self,
        query_ids: torch.Tensor,
        reference_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return residual/direct center and scale for each sample/query."""
        keys = self._flat_index(query_ids)
        c_ref = self.center_ref[keys]
        s_ref = self.scale_ref[keys]
        c_dir = self.center_direct[keys]
        s_dir = self.scale_direct[keys]
        fit_ref = self.fitted_ref[keys]

        use_ref_stats = reference_mask.bool() & fit_ref
        center = torch.where(use_ref_stats, c_ref, c_dir)
        scale = torch.where(use_ref_stats, s_ref, s_dir).clamp_min(self.floor)
        return center, scale

    def calibrated_reference(
        self,
        query_ids: torch.Tensor,
        physical_reference: torch.Tensor,
        reference_mask: torch.Tensor,
    ) -> torch.Tensor:
        keys = self._flat_index(query_ids)
        gain = self.reference_gain[keys]
        bias = self.reference_bias[keys]
        base = gain * physical_reference + bias
        return torch.where(reference_mask.bool(), base, torch.zeros_like(base))

    def reference_parameters_for(self, query_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Expose deterministic calibration coefficients for audits/tests."""
        keys = self._flat_index(query_ids)
        return self.reference_gain[keys], self.reference_bias[keys]

    def normalize_target(
        self,
        target: torch.Tensor,
        query_ids: torch.Tensor,
        physical_reference: torch.Tensor,
        reference_mask: torch.Tensor,
    ) -> torch.Tensor:
        center, scale = self.parameters_for(query_ids, reference_mask)
        base = self.calibrated_reference(query_ids, physical_reference, reference_mask)
        return (target - base - center) / scale

    def reconstruct(
        self,
        normalized_innovation: torch.Tensor,
        query_ids: torch.Tensor,
        physical_reference: torch.Tensor,
        reference_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        center, scale = self.parameters_for(query_ids, reference_mask)
        base = self.calibrated_reference(query_ids, physical_reference, reference_mask)
        point = base + center + scale * normalized_innovation
        return point, center, scale
