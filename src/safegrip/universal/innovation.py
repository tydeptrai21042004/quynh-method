from __future__ import annotations

"""Relationally Anchored Normalized Physical Innovation (RA-NPI).

The neural decoder always predicts a dimensionless innovation.  When a semantic
physical reference is observable, RA-NPI keeps the successful v3 identity
coordinate as the prior and estimates only one deterministic query-semantic gain
``alpha_q`` from TRAINING data:

    y_hat = alpha_q P_q(X) + c_q + s_q R_theta(X, q)

The gain is robustly fitted with an identity anchor ``(alpha_q - 1)^2``.  Thus
v3 is the preferred coordinate unless the training evidence consistently
supports a scale correction.  No dataset identity, dataset-specific branch, or
trainable calibration parameter is introduced.

When a reference is not observable (including sensor dropout), the same decoder
uses the training-only robust direct-target coordinate.  This is mask algebra,
not a dataset/task architecture branch.
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


def _anchored_robust_gain(
    reference: torch.Tensor,
    target: torch.Tensor,
    target_scale: float,
    *,
    anchor_strength: float,
    floor: float,
) -> float:
    """Fit one robust gain anchored to the v3 identity value ``alpha=1``.

    The fit is performed on centered reference/target variation, while the
    residual location is handled later by ``center_ref``.  Both variables are
    divided by the target scale so one global anchor strength has comparable
    meaning across physical targets.  A small Huber-IRLS loop solves the scalar
    anchored problem without adding a learned model component.
    """
    x = reference.detach().double().reshape(-1)
    y = target.detach().double().reshape(-1)
    finite = torch.isfinite(x) & torch.isfinite(y)
    x, y = x[finite], y[finite]
    if x.numel() < 2:
        return 1.0

    # Centering makes the scalar gain describe variation rather than absorbing
    # an offset.  The offset remains part of the robust innovation center c_q.
    x0 = x - torch.median(x)
    y0 = y - torch.median(y)
    scale = max(float(target_scale), floor)
    xs = x0 / scale
    ys = y0 / scale

    energy = torch.mean(xs * xs)
    if not torch.isfinite(energy) or float(energy) <= floor * floor:
        return 1.0

    lam = max(float(anchor_strength), 0.0)
    alpha = x.new_tensor(1.0)
    huber_delta = 1.345

    for _ in range(8):
        residual = ys - alpha * xs
        abs_r = torch.abs(residual)
        weights = torch.where(
            abs_r <= huber_delta,
            torch.ones_like(abs_r),
            huber_delta / abs_r.clamp_min(floor),
        )
        numerator = torch.mean(weights * xs * ys) + lam
        denominator = torch.mean(weights * xs * xs) + lam
        if not torch.isfinite(numerator) or not torch.isfinite(denominator):
            return 1.0
        if float(denominator) <= floor:
            return 1.0
        new_alpha = numerator / denominator
        if not torch.isfinite(new_alpha):
            return 1.0
        if float(torch.abs(new_alpha - alpha)) < 1e-8:
            alpha = new_alpha
            break
        alpha = new_alpha

    return float(alpha)


class SemanticInnovationNormalizer(nn.Module):
    """Query-semantic RA-NPI coordinate fitted on training data only."""

    def __init__(self, *, floor: float = 1e-3, anchor_strength: float = 10.0):
        super().__init__()
        sizes = OntologySizes()
        self._sizes = (sizes.quantities, sizes.axes, sizes.locations, sizes.target_types)
        n = math.prod(self._sizes)
        self.floor = float(floor)
        self.anchor_strength = float(anchor_strength)
        if self.anchor_strength < 0:
            raise ValueError("anchor_strength must be non-negative")

        self.register_buffer("center_ref", torch.zeros(n, dtype=torch.float32))
        self.register_buffer("scale_ref", torch.ones(n, dtype=torch.float32))
        self.register_buffer("center_direct", torch.zeros(n, dtype=torch.float32))
        self.register_buffer("scale_direct", torch.ones(n, dtype=torch.float32))
        # alpha=1 is exactly the v3 physical-reference coordinate before fit().
        self.register_buffer("reference_gain", torch.ones(n, dtype=torch.float32))
        # Retained as a zero buffer for API/checkpoint audit compatibility.  The
        # final RA-NPI base has no free affine bias; offset belongs to center_ref.
        self.register_buffer("reference_bias", torch.zeros(n, dtype=torch.float32))
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

        # Direct target statistics are also the universal scale used by the
        # anchored gain objective, so they are fitted first.
        for key, chunks in by_key_direct.items():
            c, s = _robust_center_scale(torch.cat(chunks), self.floor)
            self.center_direct[key] = c
            self.scale_direct[key] = s
            self.fitted_direct[key] = True

        for key, x_chunks in by_key_ref_x.items():
            x = torch.cat(x_chunks)
            y = torch.cat(by_key_ref_y[key])
            target_scale = float(self.scale_direct[key]) if bool(self.fitted_direct[key]) else 1.0
            gain = _anchored_robust_gain(
                x,
                y,
                target_scale,
                anchor_strength=self.anchor_strength,
                floor=max(self.floor * 1e-3, 1e-8),
            )
            residual = y - gain * x
            c, s = _robust_center_scale(residual, self.floor)
            self.reference_gain[key] = gain
            self.reference_bias[key] = 0.0
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
        base = gain * physical_reference
        return torch.where(reference_mask.bool(), base, torch.zeros_like(base))

    def reference_parameters_for(self, query_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Expose deterministic RA-NPI gain and zero bias for audits/tests."""
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
