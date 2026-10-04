from __future__ import annotations

"""Training-only semantic standardization and relational sensor contrast.

Two parameter-free transforms are used before the unchanged token encoder:

1) ``SemanticFeatureNormalizer`` standardizes patch descriptors with TRAINING
   statistics indexed only by physical sensor semantics
   (quantity, axis, location, unit class).
2) ``SemanticRelationalContrast`` preserves each descriptor's semantic-group
   mean while exposing its deviation from same-semantic sensors at the same
   physical patch time:

       d* = d + (d - mean_group(d)).

For singleton groups the transform is exactly the identity.  Both transforms are
permutation equivariant and never use dataset identity or arbitrary channel
position.  Raw canonical endpoint/RMS/integral values in ``UniversalSensorBatch``
remain untouched, so the physical-reference operator still works in physical
units.
"""

import math
import torch
from torch import nn

from .ontology import OntologySizes


class SemanticFeatureNormalizer(nn.Module):
    """Per-semantic-type feature standardization fitted on training data only."""

    def __init__(self, feature_dim: int, *, floor: float = 1e-8):
        super().__init__()
        if feature_dim < 1:
            raise ValueError("feature_dim must be positive")
        sizes = OntologySizes()
        self._sizes = (sizes.quantities, sizes.axes, sizes.locations, sizes.unit_classes)
        n = math.prod(self._sizes)
        self.feature_dim = int(feature_dim)
        self.floor = float(floor)
        self.register_buffer("center", torch.zeros((n, feature_dim), dtype=torch.float32))
        self.register_buffer("scale", torch.ones((n, feature_dim), dtype=torch.float32))
        self.register_buffer("fitted", torch.zeros(n, dtype=torch.bool))

    def _flat_index(
        self,
        quantity_ids: torch.Tensor,
        axis_ids: torch.Tensor,
        location_ids: torch.Tensor,
        unit_class_ids: torch.Tensor,
    ) -> torch.Tensor:
        _, na, nl, nu = self._sizes
        q = quantity_ids.long()
        a = axis_ids.long()
        l = location_ids.long()
        u = unit_class_ids.long()
        return (((q * na + a) * nl + l) * nu + u).long()

    @torch.no_grad()
    def fit(self, batches) -> "SemanticFeatureNormalizer":
        if not batches:
            raise ValueError("batches cannot be empty")

        n_groups = self.center.shape[0]
        sums = torch.zeros((n_groups, self.feature_dim), dtype=torch.float64)
        sums2 = torch.zeros_like(sums)
        counts = torch.zeros(n_groups, dtype=torch.float64)

        for batch in batches:
            sensors = batch.sensors if hasattr(batch, "sensors") else batch
            features = sensors.features.detach().cpu().double()
            if features.shape[-1] != self.feature_dim:
                raise ValueError("feature dimension does not match normalizer")
            keys = self._flat_index(
                sensors.quantity_ids.detach().cpu(),
                sensors.axis_ids.detach().cpu(),
                sensors.location_ids.detach().cpu(),
                sensors.unit_class_ids.detach().cpu(),
            )
            valid_tokens = sensors.token_mask.detach().cpu().bool()
            for key_t in torch.unique(keys[valid_tokens]):
                key = int(key_t)
                mask = valid_tokens & (keys == key)
                x = features[mask]
                if x.numel() == 0:
                    continue
                finite_rows = torch.isfinite(x).all(dim=1)
                x = x[finite_rows]
                if x.numel() == 0:
                    continue
                sums[key] += x.sum(dim=0)
                sums2[key] += torch.square(x).sum(dim=0)
                counts[key] += float(x.shape[0])

        fitted = counts > 0
        if torch.any(fitted):
            idx = torch.where(fitted)[0]
            mean = sums[idx] / counts[idx].unsqueeze(1)
            var = sums2[idx] / counts[idx].unsqueeze(1) - torch.square(mean)
            std = torch.sqrt(torch.clamp(var, min=0.0))
            # Constant descriptor coordinates contain no scale information.
            # Leaving their divisor at one avoids amplifying numerical noise.
            std = torch.where(std >= self.floor, std, torch.ones_like(std))
            self.center[idx] = mean.float()
            self.scale[idx] = std.float()
            self.fitted[idx] = True
        return self

    def forward(
        self,
        features: torch.Tensor,
        quantity_ids: torch.Tensor,
        axis_ids: torch.Tensor,
        location_ids: torch.Tensor,
        unit_class_ids: torch.Tensor,
    ) -> torch.Tensor:
        keys = self._flat_index(quantity_ids, axis_ids, location_ids, unit_class_ids)
        center = self.center[keys].to(features)
        scale = self.scale[keys].to(features).clamp_min(self.floor)
        fitted = self.fitted[keys].to(features.device).unsqueeze(-1)
        normalized = (features - center) / scale
        return torch.where(fitted, normalized, features)


class SemanticRelationalContrast(nn.Module):
    """Expose same-semantic sensor differences without channel IDs or parameters.

    Tokens are grouped by physical semantics AND patch time.  Static context
    tokens and padded/dropped tokens are left unchanged.  Within each visible
    sensor group, ``d* = 2d - mean(d)`` preserves the group mean and makes the
    relative sensor evidence explicit.  A one-token group maps exactly to itself.
    """

    _TIME_QUANTIZATION = 1_000_000.0  # microsecond key; only groups equal patch times

    def __init__(self):
        super().__init__()
        sizes = OntologySizes()
        self._sizes = (sizes.quantities, sizes.axes, sizes.locations, sizes.unit_classes)

    def _semantic_index(
        self,
        quantity_ids: torch.Tensor,
        axis_ids: torch.Tensor,
        location_ids: torch.Tensor,
        unit_class_ids: torch.Tensor,
    ) -> torch.Tensor:
        _, na, nl, nu = self._sizes
        return (
            ((quantity_ids.long() * na + axis_ids.long()) * nl + location_ids.long()) * nu
            + unit_class_ids.long()
        )

    def forward(
        self,
        features: torch.Tensor,
        token_mask: torch.Tensor,
        quantity_ids: torch.Tensor,
        axis_ids: torch.Tensor,
        location_ids: torch.Tensor,
        unit_class_ids: torch.Tensor,
        times_sec: torch.Tensor,
        channel_ids: torch.Tensor,
    ) -> torch.Tensor:
        if features.ndim != 3:
            raise ValueError("features must have shape [batch, tokens, feature_dim]")
        if token_mask.shape != features.shape[:2]:
            raise ValueError("token_mask shape must match features[:2]")

        semantic = self._semantic_index(quantity_ids, axis_ids, location_ids, unit_class_ids)
        time_key = torch.round(times_sec * self._TIME_QUANTIZATION).long()
        out = features.clone()

        # Batch-local grouping keeps memory linear in token count and avoids a
        # dense O(N^2) pairwise relation matrix.
        for b in range(features.shape[0]):
            visible_sensor = token_mask[b].bool() & (channel_ids[b].long() >= 0)
            idx = torch.where(visible_sensor)[0]
            if idx.numel() == 0:
                continue

            pair_keys = torch.stack((semantic[b, idx], time_key[b, idx]), dim=1)
            _, inverse = torch.unique(pair_keys, dim=0, return_inverse=True)
            n_groups = int(inverse.max().item()) + 1

            x = features[b, idx]
            sums = torch.zeros((n_groups, x.shape[-1]), dtype=x.dtype, device=x.device)
            sums.index_add_(0, inverse, x)
            counts = torch.bincount(inverse, minlength=n_groups).to(x).unsqueeze(1).clamp_min(1.0)
            group_mean = sums / counts
            out[b, idx] = x + (x - group_mean[inverse])

        return out
