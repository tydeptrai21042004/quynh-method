from __future__ import annotations

"""Training-only semantic standardization for universal sensor tokens.

The transform is deliberately parameter-free.  Token features are standardized
using statistics indexed only by physical sensor semantics
(quantity, axis, location, unit class), never by dataset identity or channel
position.  The raw canonical values retained in ``UniversalSensorBatch`` remain
untouched, so the physical-reference operator continues to work in physical
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
        nq, na, nl, nu = self._sizes
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
