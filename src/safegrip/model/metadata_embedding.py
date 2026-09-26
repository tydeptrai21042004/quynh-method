from __future__ import annotations

import torch
from torch import nn

from safegrip.universal.ontology import OntologySizes


class PhysicallyTypedTokenEncoder(nn.Module):
    """Encode deterministic patch descriptors plus physical metadata."""

    def __init__(self, patch_feature_dim: int, token_dim: int = 128, hidden_dim: int | None = None):
        super().__init__()
        sizes = OntologySizes()
        hidden = int(hidden_dim or max(token_dim, patch_feature_dim))
        self.patch_encoder = nn.Sequential(
            nn.Linear(patch_feature_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, token_dim),
        )
        self.quantity = nn.Embedding(sizes.quantities, token_dim)
        self.axis = nn.Embedding(sizes.axes, token_dim)
        self.location = nn.Embedding(sizes.locations, token_dim)
        self.unit_class = nn.Embedding(sizes.unit_classes, token_dim)
        self.rate_encoder = nn.Sequential(nn.Linear(1, token_dim), nn.Tanh())
        self.time_encoder = nn.Sequential(nn.Linear(2, token_dim), nn.Tanh())
        self.norm = nn.LayerNorm(token_dim)

    def forward(
        self,
        patch_features: torch.Tensor,
        quantity_ids: torch.Tensor,
        axis_ids: torch.Tensor,
        location_ids: torch.Tensor,
        unit_class_ids: torch.Tensor,
        sample_rates_hz: torch.Tensor,
        times_sec: torch.Tensor,
    ) -> torch.Tensor:
        x = self.patch_encoder(patch_features)
        rate = torch.log1p(torch.clamp(sample_rates_hz, min=0.0)).unsqueeze(-1)
        time_feat = torch.stack((times_sec, torch.log1p(torch.clamp(times_sec, min=0.0))), dim=-1)
        x = (
            x
            + self.quantity(quantity_ids)
            + self.axis(axis_ids)
            + self.location(location_ids)
            + self.unit_class(unit_class_ids)
            + self.rate_encoder(rate)
            + self.time_encoder(time_feat)
        )
        return self.norm(x)
