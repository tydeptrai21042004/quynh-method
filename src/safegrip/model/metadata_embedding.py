from __future__ import annotations

import torch
from torch import nn

from safegrip.universal.ontology import OntologySizes


class PhysicallyTypedTokenEncoder(nn.Module):
    """Encode deterministic patch descriptors plus selectively enabled metadata."""

    def __init__(
        self,
        patch_feature_dim: int,
        token_dim: int = 128,
        hidden_dim: int | None = None,
        *,
        use_physical_metadata: bool = True,
        use_unit_metadata: bool = True,
        use_sampling_rate_metadata: bool = True,
        use_time_metadata: bool = True,
        use_channel_id_embedding: bool = False,
        max_channel_ids: int = 128,
    ):
        super().__init__()
        sizes = OntologySizes()
        hidden = int(hidden_dim or max(token_dim, patch_feature_dim))
        self.patch_encoder = nn.Sequential(
            nn.Linear(patch_feature_dim, hidden), nn.GELU(), nn.LayerNorm(hidden), nn.Linear(hidden, token_dim)
        )
        self.quantity = nn.Embedding(sizes.quantities, token_dim)
        self.axis = nn.Embedding(sizes.axes, token_dim)
        self.location = nn.Embedding(sizes.locations, token_dim)
        self.unit_class = nn.Embedding(sizes.unit_classes, token_dim)
        self.rate_encoder = nn.Sequential(nn.Linear(1, token_dim), nn.Tanh())
        self.time_encoder = nn.Sequential(nn.Linear(2, token_dim), nn.Tanh())
        self.channel_id = nn.Embedding(int(max_channel_ids) + 2, token_dim)
        self.norm = nn.LayerNorm(token_dim)
        self.use_physical_metadata = bool(use_physical_metadata)
        self.use_unit_metadata = bool(use_unit_metadata)
        self.use_sampling_rate_metadata = bool(use_sampling_rate_metadata)
        self.use_time_metadata = bool(use_time_metadata)
        self.use_channel_id_embedding = bool(use_channel_id_embedding)
        self.max_channel_ids = int(max_channel_ids)

    def forward(
        self,
        patch_features: torch.Tensor,
        quantity_ids: torch.Tensor,
        axis_ids: torch.Tensor,
        location_ids: torch.Tensor,
        unit_class_ids: torch.Tensor,
        sample_rates_hz: torch.Tensor,
        times_sec: torch.Tensor,
        channel_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.patch_encoder(patch_features)
        if self.use_physical_metadata:
            x = x + self.quantity(quantity_ids) + self.axis(axis_ids) + self.location(location_ids)
        if self.use_unit_metadata:
            x = x + self.unit_class(unit_class_ids)
        if self.use_sampling_rate_metadata:
            rate = torch.log1p(torch.clamp(sample_rates_hz, min=0.0)).unsqueeze(-1)
            x = x + self.rate_encoder(rate)
        if self.use_time_metadata:
            time_feat = torch.stack((times_sec, torch.log1p(torch.clamp(times_sec, min=0.0))), dim=-1)
            x = x + self.time_encoder(time_feat)
        if self.use_channel_id_embedding:
            if channel_ids is None:
                raise ValueError("channel_ids are required when channel-ID embedding is enabled")
            # Reserve 0 for negative/static IDs; real channel k maps to k+1.
            ids = torch.where(channel_ids < 0, torch.zeros_like(channel_ids), channel_ids + 1)
            ids = ids.clamp(max=self.max_channel_ids + 1)
            x = x + self.channel_id(ids)
        return self.norm(x)
