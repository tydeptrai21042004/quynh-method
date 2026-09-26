from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import nn
import torch.nn.functional as F

from safegrip.universal.ontology import OntologySizes


@dataclass
class QueryPrediction:
    point: torch.Tensor
    scale: torch.Tensor
    hidden: torch.Tensor


class CompositionalQueryDecoder(nn.Module):
    """Decode physical targets from a shared latent state using semantic queries."""

    def __init__(self, latent_dim: int = 192, heads: int = 6, scale_floor: float = 1e-4):
        super().__init__()
        if latent_dim % heads != 0:
            raise ValueError("latent_dim must be divisible by heads")
        sizes = OntologySizes()
        self.quantity = nn.Embedding(sizes.quantities, latent_dim)
        self.axis = nn.Embedding(sizes.axes, latent_dim)
        self.location = nn.Embedding(sizes.locations, latent_dim)
        self.target_type = nn.Embedding(sizes.target_types, latent_dim)
        self.query_norm = nn.LayerNorm(latent_dim)
        self.attention = nn.MultiheadAttention(latent_dim, heads, batch_first=True)
        self.post = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
        )
        self.point_head = nn.Linear(latent_dim, 1)
        self.scale_head = nn.Linear(latent_dim, 1)
        self.scale_floor = float(scale_floor)

    def forward(self, latent: torch.Tensor, query_ids: torch.Tensor) -> QueryPrediction:
        if query_ids.ndim != 3 or query_ids.shape[-1] != 4:
            raise ValueError("query_ids must be [B,Q,4]")
        q = (
            self.quantity(query_ids[..., 0])
            + self.axis(query_ids[..., 1])
            + self.location(query_ids[..., 2])
            + self.target_type(query_ids[..., 3])
        )
        q = self.query_norm(q)
        decoded, _ = self.attention(q, latent, latent, need_weights=False)
        h = self.post(decoded + q)
        point = self.point_head(h).squeeze(-1)
        scale = F.softplus(self.scale_head(h).squeeze(-1)) + self.scale_floor
        return QueryPrediction(point=point, scale=scale, hidden=h)
