from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import nn

from safegrip.universal.batching import UniversalSensorBatch
from safegrip.universal.queries import PhysicalQuery
from .metadata_embedding import PhysicallyTypedTokenEncoder
from .latent_backbone import SensorSetLatentBackbone
from .query_decoder import CompositionalQueryDecoder, QueryPrediction


@dataclass
class UniversalSafeGripOutput:
    point: torch.Tensor
    scale: torch.Tensor
    latent: torch.Tensor
    token_embeddings: torch.Tensor


class UniversalSafeGrip(nn.Module):
    """Dataset-independent heterogeneous sensor-set prediction model.

    Dataset identifiers are deliberately absent from the forward interface.
    Only sensor observations, physical metadata and physical target queries are
    accepted by the predictive model.
    """

    def __init__(
        self,
        patch_feature_dim: int,
        token_dim: int = 128,
        latent_dim: int = 192,
        latent_tokens: int = 32,
        cross_attention_heads: int = 4,
        latent_heads: int = 6,
        latent_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.token_encoder = PhysicallyTypedTokenEncoder(patch_feature_dim, token_dim)
        self.backbone = SensorSetLatentBackbone(
            token_dim=token_dim,
            latent_dim=latent_dim,
            latent_tokens=latent_tokens,
            cross_attention_heads=cross_attention_heads,
            latent_heads=latent_heads,
            latent_layers=latent_layers,
            dropout=dropout,
        )
        self.decoder = CompositionalQueryDecoder(latent_dim=latent_dim, heads=latent_heads)

    @staticmethod
    def query_tensor(queries: list[PhysicalQuery] | tuple[PhysicalQuery, ...], batch_size: int, device=None) -> torch.Tensor:
        if not queries:
            raise ValueError("at least one query is required")
        ids = torch.tensor([q.ids() for q in queries], dtype=torch.long, device=device)
        return ids.unsqueeze(0).expand(batch_size, -1, -1)

    def forward(self, batch: UniversalSensorBatch, queries: list[PhysicalQuery] | tuple[PhysicalQuery, ...]) -> UniversalSafeGripOutput:
        tokens = self.token_encoder(
            batch.features,
            batch.quantity_ids,
            batch.axis_ids,
            batch.location_ids,
            batch.unit_class_ids,
            batch.sample_rates_hz,
            batch.times_sec,
        )
        latent = self.backbone(tokens, batch.token_mask)
        qids = self.query_tensor(queries, batch.features.shape[0], device=batch.features.device)
        pred: QueryPrediction = self.decoder(latent, qids)
        return UniversalSafeGripOutput(pred.point, pred.scale, latent, tokens)
