from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import nn

from safegrip.ablations import UniversalAblationConfig
from safegrip.datasets import PAPER_DATASETS
from safegrip.universal.batching import UniversalSensorBatch
from safegrip.universal.queries import PhysicalQuery
from safegrip.universal.reference import PhysicalReferenceOperator
from safegrip.universal.innovation import SemanticInnovationNormalizer
from safegrip.universal.feature_normalization import SemanticFeatureNormalizer, SemanticRelationalContrast
from .metadata_embedding import PhysicallyTypedTokenEncoder
from .latent_backbone import SensorSetLatentBackbone, MeanPoolLatentBackbone
from .query_decoder import CompositionalQueryDecoder, QueryPrediction


@dataclass
class UniversalSafeGripOutput:
    point: torch.Tensor
    scale: torch.Tensor
    latent: torch.Tensor
    token_embeddings: torch.Tensor
    # Raw decoder output.  When a physical query is already represented by a
    # matching observed state channel, this is interpreted as an innovation
    # around the latest observed state rather than an absolute value.
    innovation: torch.Tensor
    physical_anchor: torch.Tensor
    anchor_mask: torch.Tensor
    innovation_center: torch.Tensor
    innovation_scale: torch.Tensor
    normalized_scale: torch.Tensor


class UniversalSafeGrip(nn.Module):
    """Shared heterogeneous sensor-set model with explicit research ablations."""

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
        ablation: UniversalAblationConfig | None = None,
        innovation_normalizer: SemanticInnovationNormalizer | None = None,
        feature_normalizer: SemanticFeatureNormalizer | None = None,
        relational_transform: SemanticRelationalContrast | None = None,
    ):
        super().__init__()
        self.ablation = ablation or UniversalAblationConfig()
        self.token_encoder = PhysicallyTypedTokenEncoder(
            patch_feature_dim, token_dim,
            use_physical_metadata=self.ablation.physical_metadata,
            use_unit_metadata=self.ablation.unit_metadata,
            use_sampling_rate_metadata=self.ablation.sampling_rate_metadata,
            use_time_metadata=self.ablation.time_metadata,
            use_channel_id_embedding=self.ablation.channel_id_embedding,
        )
        if self.ablation.aggregation == "latent_attention":
            self.backbone = SensorSetLatentBackbone(
                token_dim, latent_dim, latent_tokens, cross_attention_heads,
                latent_heads, latent_layers, dropout,
            )
        elif self.ablation.aggregation == "mean_pool":
            self.backbone = MeanPoolLatentBackbone(token_dim, latent_dim)
        else:
            raise ValueError(f"unknown aggregation mode: {self.ablation.aggregation}")
        self.decoder = CompositionalQueryDecoder(latent_dim=latent_dim, heads=latent_heads)
        # No trainable parameters are introduced by the innovation normalizer.
        # An identity normalizer preserves backwards compatibility for low-level
        # model tests; paper runners fit it on training data before optimization.
        self.innovation_normalizer = innovation_normalizer or SemanticInnovationNormalizer()
        self.feature_normalizer = feature_normalizer or SemanticFeatureNormalizer(patch_feature_dim)
        # RA-NPI relational evidence is a fixed, parameter-free transform.
        self.relational_transform = relational_transform or SemanticRelationalContrast()

        # The proposal is dataset-agnostic by construction.  Dataset identity is
        # available *only* as an explicit ablation; the default/full method does
        # not allocate dataset-specific parameters at all.
        if self.ablation.dataset_id_conditioning:
            self.dataset_to_id = {name: i for i, name in enumerate(PAPER_DATASETS)}
            self.dataset_embedding: nn.Embedding | None = nn.Embedding(len(self.dataset_to_id), token_dim)
        else:
            self.dataset_to_id = {}
            self.dataset_embedding = None

    @property
    def is_dataset_agnostic(self) -> bool:
        """True for the proposal used in the paper; False only for the ID-conditioning ablation."""
        return self.dataset_embedding is None

    @staticmethod
    def query_tensor(queries, batch_size: int, device=None) -> torch.Tensor:
        if not queries:
            raise ValueError("at least one query is required")
        ids = torch.tensor([q.ids() for q in queries], dtype=torch.long, device=device)
        return ids.unsqueeze(0).expand(batch_size, -1, -1)

    def _dataset_condition(self, tokens: torch.Tensor, domains: tuple[str, ...] | None) -> torch.Tensor:
        if not self.ablation.dataset_id_conditioning:
            return tokens
        if domains is None or len(domains) != tokens.shape[0]:
            raise ValueError("domains are required for dataset-ID conditioning ablation")
        if self.dataset_embedding is None:
            raise RuntimeError("dataset embedding exists only in the dataset-ID conditioning ablation")
        try:
            ids = torch.tensor([self.dataset_to_id[str(d).lower()] for d in domains], device=tokens.device)
        except KeyError as exc:
            raise ValueError(f"unknown dataset domain for conditioning: {exc.args[0]}") from exc
        return tokens + self.dataset_embedding(ids).unsqueeze(1)


    @staticmethod
    def _physical_query_anchor(
        batch: UniversalSensorBatch, queries
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Backward-compatible name for the semantic physical reference operator."""
        return PhysicalReferenceOperator()(batch, queries)

    def forward(self, batch: UniversalSensorBatch, queries, *, domains: tuple[str, ...] | None = None) -> UniversalSafeGripOutput:
        normalized_features = self.feature_normalizer(
            batch.features, batch.quantity_ids, batch.axis_ids, batch.location_ids, batch.unit_class_ids,
        )
        relational_features = self.relational_transform(
            normalized_features, batch.token_mask, batch.quantity_ids, batch.axis_ids,
            batch.location_ids, batch.unit_class_ids, batch.times_sec, batch.channel_ids,
        )
        tokens = self.token_encoder(
            relational_features, batch.quantity_ids, batch.axis_ids, batch.location_ids,
            batch.unit_class_ids, batch.sample_rates_hz, batch.times_sec, batch.channel_ids,
        )
        tokens = self._dataset_condition(tokens, domains)
        latent = self.backbone(tokens, batch.token_mask)
        qids = self.query_tensor(queries, batch.features.shape[0], device=batch.features.device)
        pred: QueryPrediction = self.decoder(latent, qids)
        physical_anchor, anchor_mask = self._physical_query_anchor(batch, queries)

        # Relationally Anchored NPI (RA-NPI): the decoder always predicts a
        # dimensionless O(1) innovation.  The physical reference uses one
        # training-only semantic gain anchored to the v3 identity coordinate;
        # residual offset/scale remain robust innovation statistics.  If sensor
        # dropout removes the reference, the same decoder uses the direct-target
        # coordinate rather than a dataset/task-specific branch.
        point, innovation_center, innovation_scale = self.innovation_normalizer.reconstruct(
            pred.point, qids, physical_anchor, anchor_mask
        )
        normalized_scale = pred.scale if self.ablation.learned_scale else torch.ones_like(pred.point)
        scale = normalized_scale * innovation_scale
        return UniversalSafeGripOutput(
            point, scale, latent, tokens, pred.point, physical_anchor, anchor_mask,
            innovation_center, innovation_scale, normalized_scale,
        )
