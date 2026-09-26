from __future__ import annotations

import torch
from torch import nn


class SensorSetLatentBackbone(nn.Module):
    """Permutation-invariant sensor-token fusion through a latent bank.

    Input token order has no architectural significance. Physical time is
    represented inside each token rather than by an index-based positional
    encoding, so reordering tokens leaves the set function unchanged up to
    floating-point noise.
    """

    def __init__(
        self,
        token_dim: int = 128,
        latent_dim: int = 192,
        latent_tokens: int = 32,
        cross_attention_heads: int = 4,
        latent_heads: int = 6,
        latent_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        if latent_dim % cross_attention_heads != 0:
            raise ValueError("latent_dim must be divisible by cross_attention_heads")
        if latent_dim % latent_heads != 0:
            raise ValueError("latent_dim must be divisible by latent_heads")
        self.token_projection = nn.Linear(token_dim, latent_dim)
        self.latents = nn.Parameter(torch.randn(latent_tokens, latent_dim) * 0.02)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=latent_dim,
            num_heads=cross_attention_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_norm = nn.LayerNorm(latent_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=latent_heads,
            dim_feedforward=latent_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=latent_layers)
        self.out_norm = nn.LayerNorm(latent_dim)

    def forward(self, tokens: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3 or token_mask.ndim != 2:
            raise ValueError("tokens must be [B,N,D] and token_mask [B,N]")
        if tokens.shape[:2] != token_mask.shape:
            raise ValueError("token_mask must match tokens first two dimensions")
        if torch.any(token_mask.sum(dim=1) == 0):
            raise ValueError("each sample must contain at least one valid token")
        kv = self.token_projection(tokens)
        b = tokens.shape[0]
        latent = self.latents.unsqueeze(0).expand(b, -1, -1)
        update, _ = self.cross_attention(
            latent,
            kv,
            kv,
            key_padding_mask=~token_mask,
            need_weights=False,
        )
        latent = self.cross_norm(latent + update)
        latent = self.encoder(latent)
        return self.out_norm(latent)
