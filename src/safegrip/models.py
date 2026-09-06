from __future__ import annotations
import math
import torch
from torch import nn


class TCNBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3, dilation: int = 1, dropout: float = 0.1):
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.c1 = nn.Conv1d(channels, channels, kernel_size, dilation=dilation, padding=pad)
        self.c2 = nn.Conv1d(channels, channels, kernel_size, dilation=dilation, padding=pad)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)

    @staticmethod
    def _causal_crop(y, n):
        return y[..., :n]

    def forward(self, x):
        n = x.shape[-1]
        y = self._causal_crop(self.c1(x), n)
        y = self.drop(self.act(y))
        y = self._causal_crop(self.c2(y), n)
        return self.act(x + self.drop(y))


class TCNEncoder(nn.Module):
    def __init__(self, d: int, hidden: int = 64, dropout: float = 0.1,
                 blocks: int = 4, kernel_size: int = 3):
        super().__init__()
        self.inp = nn.Conv1d(d, hidden, 1)
        self.blocks = nn.Sequential(*[
            TCNBlock(hidden, kernel_size=kernel_size, dilation=2 ** i, dropout=dropout)
            for i in range(blocks)
        ])

    def forward(self, x):
        z = self.blocks(self.inp(x.transpose(1, 2)))
        return z[:, :, -1]


class DeterministicTCN(nn.Module):
    def __init__(self, d, hidden=64, dropout=.1, blocks=4, kernel_size=3):
        super().__init__()
        self.encoder = TCNEncoder(d, hidden, dropout, blocks, kernel_size)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        return self.head(self.encoder(x)).squeeze(-1)


class SafeGripNet(nn.Module):
    """Proposal backbone: compact TCN with heteroscedastic Gaussian head."""
    def __init__(self, d, hidden=64, dropout=.1, blocks=4, kernel_size=3):
        super().__init__()
        self.encoder = TCNEncoder(d, hidden, dropout, blocks, kernel_size)
        self.mu = nn.Linear(hidden, 1)
        self.log_sigma = nn.Linear(hidden, 1)

    def forward(self, x):
        z = self.encoder(x)
        return self.mu(z).squeeze(-1), self.log_sigma(z).squeeze(-1).clamp(-6, 2)


class SafeGripNoTemporal(nn.Module):
    """Ablation: removes temporal encoder while retaining the uncertainty head."""
    def __init__(self, d, hidden=64, dropout=.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
        )
        self.mu = nn.Linear(hidden, 1)
        self.log_sigma = nn.Linear(hidden, 1)

    def forward(self, x):
        z = self.net(x[:, -1, :])
        return self.mu(z).squeeze(-1), self.log_sigma(z).squeeze(-1).clamp(-6, 2)


class Todorovic2022CNN(nn.Module):
    """Methodology-level temporal CNN baseline based on Todorovic et al. (2022).

    The original paper uses a CNN over longitudinal/lateral vehicle signals. The
    exact proprietary channel set is not present in every open benchmark; this
    implementation deliberately uses the same common production-sensor subset as
    every other direct baseline.
    """
    def __init__(self, d, hidden=64, dropout=.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(d, hidden, 5, padding=2), nn.ReLU(), nn.BatchNorm1d(hidden),
            nn.Conv1d(hidden, hidden, 3, padding=1), nn.ReLU(), nn.Dropout(dropout),
            nn.Conv1d(hidden, hidden * 2, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(hidden * 2, 1)

    def forward(self, x):
        z = self.net(x.transpose(1, 2)).squeeze(-1)
        return self.head(z).squeeze(-1)


class Lampe2023RNN(nn.Module):
    """Architecture-level LSTM/GRU baseline from Lampe et al. (2023)."""
    def __init__(self, d, kind="gru", hidden=256, dropout=.1):
        super().__init__()
        cls = nn.LSTM if kind == "lstm" else nn.GRU
        self.kind = kind
        self.rnn = cls(d, hidden, num_layers=2, batch_first=True, dropout=dropout)
        if kind == "lstm":
            self.head = nn.Sequential(nn.Linear(hidden, 256), nn.Tanh(), nn.Linear(256, 1))
        else:
            self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        y, _ = self.rnn(x)
        return self.head(y[:, -1]).squeeze(-1)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[:pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class Schaefke2023Transformer(nn.Module):
    """Methodology-level onboard-sensor Transformer baseline (Schäfke et al., 2023)."""
    def __init__(self, d, hidden=64, dropout=.1, layers=2, heads=4):
        super().__init__()
        if hidden % heads:
            heads = 1
        self.inp = nn.Linear(d, hidden)
        self.pos = PositionalEncoding(hidden)
        layer = nn.TransformerEncoderLayer(
            hidden, nhead=heads, dim_feedforward=hidden * 2, dropout=dropout,
            batch_first=True, activation="gelu"
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=layers)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        z = self.enc(self.pos(self.inp(x)))
        return self.head(z[:, -1]).squeeze(-1)


class SpatioTemporalCNN(nn.Module):
    """Feature extractor used by the Chen et al. (2025) SV-DKL baseline."""
    def __init__(self, d, hidden=64, feature_dim=16, dropout=.1):
        super().__init__()
        self.temporal = nn.Sequential(
            nn.Conv1d(d, hidden, 5, padding=2), nn.ReLU(),
            nn.Conv1d(hidden, hidden, 3, padding=1), nn.ReLU(),
            nn.Dropout(dropout), nn.AdaptiveAvgPool1d(1)
        )
        self.instant = nn.Sequential(nn.Linear(d, hidden), nn.ReLU())
        self.fuse = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.ReLU(), nn.Linear(hidden, feature_dim))

    def forward(self, x):
        a = self.temporal(x.transpose(1, 2)).squeeze(-1)
        b = self.instant(x[:, -1])
        return self.fuse(torch.cat([a, b], dim=-1))


def make_literature_baseline(name: str, d: int, *, debug_scale: bool = False, dropout: float = .1):
    """Build a cited baseline. debug_scale is never used by paper mode."""
    if name == "todorovic2022_cnn":
        return Todorovic2022CNN(d, hidden=32 if debug_scale else 64, dropout=dropout)
    if name == "lampe2023_lstm":
        return Lampe2023RNN(d, "lstm", hidden=32 if debug_scale else 256, dropout=dropout)
    if name == "lampe2023_gru":
        return Lampe2023RNN(d, "gru", hidden=32 if debug_scale else 256, dropout=dropout)
    if name == "schaefke2023_transformer":
        return Schaefke2023Transformer(d, hidden=32 if debug_scale else 64, dropout=dropout,
                                       layers=1 if debug_scale else 2, heads=4)
    raise ValueError(name)
