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
        # Stable, data-independent UQ initialization.  Starting at sigma≈0.135
        # prevents the early NLL from being dominated by an arbitrary sigma≈1
        # while leaving both mean and uncertainty fully learnable.
        nn.init.zeros_(self.log_sigma.weight)
        nn.init.constant_(self.log_sigma.bias, -2.0)

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
        nn.init.zeros_(self.log_sigma.weight)
        nn.init.constant_(self.log_sigma.bias, -2.0)

    def forward(self, x):
        z = self.net(x[:, -1, :])
        return self.mu(z).squeeze(-1), self.log_sigma(z).squeeze(-1).clamp(-6, 2)


class Todorovic2022CNN(nn.Module):
    """Architecture-faithful adaptation of Todorovic et al. (2022).

    The source architecture uses a 100-sample temporal input, three Conv1D +
    MaxPool stages (128/128/256 channels) and a 400-unit dense layer. The
    original paper predicts longitudinal and lateral friction potentials; the
    common LiRA benchmark changes only the input channel count and final output
    to the shared scalar road-friction reference.
    """
    def __init__(self, d: int, sequence_length: int = 100, dropout: float = 0.0,
                 debug_scale: bool = False):
        super().__init__()
        c1, c2, c3, dense = ((32, 32, 64, 64) if debug_scale else (128, 128, 256, 400))
        # padding='same' preserves the source temporal length before each pool;
        # for L=100: 100 -> 50 -> 25 -> 12, hence 12*256=3072 features.
        self.features = nn.Sequential(
            nn.Conv1d(d, c1, kernel_size=14, padding="same"), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(c1, c2, kernel_size=10, padding="same"), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(c2, c3, kernel_size=10, padding="same"), nn.ReLU(), nn.MaxPool1d(2),
        )
        if int(sequence_length) < 8:
            raise ValueError("Todorovic2022CNN requires sequence_length >= 8")
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            # LazyLinear preserves the source Flatten->Dense(400) design while
            # allowing quick tests to use a shorter synthetic input. In paper
            # mode L=100 materializes exactly 12*256=3072 input features.
            nn.LazyLinear(dense), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dense, 1),
        )

    def forward(self, x):
        return self.head(self.features(x.transpose(1, 2))).squeeze(-1)


def _init_lampe(module: nn.Module) -> None:
    """Lampe et al.: orthogonal recurrent weights, Glorot input/dense weights."""
    if isinstance(module, (nn.LSTM, nn.GRU)):
        for name, param in module.named_parameters():
            if "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
    elif isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class Lampe2023RNN(nn.Module):
    """Architecture-faithful LSTM/GRU adaptation from Lampe et al. (2023)."""
    def __init__(self, d, kind="gru", hidden=256, dropout=0.0):
        super().__init__()
        cls = nn.LSTM if kind == "lstm" else nn.GRU
        self.kind = kind
        self.rnn = cls(d, hidden, num_layers=2, batch_first=True, dropout=dropout)
        if kind == "lstm":
            self.head = nn.Sequential(nn.Linear(hidden, 256), nn.Tanh(), nn.Linear(256, 1))
        else:
            self.head = nn.Linear(hidden, 1)
        self.apply(_init_lampe)

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
    """Methodology-level common-sensor Transformer adaptation (Schäfke et al., 2023)."""
    def __init__(self, d, hidden=64, dropout=.1, layers=2, heads=4, ff_mult=2):
        super().__init__()
        if hidden % heads:
            raise ValueError("Transformer hidden size must be divisible by number of heads")
        self.inp = nn.Linear(d, hidden)
        self.pos = PositionalEncoding(hidden)
        layer = nn.TransformerEncoderLayer(
            hidden, nhead=heads, dim_feedforward=hidden * ff_mult, dropout=dropout,
            batch_first=True, activation="gelu"
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=layers)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        z = self.enc(self.pos(self.inp(x)))
        return self.head(z[:, -1]).squeeze(-1)


class SpatioTemporalCNN(nn.Module):
    """Feature extractor for the adapted Chen et al. (2025) SV-DKL comparator.

    The public paper includes a data-category-selection stage that cannot be
    reproduced from LiRA with the same electric-wheel-vehicle state variables.
    This class therefore implements only the spatio-temporal feature + SV-DKL
    portion and is explicitly labelled as an adapted comparator in provenance.
    """
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


def make_literature_baseline(name: str, d: int, *, sequence_length: int = 64,
                             debug_scale: bool = False, dropout: float = .1,
                             hidden: int | None = None, layers: int = 2,
                             heads: int = 4, ff_mult: int = 2):
    """Build a literature comparator; paper mode never uses debug_scale."""
    if name == "todorovic2022_cnn":
        return Todorovic2022CNN(d, sequence_length=sequence_length, dropout=dropout, debug_scale=debug_scale)
    if name == "lampe2023_lstm":
        return Lampe2023RNN(d, "lstm", hidden=32 if debug_scale else 256, dropout=dropout)
    if name == "lampe2023_gru":
        return Lampe2023RNN(d, "gru", hidden=32 if debug_scale else 256, dropout=dropout)
    if name == "schaefke2023_transformer":
        h = 32 if debug_scale else int(hidden or 64)
        return Schaefke2023Transformer(d, hidden=h, dropout=dropout,
                                       layers=1 if debug_scale else int(layers), heads=int(heads), ff_mult=int(ff_mult))
    raise ValueError(name)
