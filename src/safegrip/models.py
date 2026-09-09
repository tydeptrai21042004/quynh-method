from __future__ import annotations
import math
import torch
from torch import nn


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


class SafeGripV3Net(nn.Module):
    """SafeGrip-v3: prior + excitation-controlled dynamic evidence update.

    The proposal intentionally separates two roles that were conflated in v2:

    * a *slow prior* estimates the persistent friction level from the complete
      context window; and
    * a *short evidence branch* predicts a bounded correction supported by the
      most recent dynamics.

    Dynamic evidence is admitted through a monotone reliability function

        r(E) = sigmoid(softplus(a) * (E - sigmoid(tau)))

    so increasing the label-free excitation score can never reduce the amount
    of dynamic evidence used.  The update is performed in identified-set logit
    coordinates:

        q = q_prior + r(E) * delta
        mu = lower + (upper - lower) * sigmoid(q)

    which preserves ``lower <= mu <= upper`` by construction.  ``use_gate=False``
    is the controlled fixed-reliability ablation and ``use_temporal=False``
    removes the dynamic-evidence branch entirely.
    """

    def __init__(self, d: int, excitation_index: int | None, hidden: int = 64,
                 gru_hidden: int = 32, dropout: float = 0.1,
                 evidence_window: int = 8, delta_scale: float = 2.0,
                 gate_init_slope: float = 6.0, gate_init_threshold: float = 0.25,
                 use_temporal: bool = True, use_gate: bool = True,
                 use_bound: bool = True, endpoint_only: bool = False):
        super().__init__()
        self.d = int(d)
        self.excitation_index = excitation_index
        self.hidden = int(hidden)
        self.use_temporal = bool(use_temporal)
        self.use_gate = bool(use_gate) and self.use_temporal
        self.use_bound = bool(use_bound)
        self.endpoint_only = bool(endpoint_only)
        self.evidence_window = max(2, int(evidence_window))
        self.delta_scale = max(1e-4, float(delta_scale))

        # Persistent/slow prior: summary statistics plus a full-window GRU.
        self.prior_static = nn.Sequential(
            nn.Linear((3 * d) if self.use_temporal else d, hidden), nn.SiLU(), nn.LayerNorm(hidden),
            nn.Dropout(dropout), nn.Linear(hidden, hidden), nn.SiLU(),
        )
        if self.use_temporal:
            self.prior_gru = nn.GRU(d, gru_hidden, num_layers=1, batch_first=True)
            self.prior_temporal_proj = nn.Sequential(nn.Linear(gru_hidden, hidden), nn.SiLU())
            self.prior_fuse = nn.Sequential(
                nn.Linear(2 * hidden, hidden), nn.SiLU(), nn.LayerNorm(hidden)
            )
        else:
            self.prior_gru = None
            self.prior_temporal_proj = None
            self.prior_fuse = None

        inner = max(16, hidden // 2)
        self.prior_head = nn.Sequential(
            nn.Linear(hidden, inner), nn.SiLU(), nn.Dropout(dropout), nn.Linear(inner, 1)
        )

        # Short-window branch predicts *evidence/correction*, not absolute mu.
        if self.use_temporal:
            self.evidence_static = nn.Sequential(
                nn.Linear(3 * d, hidden), nn.SiLU(), nn.LayerNorm(hidden),
                nn.Dropout(dropout), nn.Linear(hidden, hidden), nn.SiLU(),
            )
            self.evidence_gru = nn.GRU(d, gru_hidden, num_layers=1, batch_first=True)
            self.evidence_temporal_proj = nn.Sequential(nn.Linear(gru_hidden, hidden), nn.SiLU())
            self.evidence_fuse = nn.Sequential(
                nn.Linear(2 * hidden, hidden), nn.SiLU(), nn.LayerNorm(hidden)
            )
            self.evidence_head = nn.Sequential(
                nn.Linear(hidden, inner), nn.SiLU(), nn.Dropout(dropout), nn.Linear(inner, 1)
            )
        else:
            self.evidence_static = None
            self.evidence_gru = None
            self.evidence_temporal_proj = None
            self.evidence_fuse = None
            self.evidence_head = None

        # Monotone reliability parameters.  softplus(slope_raw)>0 guarantees
        # dr/dE >= 0 for every parameter value during training.
        init_slope = max(float(gate_init_slope), 1e-3)
        init_thr = min(max(float(gate_init_threshold), 1e-4), 1.0 - 1e-4)
        inv_sp = math.log(math.expm1(init_slope)) if init_slope < 20 else init_slope
        self.reliability_slope_raw = nn.Parameter(torch.tensor(inv_sp, dtype=torch.float32))
        self.reliability_threshold_raw = nn.Parameter(
            torch.tensor(math.log(init_thr / (1.0 - init_thr)), dtype=torch.float32)
        )

        # Fixed-reliability ablations use this value rather than a learned gate.
        self.fixed_reliability = 0.5

        # UQ representation intentionally contains both slow and fast evidence.
        self.uq_fuse = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.SiLU(), nn.LayerNorm(hidden)
        )
        self.scale_head: ResidualScaleHead | None = None
        self.conformal_q: float | None = None
        self.conformal_block_size: int | None = None
        self.excitation_beta: float = 1.0

    def _stats(self, x: torch.Tensor) -> torch.Tensor:
        # Endpoint-only is a genuine static ablation: no window mean/std are
        # available to the estimator. Temporal engineered channels are also
        # removed by the matching feature bundle in benchmark.py.
        if self.endpoint_only:
            return x[:, -1, :]
        return torch.cat([
            x[:, -1, :],
            x.mean(dim=1),
            x.std(dim=1, unbiased=False),
        ], dim=-1)

    def reliability(self, excitation: torch.Tensor) -> torch.Tensor:
        e = torch.clamp(excitation.reshape(-1), 0.0, 1.0)
        if not self.use_temporal:
            return torch.zeros_like(e)
        if not self.use_gate:
            return torch.full_like(e, float(self.fixed_reliability))
        slope = nn.functional.softplus(self.reliability_slope_raw) + 1e-4
        threshold = torch.sigmoid(self.reliability_threshold_raw)
        return torch.sigmoid(slope * (e - threshold))

    def _excitation_from_input(self, x: torch.Tensor, excitation: torch.Tensor | None) -> torch.Tensor:
        if excitation is not None:
            return torch.clamp(excitation.reshape(-1).to(dtype=x.dtype, device=x.device), 0.0, 1.0)
        if self.excitation_index is None:
            return torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)
        # make_bundle preserves sg_excitation_score unscaled, so this fallback
        # remains physically interpretable in [0, 1].
        return torch.clamp(x[:, -1, self.excitation_index], 0.0, 1.0)

    def encode(self, x: torch.Tensor, excitation: torch.Tensor | None = None):
        # The non-temporal ablation is genuinely endpoint-only: it cannot use
        # window means/stds and therefore contains no hidden temporal summary.
        prior_input = self._stats(x) if self.use_temporal else x[:, -1, :]
        prior_static = self.prior_static(prior_input)
        if self.use_temporal:
            y_long, _ = self.prior_gru(x)
            prior_temporal = self.prior_temporal_proj(y_long[:, -1])
            prior_h = self.prior_fuse(torch.cat([prior_static, prior_temporal], dim=-1))
        else:
            prior_h = prior_static

        q_prior = self.prior_head(prior_h).squeeze(-1)
        e = self._excitation_from_input(x, excitation)

        if self.use_temporal:
            short = x[:, -min(self.evidence_window, x.shape[1]):, :]
            ev_static = self.evidence_static(self._stats(short))
            y_short, _ = self.evidence_gru(short)
            ev_temporal = self.evidence_temporal_proj(y_short[:, -1])
            evidence_h = self.evidence_fuse(torch.cat([ev_static, ev_temporal], dim=-1))
            # Bounded correction prevents a poorly observed short maneuver from
            # completely overwriting the persistent prior in one update.
            delta = self.delta_scale * torch.tanh(self.evidence_head(evidence_h).squeeze(-1))
            r = self.reliability(e)
        else:
            evidence_h = torch.zeros_like(prior_h)
            delta = torch.zeros_like(q_prior)
            r = torch.zeros_like(q_prior)

        q = q_prior + r * delta
        uq_h = self.uq_fuse(torch.cat([prior_h, evidence_h], dim=-1))
        return uq_h, r, q_prior, delta, q

    def forward_details(self, x: torch.Tensor, lower: torch.Tensor | None = None,
                        upper: float | torch.Tensor = 1.3,
                        excitation: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        h, reliability, q_prior, delta, q = self.encode(x, excitation=excitation)
        upper_t = torch.as_tensor(upper, dtype=q.dtype, device=q.device)
        if self.use_bound:
            if lower is None:
                raise ValueError("SafeGripV3Net requires lower when use_bound=True")
            span = torch.clamp(upper_t - lower, min=1e-6)
            prior = lower + span * torch.sigmoid(q_prior)
            pred = lower + span * torch.sigmoid(q)
        else:
            prior = upper_t * torch.sigmoid(q_prior)
            pred = upper_t * torch.sigmoid(q)
        return {
            "prediction": pred,
            "prior_prediction": prior,
            "latent": q,
            "prior_latent": q_prior,
            "evidence_delta": delta,
            "reliability": reliability,
            "features": h,
        }

    def forward(self, x: torch.Tensor, lower: torch.Tensor | None = None,
                upper: float | torch.Tensor = 1.3,
                excitation: torch.Tensor | None = None):
        d = self.forward_details(x, lower=lower, upper=upper, excitation=excitation)
        return d["prediction"], d["latent"], d["reliability"]


# Backward-compatible import alias for notebooks built against v0.6.x.  The
# implementation is v3; new code and documentation should use SafeGripV3Net.
SafeGripV2Net = SafeGripV3Net


class SafeGripBackboneNet(nn.Module):
    """Data-only control with the proposal's temporal capacity but no physics/gate.

    This network deliberately has no sample-specific lower bound, no explicit
    excitation reliability mechanism, and no proposal-specific relative/ranking
    regularizers.  It can be fed raw sensors (``safegrip_backbone_raw``) or the
    same label-free engineered representation (``safegrip_features_only``).
    """
    def __init__(self, d: int, hidden: int = 64, gru_hidden: int = 32, dropout: float = 0.1):
        super().__init__()
        self.hidden = int(hidden)
        self.static = nn.Sequential(
            nn.Linear(3 * d, hidden), nn.SiLU(), nn.LayerNorm(hidden),
            nn.Dropout(dropout), nn.Linear(hidden, hidden), nn.SiLU(),
        )
        self.gru = nn.GRU(d, gru_hidden, num_layers=1, batch_first=True)
        self.temporal = nn.Sequential(nn.Linear(gru_hidden, hidden), nn.SiLU())
        self.fuse = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.SiLU(), nn.LayerNorm(hidden))
        inner=max(16, hidden // 2)
        self.head = nn.Sequential(nn.Linear(hidden, inner), nn.SiLU(), nn.Dropout(dropout), nn.Linear(inner, 1))
        self.scale_head: ResidualScaleHead | None = None
        self.conformal_q: float | None = None
        self.conformal_block_size: int | None = None
        self.excitation_beta: float = 0.0

    @staticmethod
    def _stats(x: torch.Tensor) -> torch.Tensor:
        return torch.cat([x[:, -1, :], x.mean(dim=1), x.std(dim=1, unbiased=False)], dim=-1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        s=self.static(self._stats(x))
        y,_=self.gru(x)
        t=self.temporal(y[:, -1])
        return self.fuse(torch.cat([s,t],dim=-1))

    def forward_details(self, x: torch.Tensor, lower: torch.Tensor | None = None,
                        upper: float | torch.Tensor = 1.3,
                        excitation: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        h=self.encode(x)
        q=self.head(h).squeeze(-1)
        upper_t=torch.as_tensor(upper,dtype=q.dtype,device=q.device)
        pred=upper_t*torch.sigmoid(q)
        zero=torch.zeros_like(pred)
        return {
            "prediction":pred, "prior_prediction":pred, "latent":q,
            "prior_latent":q, "evidence_delta":zero, "reliability":zero,
            "features":h,
        }

    def forward(self, x: torch.Tensor, lower: torch.Tensor | None = None,
                upper: float | torch.Tensor = 1.3,
                excitation: torch.Tensor | None = None):
        d=self.forward_details(x,lower=lower,upper=upper,excitation=excitation)
        return d["prediction"],d["latent"],d["reliability"]


class ResidualScaleHead(nn.Module):
    """Positive residual-scale head fitted after point-model selection.

    ``initial_scale`` initializes the last bias near the validation residual
    magnitude.  This avoids the old softplus(0) ~= 0.69 initialization that was
    orders of magnitude larger than LiRA errors and forced conformal calibration
    to compensate for a poorly scaled head.
    """

    def __init__(self, hidden: int, floor: float = 1e-2, initial_scale: float | None = None):
        super().__init__()
        inner = max(8, hidden // 2)
        self.fc1 = nn.Linear(hidden, inner)
        self.fc2 = nn.Linear(inner, 1)
        self.floor = float(floor)
        if initial_scale is not None:
            target = max(float(initial_scale) - self.floor, 1e-6)
            raw = math.log(math.expm1(target)) if target < 20 else target
            nn.init.zeros_(self.fc2.weight)
            nn.init.constant_(self.fc2.bias, raw)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        z = nn.functional.silu(self.fc1(h))
        return nn.functional.softplus(self.fc2(z).squeeze(-1)) + self.floor
