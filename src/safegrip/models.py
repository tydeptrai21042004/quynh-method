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


class SafeGripCINet(nn.Module):
    """SafeGrip-CI: counterfactual-identifiability friction state estimator.

    The proposal has one central principle: a learned friction innovation is
    allowed to update the persistent friction state only when the current
    vehicle response can *counterfactually distinguish* nearby friction
    hypotheses and when the candidate state explains the observed dynamics
    better than the prior state.

    v1.0 uses a *multi-scale counterfactual trust region*.  The learned
    candidate innovation is no longer multiplied by a symmetric sigmoid that
    defaults to 0.5 when the dynamics residuals are nearly equal.  Instead, a
    candidate keeps the identifiability authority unless there is affirmative
    counterfactual evidence that it makes the observed dynamics worse.  Local
    sensitivity is estimated at several friction displacements and discounted
    when those derivative estimates are mutually inconsistent.

    For prior friction ``mu_prior`` and candidate innovation ``nu``:

        q_cand = q_prior + nu
        I      = robust multi-scale sensitivity of G(context, mu) to mu
        C      = local sensitivity consistency / linearity score
        z      = (R_prior - R_cand) / (R_prior + R_cand + eps)
        A      = learned-vs-inverse-dynamics correction agreement
        V      = asymmetric residual veto times a disagreement veto
        K      = V * C * I / (I + lambda)
        q      = q_prior + K * nu
        mu     = L + (U-L) * sigmoid(q)

    ``G`` is a friction-conditioned dynamics model trained jointly with the
    estimator.  It predicts a selected subset of standardized vehicle-dynamic
    channels at the current endpoint from the causal prefix of the window and a
    friction hypothesis.  ``I`` is therefore a local counterfactual
    identifiability certificate rather than a hand-crafted excitation proxy.
    The same local dynamics model also yields a damped Gauss--Newton friction
    correction.  That correction is exposed as a diagnostic and can be used as
    an auxiliary agreement target during training.  In v1.0, strong directional
    disagreement also supplies a mild asymmetric veto at inference.  Neither
    path consumes test friction labels.

    A previous predicted friction can be passed through ``prior_mu``.  This is
    how benchmark inference maintains a real persistent state inside a segment.
    At a segment boundary the estimator falls back to its learned context prior.
    """

    def __init__(self, d: int, excitation_index: int | None = None,
                 dynamics_indices: list[int] | tuple[int, ...] | None = None,
                 hidden: int = 64, gru_hidden: int = 32, dropout: float = 0.1,
                 evidence_window: int = 8, delta_scale: float = 1.5,
                 counterfactual_delta: float = 0.08,
                 identifiability_lambda: float = 0.20,
                 acceptance_temperature: float = 12.0,
                 acceptance_margin: float = 0.0,
                 acceptance_tolerance: float = 0.10,
                 acceptance_strength: float = 0.35,
                 counterfactual_scale_span: float = 2.0,
                 use_multiscale_counterfactual: bool = True,
                 use_linearity_consistency: bool = True,
                 linearity_penalty: float = 0.50,
                 inverse_dynamics_ridge: float = 0.001,
                 inverse_dynamics_max_step: float = 0.12,
                 use_agreement_veto: bool = True,
                 agreement_temperature: float = 10.0,
                 agreement_threshold: float = 0.35,
                 agreement_strength: float = 0.15,
                 gate_init_slope: float | None = None, gate_init_threshold: float | None = None,
                 use_temporal: bool = True, use_gate: bool = True,
                 use_bound: bool = True, endpoint_only: bool = False,
                 use_identifiability: bool = True,
                 use_acceptance: bool = True,
                 use_persistent_state: bool = True,
                 use_excitation_proxy: bool = False,
                 use_innovation: bool = True, state_persistence: float = 0.85):
        super().__init__()
        self.d = int(d)
        self.excitation_index = excitation_index
        self.hidden = int(hidden)
        self.use_temporal = bool(use_temporal)
        self.use_bound = bool(use_bound)
        self.endpoint_only = bool(endpoint_only)
        self.use_identifiability = bool(use_identifiability) and self.use_temporal
        self.use_acceptance = bool(use_acceptance) and self.use_temporal
        self.use_persistent_state = bool(use_persistent_state)
        self.use_excitation_proxy = bool(use_excitation_proxy)
        self.use_innovation = bool(use_innovation) and self.use_temporal
        self.state_persistence = min(max(float(state_persistence),0.0),1.0)
        # ``use_gate`` is kept for API compatibility; in CI it means use the
        # counterfactual authority mechanism rather than an unconditional update.
        self.use_gate = bool(use_gate)
        self.evidence_window = max(2, int(evidence_window))
        self.delta_scale = max(1e-4, float(delta_scale))
        self.counterfactual_delta = max(1e-4, float(counterfactual_delta))
        self.identifiability_lambda = max(1e-6, float(identifiability_lambda))
        self.acceptance_temperature = max(1e-3, float(acceptance_temperature))
        self.acceptance_margin = float(acceptance_margin)
        self.acceptance_tolerance = max(0.0, float(acceptance_tolerance))
        self.acceptance_strength = min(max(float(acceptance_strength), 0.0), 1.0)
        self.counterfactual_scale_span = max(1.0, float(counterfactual_scale_span))
        self.use_multiscale_counterfactual = bool(use_multiscale_counterfactual)
        self.use_linearity_consistency = bool(use_linearity_consistency)
        self.linearity_penalty = max(0.0, float(linearity_penalty))
        self.inverse_dynamics_ridge = max(1e-8, float(inverse_dynamics_ridge))
        self.inverse_dynamics_max_step = max(1e-4, float(inverse_dynamics_max_step))
        self.use_agreement_veto = bool(use_agreement_veto)
        self.agreement_temperature = max(1e-3, float(agreement_temperature))
        self.agreement_threshold = min(max(float(agreement_threshold), 0.0), 1.0)
        self.agreement_strength = min(max(float(agreement_strength), 0.0), 1.0)

        if dynamics_indices is None:
            dynamics_indices = list(range(min(4, self.d)))
        dynamics_indices = [int(i) for i in dynamics_indices if 0 <= int(i) < self.d]
        if not dynamics_indices:
            dynamics_indices = [0]
        self.dynamics_indices = tuple(dict.fromkeys(dynamics_indices))
        dyn_out = len(self.dynamics_indices)

        # Context prior.  It is used only when a segment has no persistent state.
        if self.use_temporal:
            self.prior_static = nn.Sequential(
                nn.Linear(3 * d, hidden), nn.SiLU(), nn.LayerNorm(hidden),
                nn.Dropout(dropout), nn.Linear(hidden, hidden), nn.SiLU(),
            )
            self.prior_gru = nn.GRU(d, gru_hidden, num_layers=1, batch_first=True)
            self.prior_temporal_proj = nn.Sequential(nn.Linear(gru_hidden, hidden), nn.SiLU())
            self.prior_fuse = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.SiLU(), nn.LayerNorm(hidden))
        else:
            self.prior_static = nn.Sequential(
                nn.Linear(d, hidden), nn.SiLU(), nn.LayerNorm(hidden),
                nn.Dropout(dropout), nn.Linear(hidden, hidden), nn.SiLU(),
            )
            self.prior_gru = None
            self.prior_temporal_proj = None
            self.prior_fuse = None
        inner = max(16, hidden // 2)
        self.prior_head = nn.Sequential(nn.Linear(hidden, inner), nn.SiLU(), nn.Linear(inner, 1))

        # Candidate innovation branch.  It never receives a hand-crafted
        # excitation score in the full proposal; it learns direction from raw
        # dynamics and is supervised directly in latent friction coordinates.
        if self.use_temporal:
            self.innovation_static = nn.Sequential(
                nn.Linear(3 * d, hidden), nn.SiLU(), nn.LayerNorm(hidden),
                nn.Dropout(dropout), nn.Linear(hidden, hidden), nn.SiLU(),
            )
            self.innovation_gru = nn.GRU(d, gru_hidden, num_layers=1, batch_first=True)
            self.innovation_temporal_proj = nn.Sequential(nn.Linear(gru_hidden, hidden), nn.SiLU())
            self.innovation_fuse = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.SiLU(), nn.LayerNorm(hidden))
            self.innovation_head = nn.Sequential(nn.Linear(hidden, inner), nn.SiLU(), nn.Linear(inner, 1))
        else:
            self.innovation_static = None
            self.innovation_gru = None
            self.innovation_temporal_proj = None
            self.innovation_fuse = None
            self.innovation_head = None

        # Friction-conditioned counterfactual dynamics model G(context, mu).
        # Only the causal prefix x[:,:-1] is encoded, preventing trivial copying
        # of the endpoint response used in the residual check.
        self.dynamics_gru = nn.GRU(d, gru_hidden, num_layers=1, batch_first=True)
        self.dynamics_context = nn.Sequential(nn.Linear(gru_hidden, hidden), nn.SiLU(), nn.LayerNorm(hidden))
        self.dynamics_head = nn.Sequential(
            nn.Linear(hidden + 1, hidden), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2 if hidden >= 16 else hidden), nn.SiLU(),
            nn.Linear(hidden // 2 if hidden >= 16 else hidden, dyn_out),
        )

        # Native UQ representation combines state, innovation and dynamics context.
        self.uq_fuse = nn.Sequential(nn.Linear(3 * hidden, hidden), nn.SiLU(), nn.LayerNorm(hidden))
        self.scale_head: ResidualScaleHead | None = None
        self.conformal_q: float | None = None
        self.conformal_block_size: int | None = None
        self.information_beta: float = 1.0

        # Legacy monotone excitation proxy kept only for an explicit ablation.
        proxy_slope = 4.0 if gate_init_slope is None else max(float(gate_init_slope), 1e-3)
        proxy_thr = 0.5 if gate_init_threshold is None else min(max(float(gate_init_threshold), 1e-4), 1.0-1e-4)
        inv_sp = math.log(math.expm1(proxy_slope)) if proxy_slope < 20 else proxy_slope
        self.proxy_slope_raw = nn.Parameter(torch.tensor(inv_sp, dtype=torch.float32))
        self.proxy_threshold_raw = nn.Parameter(torch.tensor(math.log(proxy_thr/(1.0-proxy_thr)), dtype=torch.float32))

    @staticmethod
    def _logit(p: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
        p = torch.clamp(p, eps, 1.0 - eps)
        return torch.log(p) - torch.log1p(-p)

    def _stats(self, x: torch.Tensor) -> torch.Tensor:
        if self.endpoint_only or not self.use_temporal:
            return x[:, -1, :]
        return torch.cat([x[:, -1, :], x.mean(dim=1), x.std(dim=1, unbiased=False)], dim=-1)

    def reliability(self, excitation: torch.Tensor) -> torch.Tensor:
        """Legacy monotone proxy used only by ``safegrip_excitation_proxy``."""
        e = torch.clamp(excitation.reshape(-1), 0.0, 1.0)
        slope = nn.functional.softplus(self.proxy_slope_raw) + 1e-4
        threshold = torch.sigmoid(self.proxy_threshold_raw)
        return torch.sigmoid(slope * (e - threshold))

    def _excitation_from_input(self, x: torch.Tensor, excitation: torch.Tensor | None) -> torch.Tensor:
        if excitation is not None:
            return torch.clamp(excitation.reshape(-1).to(dtype=x.dtype, device=x.device), 0.0, 1.0)
        if self.excitation_index is None:
            return torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)
        return torch.clamp(x[:, -1, self.excitation_index], 0.0, 1.0)

    def _support_to_latent(self, mu: torch.Tensor, lower: torch.Tensor | None,
                           upper: float | torch.Tensor) -> torch.Tensor:
        upper_t = torch.as_tensor(upper, dtype=mu.dtype, device=mu.device)
        if self.use_bound:
            if lower is None:
                raise ValueError("SafeGripCINet requires lower when use_bound=True")
            span = torch.clamp(upper_t - lower, min=1e-6)
            frac = (torch.clamp(mu, min=0.0) - lower) / span
        else:
            frac = torch.clamp(mu, min=0.0) / torch.clamp(upper_t, min=1e-6)
        return self._logit(frac)

    def _latent_to_support(self, q: torch.Tensor, lower: torch.Tensor | None,
                           upper: float | torch.Tensor) -> torch.Tensor:
        upper_t = torch.as_tensor(upper, dtype=q.dtype, device=q.device)
        if self.use_bound:
            if lower is None:
                raise ValueError("SafeGripCINet requires lower when use_bound=True")
            span = torch.clamp(upper_t - lower, min=1e-6)
            return lower + span * torch.sigmoid(q)
        return upper_t * torch.sigmoid(q)

    def _encode_prior(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.use_temporal:
            s = self.prior_static(self._stats(x))
            y, _ = self.prior_gru(x)
            t = self.prior_temporal_proj(y[:, -1])
            h = self.prior_fuse(torch.cat([s, t], dim=-1))
        else:
            h = self.prior_static(x[:, -1, :])
        return h, self.prior_head(h).squeeze(-1)

    def _encode_innovation(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.use_temporal:
            z = torch.zeros((x.shape[0], self.hidden), dtype=x.dtype, device=x.device)
            return z, torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)
        short = x[:, -min(self.evidence_window, x.shape[1]):, :]
        s = self.innovation_static(torch.cat([
            short[:, -1, :], short.mean(dim=1), short.std(dim=1, unbiased=False)
        ], dim=-1))
        y, _ = self.innovation_gru(short)
        t = self.innovation_temporal_proj(y[:, -1])
        h = self.innovation_fuse(torch.cat([s, t], dim=-1))
        nu = self.delta_scale * torch.tanh(self.innovation_head(h).squeeze(-1))
        return h, nu

    def _encode_dynamics_context(self, x: torch.Tensor) -> torch.Tensor:
        prefix = x[:, :-1, :] if x.shape[1] > 1 else x
        y, _ = self.dynamics_gru(prefix)
        return self.dynamics_context(y[:, -1])

    def dynamics_prediction_from_context(self, context_h: torch.Tensor, mu: torch.Tensor,
                                         upper: float | torch.Tensor = 1.3) -> torch.Tensor:
        upper_t = torch.as_tensor(upper, dtype=mu.dtype, device=mu.device)
        mu_norm = (mu / torch.clamp(upper_t, min=1e-6)).reshape(-1, 1)
        return self.dynamics_head(torch.cat([context_h, mu_norm], dim=-1))

    def dynamics_prediction(self, x: torch.Tensor, mu: torch.Tensor,
                            upper: float | torch.Tensor = 1.3) -> tuple[torch.Tensor, torch.Tensor]:
        h = self._encode_dynamics_context(x)
        pred = self.dynamics_prediction_from_context(h, mu, upper)
        idx = torch.as_tensor(self.dynamics_indices, dtype=torch.long, device=x.device)
        target = torch.index_select(x[:, -1, :], 1, idx)
        return pred, target

    def _counterfactual_authority(self, x: torch.Tensor, dyn_h: torch.Tensor,
                                  prior_mu: torch.Tensor, cand_mu: torch.Tensor,
                                  upper: float | torch.Tensor,
                                  excitation: torch.Tensor | None = None) -> tuple[torch.Tensor, ...]:
        idx = torch.as_tensor(self.dynamics_indices, dtype=torch.long, device=x.device)
        observed = torch.index_select(x[:, -1, :], 1, idx)
        upper_t = torch.as_tensor(upper, dtype=prior_mu.dtype, device=prior_mu.device)
        upper_scalar = float(upper_t.detach().cpu())

        # Multi-scale local counterfactual observability.  A single finite
        # difference can be spuriously large because of local curvature or a
        # dynamics-model artifact.  Estimate dG/dmu at three symmetric scales,
        # aggregate the information robustly, and explicitly discount poor
        # cross-scale consistency.  The single-scale path is kept as a clean
        # ablation and for backward compatibility.
        if self.use_multiscale_counterfactual and self.counterfactual_scale_span > 1.0 + 1e-8:
            span = self.counterfactual_scale_span
            scales = (1.0 / span, 1.0, span)
        else:
            scales = (1.0,)
        sensitivities = []
        scale_information = []
        for scale in scales:
            dmu = float(self.counterfactual_delta) * float(scale)
            mu_plus = torch.clamp(prior_mu + dmu, min=0.0, max=upper_scalar)
            mu_minus = torch.clamp(prior_mu - dmu, min=0.0, max=upper_scalar)
            g_plus = self.dynamics_prediction_from_context(dyn_h, mu_plus, upper)
            g_minus = self.dynamics_prediction_from_context(dyn_h, mu_minus, upper)
            denom = torch.clamp((mu_plus - mu_minus).reshape(-1, 1), min=1e-4)
            sens = (g_plus - g_minus) / denom
            sensitivities.append(sens)
            scale_information.append(torch.mean(sens.square(), dim=-1))

        sensitivity_stack = torch.stack(sensitivities, dim=1)  # [B, S, D]
        information_stack = torch.stack(scale_information, dim=1)  # [B, S]
        sensitivity = torch.mean(sensitivity_stack, dim=1)
        if information_stack.shape[1] > 1:
            # Median prevents one pathological scale from creating authority.
            information_raw = torch.median(information_stack, dim=1).values
            info_mean = torch.mean(information_stack, dim=1)
            info_std = torch.std(information_stack, dim=1, unbiased=False)
            information_scale_cv = info_std / torch.clamp(info_mean, min=1e-8)
            local_linearity = 1.0 / (1.0 + self.linearity_penalty * information_scale_cv.square())
        else:
            information_raw = information_stack[:, 0]
            information_scale_cv = torch.zeros_like(information_raw)
            local_linearity = torch.ones_like(information_raw)
        if not self.use_linearity_consistency:
            local_linearity = torch.ones_like(local_linearity)
        info_gain = information_raw / (information_raw + self.identifiability_lambda)
        info_gain = torch.clamp(info_gain * local_linearity, 0.0, 1.0)

        g_prior = self.dynamics_prediction_from_context(dyn_h, prior_mu, upper)
        g_cand = self.dynamics_prediction_from_context(dyn_h, cand_mu, upper)
        residual_prior = torch.mean((observed - g_prior).square(), dim=-1)
        residual_candidate = torch.mean((observed - g_cand).square(), dim=-1)
        improvement = residual_prior - residual_candidate

        # Scale-free counterfactual evidence.  The previous raw residual
        # difference was often ~1e-4 on standardized LiRA channels, making a
        # symmetric sigmoid collapse to ~0.5 and unnecessarily halve every
        # update.  The normalized quantity is bounded and comparable across
        # operating regimes.
        residual_scale = torch.clamp(residual_prior + residual_candidate, min=1e-8)
        normalized_improvement = improvement / residual_scale

        # Asymmetric veto: neutral/slightly noisy evidence leaves authority near
        # one; only a materially worse candidate is attenuated.  This preserves
        # the empirically useful identifiability-only path while still providing
        # a safety-oriented rejection mechanism.
        veto_probability = torch.sigmoid(
            self.acceptance_temperature * (
                -normalized_improvement - self.acceptance_tolerance - self.acceptance_margin
            )
        )
        acceptance = 1.0 - self.acceptance_strength * veto_probability

        # Local inverse-dynamics correction from one damped Gauss--Newton step.
        # e ~= s * dmu, so dmu = <s,e>/(||s||^2 + ridge).  The step is clipped
        # to a physical trust region and detached by the training loss when used
        # as a pseudo-target, preventing the estimator from gaming G through the
        # agreement term.
        residual_vector = observed - g_prior
        sens_energy = torch.sum(sensitivity.square(), dim=-1)
        cf_delta_mu = torch.sum(sensitivity * residual_vector, dim=-1) / (
            sens_energy + self.inverse_dynamics_ridge
        )
        cf_delta_mu = torch.clamp(
            cf_delta_mu, -self.inverse_dynamics_max_step, self.inverse_dynamics_max_step
        )
        candidate_delta_mu = cand_mu - prior_mu
        agreement_scale = torch.clamp(
            torch.abs(candidate_delta_mu) + torch.abs(cf_delta_mu), min=1e-4
        )
        # Normalized correction agreement in [0,1].  Opposite-direction
        # corrections map to 0, equal corrections to 1.  Earlier releases
        # remapped this already nonnegative ratio through 0.5*(x+1), which
        # unintentionally compressed the useful disagreement range to [0.5,1].
        cf_agreement = torch.clamp(
            1.0 - torch.abs(candidate_delta_mu - cf_delta_mu) / agreement_scale,
            min=0.0,
            max=1.0,
        )

        # Select the authority source *before* the agreement veto.  This is
        # essential for clean ablations: no-identifiability must not leak the
        # learned information score through the veto, and the handcrafted
        # excitation comparator must use its proxy everywhere authority is
        # confidence-weighted while keeping the rest of the architecture fixed.
        if self.use_excitation_proxy:
            e = self._excitation_from_input(x, excitation)
            info_gain = self.reliability(e)
        elif not self.use_identifiability:
            info_gain = torch.ones_like(info_gain)

        # Agreement-aware asymmetric veto.  It never boosts an update and is
        # deliberately mild by default: only strong disagreement between the
        # learned candidate and a local inverse-dynamics correction can further
        # attenuate authority.  This makes inverse-dynamics consistency part of
        # inference rather than only an auxiliary training loss.
        agreement_veto_probability = torch.sigmoid(
            self.agreement_temperature * (self.agreement_threshold - cf_agreement)
        )
        if self.use_agreement_veto and self.use_acceptance:
            # The inverse-dynamics correction itself is unreliable when the
            # active authority source is weak.  For the full model that source
            # is counterfactual identifiability; for the proxy ablation it is
            # the handcrafted excitation score.
            acceptance = acceptance * (
                1.0 - self.agreement_strength * agreement_veto_probability * info_gain
            )
        else:
            agreement_veto_probability = torch.zeros_like(agreement_veto_probability)

        if not self.use_acceptance:
            acceptance = torch.ones_like(acceptance)
            veto_probability = torch.zeros_like(veto_probability)
            agreement_veto_probability = torch.zeros_like(agreement_veto_probability)

        if not self.use_gate:
            authority = torch.ones_like(info_gain)
        else:
            authority = torch.clamp(info_gain * acceptance, 0.0, 1.0)
        return (
            authority, info_gain, acceptance, information_raw,
            residual_prior, residual_candidate, normalized_improvement,
            veto_probability, cf_delta_mu, cf_agreement,
            information_scale_cv, local_linearity, agreement_veto_probability,
        )

    def forward_details(self, x: torch.Tensor, lower: torch.Tensor | None = None,
                        upper: float | torch.Tensor = 1.3,
                        excitation: torch.Tensor | None = None,
                        prior_mu: torch.Tensor | None = None,
                        prior_mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        prior_h, q_context = self._encode_prior(x)
        context_prior = self._latent_to_support(q_context, lower, upper)

        if prior_mu is None or not self.use_persistent_state:
            q_prior = q_context
            prior_prediction = context_prior
            persistent_mask = torch.zeros_like(q_context, dtype=torch.bool)
        else:
            pmu = prior_mu.reshape(-1).to(dtype=x.dtype, device=x.device)
            if prior_mask is None:
                persistent_mask = torch.ones_like(q_context, dtype=torch.bool)
            else:
                persistent_mask = prior_mask.reshape(-1).to(device=x.device, dtype=torch.bool)
            mixed_mu = self.state_persistence * pmu + (1.0-self.state_persistence) * context_prior
            q_state = self._support_to_latent(mixed_mu, lower, upper)
            q_prior = torch.where(persistent_mask, q_state, q_context)
            state_prior = self._latent_to_support(q_prior, lower, upper)
            prior_prediction = state_prior

        innovation_h, innovation = self._encode_innovation(x)
        if not self.use_innovation:
            innovation = torch.zeros_like(innovation)
        q_candidate = q_prior + innovation
        candidate_prediction = self._latent_to_support(q_candidate, lower, upper)
        dyn_h = self._encode_dynamics_context(x)

        (authority, info_gain, acceptance, information_raw, r_prior, r_candidate,
         normalized_improvement, veto_probability, cf_delta_mu, cf_agreement,
         information_scale_cv, local_linearity, agreement_veto_probability) = self._counterfactual_authority(
            x, dyn_h, prior_prediction, candidate_prediction, upper, excitation=excitation
        )
        effective_innovation = authority * innovation
        q = q_prior + effective_innovation
        pred = self._latent_to_support(q, lower, upper)
        uq_h = self.uq_fuse(torch.cat([prior_h, innovation_h, dyn_h], dim=-1))

        return {
            "prediction": pred,
            "prior_prediction": prior_prediction,
            "context_prior_prediction": context_prior,
            "candidate_prediction": candidate_prediction,
            "latent": q,
            "prior_latent": q_prior,
            "evidence_delta": innovation,
            "innovation": innovation,
            "effective_innovation": effective_innovation,
            "reliability": authority,
            "authority": authority,
            "identifiability": info_gain,
            "acceptance": acceptance,
            "information_raw": information_raw,
            "dynamics_residual_prior": r_prior,
            "dynamics_residual_candidate": r_candidate,
            "normalized_improvement": normalized_improvement,
            "veto_probability": veto_probability,
            "counterfactual_delta_mu": cf_delta_mu,
            "counterfactual_agreement": cf_agreement,
            "information_scale_cv": information_scale_cv,
            "local_linearity": local_linearity,
            "agreement_veto_probability": agreement_veto_probability,
            "persistent_state_used": persistent_mask.to(dtype=x.dtype),
            "features": uq_h,
        }

    def forward(self, x: torch.Tensor, lower: torch.Tensor | None = None,
                upper: float | torch.Tensor = 1.3,
                excitation: torch.Tensor | None = None,
                prior_mu: torch.Tensor | None = None,
                prior_mask: torch.Tensor | None = None):
        d = self.forward_details(
            x, lower=lower, upper=upper, excitation=excitation,
            prior_mu=prior_mu, prior_mask=prior_mask,
        )
        return d["prediction"], d["latent"], d["reliability"]


class SafeGripCI11Net(SafeGripCINet):
    """SafeGrip-CI v1.1 with dual correction experts and learned arbitration.

    v1.1 keeps the v1.0 counterfactual dynamics model as an observability and
    inverse-dynamics instrument, but no longer treats identifiability or
    disagreement as a chain of multiplicative gates.  Instead it estimates
    three mutually exclusive actions at every endpoint: keep the persistent
    prior, apply the learned neural correction, or apply a local
    inverse-dynamics correction.  A small arbitration head mixes those actions
    using only inference-time evidence.

    The persistent state can also use a learned, sample-dependent persistence
    coefficient.  The neural innovation is factorized into direction and
    magnitude, making the most failure-prone part of the update explicitly
    trainable and auditable.
    """

    def __init__(self, *args,
                 use_dual_expert: bool = False,
                 use_learned_arbitration: bool = False,
                 use_inverse_expert: bool = True,
                 use_adaptive_persistence: bool = False,
                 use_split_innovation: bool = False,
                 persistence_min: float = 0.05,
                 persistence_max: float = 0.98,
                 inverse_expert_scale: float = 1.0,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.use_dual_expert = bool(use_dual_expert)
        self.use_learned_arbitration = bool(use_learned_arbitration) and self.use_dual_expert
        self.use_inverse_expert = bool(use_inverse_expert) and self.use_dual_expert
        self.use_adaptive_persistence = bool(use_adaptive_persistence) and self.use_persistent_state
        self.use_split_innovation = bool(use_split_innovation) and self.use_innovation
        self.persistence_min = min(max(float(persistence_min), 0.0), 0.95)
        self.persistence_max = min(max(float(persistence_max), self.persistence_min + 1e-3), 0.999)
        self.inverse_expert_scale = max(0.0, float(inverse_expert_scale))
        inner = max(16, self.hidden // 2)

        # Factorized innovation: sign/direction and magnitude are optimized
        # separately.  This retains a bounded latent correction while giving the
        # training objective direct access to direction errors.
        self.innovation_direction_head = nn.Sequential(
            nn.Linear(self.hidden, inner), nn.SiLU(), nn.Linear(inner, 1)
        )
        self.innovation_magnitude_head = nn.Sequential(
            nn.Linear(self.hidden, inner), nn.SiLU(), nn.Linear(inner, 1)
        )

        # Adaptive state persistence starts at the configured scalar persistence,
        # then learns a context-dependent deviation.  Initializing the final
        # layer to zero makes the v1.1 model safe to warm-start from v1.0 behavior.
        self.persistence_head = nn.Sequential(
            nn.Linear(self.hidden, inner), nn.SiLU(), nn.Linear(inner, 1)
        )
        final_p = self.persistence_head[-1]
        nn.init.zeros_(final_p.weight)
        frac = (self.state_persistence - self.persistence_min) / max(
            self.persistence_max - self.persistence_min, 1e-6
        )
        frac = min(max(frac, 1e-4), 1.0 - 1e-4)
        nn.init.constant_(final_p.bias, math.log(frac / (1.0 - frac)))

        # Evidence vector = [I, A, normalized residual improvement,
        # inverse correction, neural correction, direction agreement,
        # magnitude agreement, cross-scale variability].
        arb_in = 2 * self.hidden + 8
        self.arbitration_head = nn.Sequential(
            nn.Linear(arb_in, self.hidden), nn.SiLU(), nn.LayerNorm(self.hidden),
            nn.Dropout(0.05), nn.Linear(self.hidden, 3),
        )
        # Conservative initial policy: prefer the prior, allow modest neural and
        # inverse-dynamics corrections.  Training can move away from this.
        final_a = self.arbitration_head[-1]
        nn.init.zeros_(final_a.weight)
        with torch.no_grad():
            final_a.bias.copy_(torch.tensor([1.15, 0.0, -0.15], dtype=final_a.bias.dtype))

        self.uq_evidence = nn.Sequential(nn.Linear(5, self.hidden), nn.SiLU())
        self.uq_fuse_v11 = nn.Sequential(
            nn.Linear(4 * self.hidden, self.hidden), nn.SiLU(), nn.LayerNorm(self.hidden)
        )

    def _encode_innovation(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h, legacy_nu = super()._encode_innovation(x)
        if not self.use_split_innovation or not self.use_temporal:
            return h, legacy_nu
        direction = torch.tanh(self.innovation_direction_head(h).squeeze(-1))
        magnitude = self.delta_scale * torch.sigmoid(self.innovation_magnitude_head(h).squeeze(-1))
        return h, direction * magnitude

    def _innovation_parts(self, h: torch.Tensor, innovation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.use_split_innovation:
            direction = torch.tanh(self.innovation_direction_head(h).squeeze(-1))
            magnitude = self.delta_scale * torch.sigmoid(self.innovation_magnitude_head(h).squeeze(-1))
            return direction, magnitude
        direction = torch.tanh(innovation / max(self.delta_scale, 1e-6))
        magnitude = torch.abs(innovation)
        return direction, magnitude

    def _adaptive_rho(self, prior_h: torch.Tensor) -> torch.Tensor:
        if not self.use_adaptive_persistence:
            return torch.full(
                (prior_h.shape[0],), self.state_persistence,
                dtype=prior_h.dtype, device=prior_h.device,
            )
        raw = self.persistence_head(prior_h).squeeze(-1)
        return self.persistence_min + (self.persistence_max - self.persistence_min) * torch.sigmoid(raw)

    @staticmethod
    def _direction_agreement(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        scale = torch.clamp(torch.abs(a) + torch.abs(b), min=1e-4)
        return torch.tanh(6.0 * (a * b) / scale.square())

    @staticmethod
    def _magnitude_agreement(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        eps = 1e-4
        ratio = (torch.abs(a) + eps) / (torch.abs(b) + eps)
        return torch.exp(-torch.abs(torch.log(ratio)))

    def forward_details(self, x: torch.Tensor, lower: torch.Tensor | None = None,
                        upper: float | torch.Tensor = 1.3,
                        excitation: torch.Tensor | None = None,
                        prior_mu: torch.Tensor | None = None,
                        prior_mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        prior_h, q_context = self._encode_prior(x)
        context_prior = self._latent_to_support(q_context, lower, upper)
        adaptive_rho = self._adaptive_rho(prior_h)

        if prior_mu is None or not self.use_persistent_state:
            q_prior = q_context
            prior_prediction = context_prior
            persistent_mask = torch.zeros_like(q_context, dtype=torch.bool)
            adaptive_rho = torch.zeros_like(q_context)
        else:
            pmu = prior_mu.reshape(-1).to(dtype=x.dtype, device=x.device)
            if prior_mask is None:
                persistent_mask = torch.ones_like(q_context, dtype=torch.bool)
            else:
                persistent_mask = prior_mask.reshape(-1).to(device=x.device, dtype=torch.bool)
            mixed_mu = adaptive_rho * pmu + (1.0 - adaptive_rho) * context_prior
            q_state = self._support_to_latent(mixed_mu, lower, upper)
            q_prior = torch.where(persistent_mask, q_state, q_context)
            prior_prediction = self._latent_to_support(q_prior, lower, upper)
            adaptive_rho = torch.where(persistent_mask, adaptive_rho, torch.zeros_like(adaptive_rho))

        innovation_h, innovation = self._encode_innovation(x)
        if not self.use_innovation:
            innovation = torch.zeros_like(innovation)
        innovation_direction, innovation_magnitude = self._innovation_parts(innovation_h, innovation)
        q_candidate = q_prior + innovation
        candidate_prediction = self._latent_to_support(q_candidate, lower, upper)
        dyn_h = self._encode_dynamics_context(x)

        (legacy_authority, info_gain, acceptance, information_raw, r_prior, r_candidate,
         normalized_improvement, veto_probability, cf_delta_mu, cf_agreement,
         information_scale_cv, local_linearity, agreement_veto_probability) = self._counterfactual_authority(
            x, dyn_h, prior_prediction, candidate_prediction, upper, excitation=excitation
        )

        neural_delta_mu = candidate_prediction - prior_prediction
        direction_agreement = self._direction_agreement(neural_delta_mu, cf_delta_mu)
        magnitude_agreement = self._magnitude_agreement(neural_delta_mu, cf_delta_mu)

        if self.use_learned_arbitration:
            scalars = torch.stack([
                info_gain,
                acceptance,
                torch.clamp(normalized_improvement, -1.0, 1.0),
                cf_delta_mu / max(self.inverse_dynamics_max_step, 1e-4),
                neural_delta_mu / max(self.inverse_dynamics_max_step, 1e-4),
                direction_agreement,
                magnitude_agreement,
                torch.clamp(information_scale_cv, 0.0, 5.0) / 5.0,
            ], dim=-1)
            arbitration_logits = self.arbitration_head(torch.cat([innovation_h, dyn_h, scalars], dim=-1))
            expert_weights = torch.softmax(arbitration_logits, dim=-1)
            w_prior, w_neural, w_inverse = expert_weights.unbind(dim=-1)

            if not self.use_innovation:
                w_prior = w_prior + w_neural
                w_neural = torch.zeros_like(w_neural)
            if not self.use_inverse_expert:
                w_prior = w_prior + w_inverse
                w_inverse = torch.zeros_like(w_inverse)
            else:
                # Identifiability is an availability certificate for the local
                # inverse model, not a correctness score for the neural expert.
                available_inverse = w_inverse * torch.clamp(info_gain, 0.0, 1.0)
                w_prior = w_prior + (w_inverse - available_inverse)
                w_inverse = available_inverse

            fused_delta_mu = (
                w_neural * neural_delta_mu
                + w_inverse * self.inverse_expert_scale * cf_delta_mu
            )
            pred_mu = prior_prediction + fused_delta_mu
            upper_t = torch.as_tensor(upper, dtype=pred_mu.dtype, device=pred_mu.device)
            if self.use_bound:
                if lower is None:
                    raise ValueError("SafeGripCI11Net requires lower when use_bound=True")
                pred = torch.maximum(pred_mu, lower)
                pred = torch.minimum(pred, upper_t.expand_as(pred))
            else:
                pred = torch.clamp(pred_mu, min=0.0)
                pred = torch.minimum(pred, upper_t.expand_as(pred))
            q = self._support_to_latent(pred, lower, upper)
            effective_innovation = q - q_prior
            authority = torch.clamp(w_neural + w_inverse, 0.0, 1.0)
            expert_weights = torch.stack([w_prior, w_neural, w_inverse], dim=-1)
        else:
            arbitration_logits = torch.zeros((x.shape[0], 3), dtype=x.dtype, device=x.device)
            expert_weights = torch.stack([
                1.0 - legacy_authority,
                legacy_authority,
                torch.zeros_like(legacy_authority),
            ], dim=-1)
            effective_innovation = legacy_authority * innovation
            q = q_prior + effective_innovation
            pred = self._latent_to_support(q, lower, upper)
            authority = legacy_authority
            fused_delta_mu = pred - prior_prediction

        uq_evidence = self.uq_evidence(torch.stack([
            torch.clamp(info_gain, 0.0, 1.0),
            torch.clamp(cf_agreement, 0.0, 1.0),
            torch.clamp((direction_agreement + 1.0) * 0.5, 0.0, 1.0),
            torch.clamp(magnitude_agreement, 0.0, 1.0),
            torch.clamp(authority, 0.0, 1.0),
        ], dim=-1))
        if self.use_learned_arbitration:
            uq_h = self.uq_fuse_v11(torch.cat([prior_h, innovation_h, dyn_h, uq_evidence], dim=-1))
        else:
            uq_h = self.uq_fuse(torch.cat([prior_h, innovation_h, dyn_h], dim=-1))

        inverse_candidate = prior_prediction + self.inverse_expert_scale * cf_delta_mu
        upper_t = torch.as_tensor(upper, dtype=inverse_candidate.dtype, device=inverse_candidate.device)
        if self.use_bound and lower is not None:
            inverse_candidate = torch.maximum(inverse_candidate, lower)
        inverse_candidate = torch.minimum(torch.clamp(inverse_candidate, min=0.0), upper_t.expand_as(inverse_candidate))

        return {
            "prediction": pred,
            "prior_prediction": prior_prediction,
            "context_prior_prediction": context_prior,
            "candidate_prediction": candidate_prediction,
            "inverse_candidate_prediction": inverse_candidate,
            "latent": q,
            "prior_latent": q_prior,
            "evidence_delta": innovation,
            "innovation": innovation,
            "innovation_direction": innovation_direction,
            "innovation_magnitude": innovation_magnitude,
            "effective_innovation": effective_innovation,
            "fused_delta_mu": fused_delta_mu,
            "reliability": authority,
            "authority": authority,
            "identifiability": info_gain,
            "acceptance": acceptance,
            "information_raw": information_raw,
            "dynamics_residual_prior": r_prior,
            "dynamics_residual_candidate": r_candidate,
            "normalized_improvement": normalized_improvement,
            "veto_probability": veto_probability,
            "counterfactual_delta_mu": cf_delta_mu,
            "counterfactual_agreement": cf_agreement,
            "direction_agreement": direction_agreement,
            "magnitude_agreement": magnitude_agreement,
            "information_scale_cv": information_scale_cv,
            "local_linearity": local_linearity,
            "agreement_veto_probability": agreement_veto_probability,
            "persistent_state_used": persistent_mask.to(dtype=x.dtype),
            "adaptive_persistence": adaptive_rho,
            "arbitration_logits": arbitration_logits,
            "expert_weights": expert_weights,
            "expert_weight_prior": expert_weights[:, 0],
            "expert_weight_neural": expert_weights[:, 1],
            "expert_weight_inverse": expert_weights[:, 2],
            "features": uq_h,
        }


# Public/backward-compatible names.  v1.1 uses the dual-expert CI implementation.
SafeGripV3Net = SafeGripCI11Net
SafeGripV2Net = SafeGripCI11Net


class SafeGripBackboneNet(nn.Module):
    """Raw data-only temporal regression control with no CI mechanism."""
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
        inner = max(16, hidden // 2)
        self.head = nn.Sequential(nn.Linear(hidden, inner), nn.SiLU(), nn.Linear(inner, 1))
        self.scale_head: ResidualScaleHead | None = None
        self.conformal_q: float | None = None
        self.conformal_block_size: int | None = None
        self.information_beta: float = 0.0

    @staticmethod
    def _stats(x: torch.Tensor) -> torch.Tensor:
        return torch.cat([x[:, -1, :], x.mean(dim=1), x.std(dim=1, unbiased=False)], dim=-1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        s = self.static(self._stats(x))
        y, _ = self.gru(x)
        t = self.temporal(y[:, -1])
        return self.fuse(torch.cat([s, t], dim=-1))

    def forward_details(self, x: torch.Tensor, lower: torch.Tensor | None = None,
                        upper: float | torch.Tensor = 1.3,
                        excitation: torch.Tensor | None = None, **kwargs) -> dict[str, torch.Tensor]:
        h = self.encode(x)
        q = self.head(h).squeeze(-1)
        upper_t = torch.as_tensor(upper, dtype=q.dtype, device=q.device)
        pred = upper_t * torch.sigmoid(q)
        zero = torch.zeros_like(pred)
        one = torch.ones_like(pred)
        return {
            "prediction": pred, "prior_prediction": pred,
            "context_prior_prediction": pred, "candidate_prediction": pred,
            "latent": q, "prior_latent": q, "evidence_delta": zero,
            "innovation": zero, "effective_innovation": zero,
            "reliability": zero, "authority": zero,
            "identifiability": zero, "acceptance": one,
            "information_raw": zero, "dynamics_residual_prior": zero,
            "dynamics_residual_candidate": zero,
            "persistent_state_used": zero, "features": h,
        }

    def forward(self, x: torch.Tensor, lower: torch.Tensor | None = None,
                upper: float | torch.Tensor = 1.3, excitation: torch.Tensor | None = None,
                **kwargs):
        d = self.forward_details(x, lower=lower, upper=upper, excitation=excitation)
        return d["prediction"], d["latent"], d["reliability"]


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
