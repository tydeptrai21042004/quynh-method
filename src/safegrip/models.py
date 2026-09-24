from __future__ import annotations
import math
import torch
from torch import nn




class _DuInceptionModule(nn.Module):
    """Multi-scale temporal module used by the Du et al. dynamics branch."""
    def __init__(self, in_ch: int, filters: int = 32, bottleneck: int = 32):
        super().__init__()
        self.bottleneck = nn.Conv1d(in_ch, bottleneck, 1, bias=False) if in_ch > 1 else nn.Identity()
        conv_in = bottleneck if in_ch > 1 else in_ch
        self.convs = nn.ModuleList([
            nn.Conv1d(conv_in, filters, k, padding="same", bias=False) for k in (40, 20, 10)
        ])
        self.pool_branch = nn.Sequential(
            nn.MaxPool1d(3, stride=1, padding=1),
            nn.Conv1d(in_ch, filters, 1, bias=False),
        )
        self.bn = nn.BatchNorm1d(filters * 4)

    def forward(self, x):
        z = self.bottleneck(x)
        parts = [conv(z) for conv in self.convs]
        parts.append(self.pool_branch(x))
        return torch.relu(self.bn(torch.cat(parts, dim=1)))


class Du2023InceptionTime(nn.Module):
    """Dynamics-only InceptionTime adaptation of Du et al. (2023).

    The full source paper fuses vision and dynamics.  The primary fair benchmark
    intentionally uses only its vehicle-dynamics InceptionTime branch so it has
    no information advantage over SafeGrip-PFR-ECR.
    """
    def __init__(self, d: int, debug_scale: bool = False):
        super().__init__()
        filters = 8 if debug_scale else 32
        # Du et al. describe six residual blocks, each containing three
        # Inception modules.  The reduced debug path is smoke-test only.
        modules = 3 if debug_scale else 18
        out_ch = filters * 4
        blocks=[]
        in_ch=d
        for _ in range(modules):
            blocks.append(_DuInceptionModule(in_ch, filters=filters, bottleneck=filters))
            in_ch=out_ch
        self.blocks=nn.ModuleList(blocks)
        self.residual_every=3
        n_groups=(modules + self.residual_every - 1)//self.residual_every
        projections=[]
        group_in=d
        for _ in range(n_groups):
            projections.append(nn.Sequential(nn.Conv1d(group_in,out_ch,1,bias=False),nn.BatchNorm1d(out_ch)))
            group_in=out_ch
        self.projections=nn.ModuleList(projections)
        self.pool=nn.AdaptiveAvgPool1d(1)
        self.head=nn.Linear(out_ch,1)

    def forward(self,x):
        z=x.transpose(1,2)
        residual=z
        group=0
        for i,block in enumerate(self.blocks):
            z=block(z)
            if (i+1)%self.residual_every==0 or i==len(self.blocks)-1:
                z=torch.relu(z+self.projections[group](residual))
                residual=z
                group+=1
        return self.head(self.pool(z).squeeze(-1)).squeeze(-1)


class Todorovic2022CNN(nn.Module):
    """Paper-supported CNN adaptation of Todorovic et al. (2022).

    The publication verifies the *method family* (CNN regression) and the
    vehicle-signal input policy, but the repository does not have enough
    source evidence to claim an exact layer-by-layer reproduction.  The
    three-stage temporal CNN below is therefore deliberately labelled an
    adaptation.  It is trained/evaluated on exactly the same LiRA endpoints
    and target as SafeGrip-PFR-ECR.
    """
    def __init__(self, d: int, sequence_length: int = 100, dropout: float = 0.0,
                 debug_scale: bool = False):
        super().__init__()
        c1, c2, c3, dense = ((32, 32, 64, 64) if debug_scale else (128, 128, 256, 400))
        self.features = nn.Sequential(
            nn.Conv1d(d, c1, kernel_size=14, padding="same"), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(c1, c2, kernel_size=10, padding="same"), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(c2, c3, kernel_size=10, padding="same"), nn.ReLU(), nn.MaxPool1d(2),
        )
        if int(sequence_length) < 8:
            raise ValueError("Todorovic2022CNN requires sequence_length >= 8")
        self.head = nn.Sequential(
            nn.Flatten(), nn.Dropout(dropout), nn.LazyLinear(dense), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(dense, 1),
        )

    def forward(self, x):
        return self.head(self.features(x.transpose(1, 2))).squeeze(-1)


def _init_lampe_gru(gru: nn.GRU, head: nn.Linear) -> None:
    """Lampe et al.: orthogonal recurrent and Glorot input/dense weights.

    GRU gate matrices are initialized gate-by-gate so each recurrent block is
    orthogonal instead of applying one orthogonal transform to the concatenated
    3*hidden matrix.
    """
    hidden = gru.hidden_size
    for name, param in gru.named_parameters():
        if "weight_hh" in name:
            for gate in param.chunk(3, dim=0):
                nn.init.orthogonal_(gate)
        elif "weight_ih" in name:
            for gate in param.chunk(3, dim=0):
                nn.init.xavier_uniform_(gate)
        elif "bias" in name:
            nn.init.zeros_(param)
    nn.init.xavier_uniform_(head.weight)
    if head.bias is not None:
        nn.init.zeros_(head.bias)


class Lampe2023GRU(nn.Module):
    """Two-layer 256-unit GRU adaptation from Lampe et al. (2023)."""
    def __init__(self, d: int, hidden: int = 256, dropout: float = 0.0):
        super().__init__()
        self.rnn = nn.GRU(d, hidden, num_layers=2, batch_first=True, dropout=dropout)
        self.head = nn.Linear(hidden, 1)
        _init_lampe_gru(self.rnn, self.head)

    def forward(self, x):
        y, _ = self.rnn(x)
        return self.head(y[:, -1]).squeeze(-1)


def make_literature_baseline(name: str, d: int, *, sequence_length: int = 64,
                             debug_scale: bool = False, dropout: float = .1,
                             hidden: int | None = None, layers: int = 2,
                             heads: int = 4, ff_mult: int = 2):
    """Build one of the primary paper-supported neural comparators."""
    if name == "du2023_inceptiontime":
        return Du2023InceptionTime(d, debug_scale=debug_scale)
    if name == "todorovic2022_cnn":
        return Todorovic2022CNN(d, sequence_length=sequence_length, dropout=dropout, debug_scale=debug_scale)
    if name == "lampe2023_gru":
        return Lampe2023GRU(d, hidden=32 if debug_scale else 256, dropout=dropout)
    raise ValueError(f"No neural paper-baseline implementation for {name}")


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


class SafeGripCI12Net(nn.Module):
    """SafeGrip-CI v1.2: risk-aware selective physics correction.

    The v1.2 proposal deliberately simplifies the v1.1 state/arbitration stack.
    A strong raw-sensor temporal estimator produces the primary friction
    estimate.  A friction-conditioned dynamics model then supplies a local
    inverse-dynamics residual correction.  A learned utility gate predicts how
    much of that correction should be applied from *inference-available*
    evidence: temporal representation, local counterfactual observability,
    dynamics residual, correction magnitude, regime-change probability and
    aleatoric scale.

    Point prediction:

        mu_base = F_theta(x[t-L:t])
        J       = d G_phi(context, mu) / d mu |_(mu_base)
        dmu_ID  = <J, y_dyn - G_phi(context,mu_base)> / (||J||^2 + ridge)
        g       = sigmoid(H_psi(h, I, residual, dmu_ID, p_change, sigma))
        mu_hat  = Projection_C(mu_base + g * clip(dmu_ID))

    Counterfactual identifiability ``I`` is an observability feature, not a
    correctness score and not a multiplicative authority term.  The dynamics
    evidence is detached before it enters the correction/gate point path, so
    the point loss cannot improve by warping the dynamics model; G is trained
    by its own supervised and counterfactual-ranking objectives.
    """

    def __init__(
        self,
        d: int,
        dynamics_indices: list[int] | tuple[int, ...] | None = None,
        hidden: int = 96,
        gru_hidden: int = 96,
        dropout: float = 0.10,
        conv_channels: int | None = None,
        gru_layers: int = 2,
        counterfactual_delta: float = 0.08,
        identifiability_lambda: float = 1e-3,
        inverse_dynamics_ridge: float = 1e-3,
        inverse_dynamics_max_step: float = 0.10,
        physics_correction_scale: float = 1.0,
        use_physics_correction: bool = True,
        use_utility_gate: bool = True,
        use_identifiability_feature: bool = True,
        use_bound: bool = True,
        use_regime_head: bool = True,
        use_aleatoric_feature: bool = True,
        aleatoric_floor: float = 0.005,
    ):
        super().__init__()
        self.d = int(d)
        self.hidden = int(hidden)
        self.gru_hidden = int(gru_hidden)
        self.use_bound = bool(use_bound)
        self.use_physics_correction = bool(use_physics_correction)
        self.use_utility_gate = bool(use_utility_gate) and self.use_physics_correction
        self.use_identifiability_feature = bool(use_identifiability_feature)
        self.use_regime_head = bool(use_regime_head)
        self.use_aleatoric_feature = bool(use_aleatoric_feature)
        self.use_persistent_state = False  # v1.2 intentionally has no recursive friction state.
        self.counterfactual_delta = max(1e-4, float(counterfactual_delta))
        self.identifiability_lambda = max(1e-8, float(identifiability_lambda))
        self.inverse_dynamics_ridge = max(1e-8, float(inverse_dynamics_ridge))
        self.inverse_dynamics_max_step = max(1e-4, float(inverse_dynamics_max_step))
        self.physics_correction_scale = max(0.0, float(physics_correction_scale))
        self.aleatoric_floor = max(1e-5, float(aleatoric_floor))

        if dynamics_indices is None:
            dynamics_indices = list(range(min(4, self.d)))
        dynamics_indices = [int(i) for i in dynamics_indices if 0 <= int(i) < self.d]
        if not dynamics_indices:
            dynamics_indices = [0]
        self.dynamics_indices = tuple(dict.fromkeys(dynamics_indices))
        dyn_out = len(self.dynamics_indices)

        conv = int(conv_channels or max(32, hidden // 2))
        # A local convolutional front-end improves short-range motion-pattern
        # extraction while the two-layer GRU carries the longer temporal state.
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(self.d, conv, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(conv, conv, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.temporal_gru = nn.GRU(
            conv,
            self.gru_hidden,
            num_layers=max(1, int(gru_layers)),
            batch_first=True,
            dropout=float(dropout) if int(gru_layers) > 1 else 0.0,
        )
        self.static_encoder = nn.Sequential(
            nn.Linear(3 * self.d, hidden), nn.SiLU(), nn.LayerNorm(hidden),
            nn.Dropout(dropout), nn.Linear(hidden, hidden), nn.SiLU(),
        )
        self.temporal_proj = nn.Sequential(nn.Linear(self.gru_hidden, hidden), nn.SiLU())
        self.fuse = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.SiLU(), nn.LayerNorm(hidden), nn.Dropout(dropout)
        )
        inner = max(24, hidden // 2)
        self.base_head = nn.Sequential(nn.Linear(hidden, inner), nn.SiLU(), nn.Linear(inner, 1))
        self.change_head = nn.Sequential(nn.Linear(hidden, inner), nn.SiLU(), nn.Linear(inner, 1))
        self.aleatoric_head = nn.Sequential(nn.Linear(hidden, inner), nn.SiLU(), nn.Linear(inner, 1))

        # Dynamics model only sees the causal prefix; the endpoint response is
        # used as the observed residual target and therefore cannot be copied.
        self.dynamics_gru = nn.GRU(self.d, self.gru_hidden, num_layers=1, batch_first=True)
        self.dynamics_context = nn.Sequential(
            nn.Linear(self.gru_hidden, hidden), nn.SiLU(), nn.LayerNorm(hidden)
        )
        self.dynamics_head = nn.Sequential(
            nn.Linear(hidden + 1, hidden), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(hidden, inner), nn.SiLU(), nn.Linear(inner, dyn_out),
        )

        # Gate scalars: observability, residual magnitude, normalized ID step,
        # regime-change probability, aleatoric scale, sensitivity energy.
        self.gate_head = nn.Sequential(
            nn.Linear(hidden + 6, hidden), nn.SiLU(), nn.LayerNorm(hidden),
            nn.Dropout(dropout), nn.Linear(hidden, 1),
        )
        # Neutral/conservative initialization: start with a small correction and
        # let the utility supervision earn larger gate values.
        nn.init.zeros_(self.gate_head[-1].weight)
        nn.init.constant_(self.gate_head[-1].bias, -1.5)

        self.uq_fuse = nn.Sequential(
            nn.Linear(hidden + 5, hidden), nn.SiLU(), nn.LayerNorm(hidden)
        )
        self.scale_head: ResidualScaleHead | None = None
        self.conformal_q: float | None = None
        self.conformal_block_size: int | None = None
        self.information_beta: float = 0.5
        self.disagreement_beta: float = 0.25

    @staticmethod
    def _stats(x: torch.Tensor) -> torch.Tensor:
        return torch.cat([x[:, -1, :], x.mean(dim=1), x.std(dim=1, unbiased=False)], dim=-1)

    @staticmethod
    def _logit(p: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
        p = torch.clamp(p, eps, 1.0 - eps)
        return torch.log(p) - torch.log1p(-p)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        static = self.static_encoder(self._stats(x))
        z = self.temporal_conv(x.transpose(1, 2)).transpose(1, 2)
        y, _ = self.temporal_gru(z)
        temporal = self.temporal_proj(y[:, -1])
        return self.fuse(torch.cat([static, temporal], dim=-1))

    def _base_prediction(self, h: torch.Tensor, upper: float | torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        upper_t = torch.as_tensor(upper, dtype=h.dtype, device=h.device)
        latent = self.base_head(h).squeeze(-1)
        mu = upper_t * torch.sigmoid(latent)
        return mu, latent

    def _encode_dynamics_context(self, x: torch.Tensor) -> torch.Tensor:
        prefix = x[:, :-1, :] if x.shape[1] > 1 else x
        y, _ = self.dynamics_gru(prefix)
        return self.dynamics_context(y[:, -1])

    def dynamics_prediction_from_context(
        self, context_h: torch.Tensor, mu: torch.Tensor, upper: float | torch.Tensor = 1.3
    ) -> torch.Tensor:
        upper_t = torch.as_tensor(upper, dtype=mu.dtype, device=mu.device)
        mu_norm = (mu / torch.clamp(upper_t, min=1e-6)).reshape(-1, 1)
        return self.dynamics_head(torch.cat([context_h, mu_norm], dim=-1))

    def dynamics_prediction(
        self, x: torch.Tensor, mu: torch.Tensor, upper: float | torch.Tensor = 1.3
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = self._encode_dynamics_context(x)
        pred = self.dynamics_prediction_from_context(h, mu, upper)
        idx = torch.as_tensor(self.dynamics_indices, dtype=torch.long, device=x.device)
        target = torch.index_select(x[:, -1, :], 1, idx)
        return pred, target

    def _physics_evidence(
        self, x: torch.Tensor, base_mu: torch.Tensor, upper: float | torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Compute detached local observability and inverse-dynamics evidence."""
        idx = torch.as_tensor(self.dynamics_indices, dtype=torch.long, device=x.device)
        observed = torch.index_select(x[:, -1, :], 1, idx)
        dyn_h = self._encode_dynamics_context(x)
        upper_t = torch.as_tensor(upper, dtype=base_mu.dtype, device=base_mu.device)
        upper_scalar = float(upper_t.detach().cpu())

        # Prevent point-estimation losses from altering the local physics model.
        mu0 = base_mu.detach()
        dyn_detached = dyn_h.detach()
        dmu = self.counterfactual_delta
        mu_plus = torch.clamp(mu0 + dmu, min=0.0, max=upper_scalar)
        mu_minus = torch.clamp(mu0 - dmu, min=0.0, max=upper_scalar)
        g_plus = self.dynamics_prediction_from_context(dyn_detached, mu_plus, upper)
        g_minus = self.dynamics_prediction_from_context(dyn_detached, mu_minus, upper)
        denom = torch.clamp((mu_plus - mu_minus).reshape(-1, 1), min=1e-4)
        sensitivity = (g_plus - g_minus) / denom
        sensitivity = sensitivity.detach()
        information_raw = torch.mean(sensitivity.square(), dim=-1)
        identifiability = information_raw / (information_raw + self.identifiability_lambda)
        identifiability = torch.clamp(identifiability, 0.0, 1.0)

        g_base = self.dynamics_prediction_from_context(dyn_detached, mu0, upper).detach()
        residual_vector = observed.detach() - g_base
        residual_base = torch.mean(residual_vector.square(), dim=-1)
        sens_energy = torch.sum(sensitivity.square(), dim=-1)
        correction = torch.sum(sensitivity * residual_vector, dim=-1) / (
            sens_energy + self.inverse_dynamics_ridge
        )
        correction = torch.clamp(correction, -self.inverse_dynamics_max_step, self.inverse_dynamics_max_step)
        correction = correction.detach()

        mu_full = torch.clamp(mu0 + self.physics_correction_scale * correction, 0.0, upper_scalar)
        g_full = self.dynamics_prediction_from_context(dyn_detached, mu_full, upper).detach()
        residual_full = torch.mean((observed.detach() - g_full).square(), dim=-1)
        normalized_improvement = (residual_base - residual_full) / torch.clamp(
            residual_base + residual_full, min=1e-8
        )
        return {
            "context": dyn_h,
            "identifiability": identifiability.detach(),
            "information_raw": information_raw.detach(),
            "residual_base": residual_base.detach(),
            "residual_full": residual_full.detach(),
            "correction": correction,
            "normalized_improvement": normalized_improvement.detach(),
        }

    def _project(
        self, mu: torch.Tensor, lower: torch.Tensor | None, upper: float | torch.Tensor
    ) -> torch.Tensor:
        upper_t = torch.as_tensor(upper, dtype=mu.dtype, device=mu.device)
        out = torch.clamp(mu, min=0.0)
        out = torch.minimum(out, upper_t.expand_as(out))
        if self.use_bound:
            if lower is None:
                raise ValueError("SafeGripCI12Net requires lower when use_bound=True")
            out = torch.maximum(out, lower)
        return out

    def forward_details(
        self,
        x: torch.Tensor,
        lower: torch.Tensor | None = None,
        upper: float | torch.Tensor = 1.3,
        excitation: torch.Tensor | None = None,
        prior_mu: torch.Tensor | None = None,
        prior_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        del excitation, prior_mu, prior_mask  # v1.2 does not use proxy/state inputs.
        h = self.encode(x)
        base_mu, base_latent = self._base_prediction(h, upper)
        change_probability = (
            torch.sigmoid(self.change_head(h).squeeze(-1))
            if self.use_regime_head
            else torch.zeros_like(base_mu)
        )
        aleatoric_scale = (
            nn.functional.softplus(self.aleatoric_head(h).squeeze(-1)) + self.aleatoric_floor
            if self.use_aleatoric_feature
            else torch.full_like(base_mu, self.aleatoric_floor)
        )

        phys = self._physics_evidence(x, base_mu, upper)
        ident = phys["identifiability"]
        ident_for_gate = ident if self.use_identifiability_feature else torch.zeros_like(ident)
        corr = phys["correction"] if self.use_physics_correction else torch.zeros_like(base_mu)
        residual_feature = torch.log1p(torch.clamp(phys["residual_base"], min=0.0))
        corr_feature = corr / max(self.inverse_dynamics_max_step, 1e-6)
        scale_feature = (
            torch.clamp(aleatoric_scale / max(float(torch.as_tensor(upper).detach().cpu()), 1e-6), 0.0, 2.0)
            if self.use_aleatoric_feature
            else torch.zeros_like(base_mu)
        )
        info_raw_feature = torch.log1p(torch.clamp(phys["information_raw"], min=0.0))
        gate_features = torch.stack([
            ident_for_gate,
            residual_feature,
            corr_feature,
            change_probability,
            scale_feature,
            info_raw_feature,
        ], dim=-1)

        if not self.use_physics_correction:
            gate = torch.zeros_like(base_mu)
            gate_logit = torch.full_like(base_mu, -20.0)
        elif self.use_utility_gate:
            gate_logit = self.gate_head(torch.cat([h, gate_features], dim=-1)).squeeze(-1)
            gate = torch.sigmoid(gate_logit)
        else:
            gate = torch.ones_like(base_mu)
            gate_logit = torch.full_like(base_mu, 20.0)

        applied_correction = gate * self.physics_correction_scale * corr
        raw_prediction = base_mu + applied_correction
        prediction = self._project(raw_prediction, lower, upper)
        physics_candidate_raw = base_mu + self.physics_correction_scale * corr
        physics_candidate = self._project(physics_candidate_raw, lower, upper)

        # Dynamics residual after the actually applied correction is a diagnostic
        # only; it is not used to backpropagate point loss into G.
        dyn_h_det = phys["context"].detach()
        idx = torch.as_tensor(self.dynamics_indices, dtype=torch.long, device=x.device)
        observed = torch.index_select(x[:, -1, :], 1, idx).detach()
        g_final = self.dynamics_prediction_from_context(dyn_h_det, prediction.detach(), upper).detach()
        residual_final = torch.mean((observed - g_final).square(), dim=-1)

        upper_t = torch.as_tensor(upper, dtype=prediction.dtype, device=prediction.device)
        frac = torch.clamp(prediction / torch.clamp(upper_t, min=1e-6), 1e-5, 1.0 - 1e-5)
        latent = self._logit(frac)
        gate_confidence = torch.clamp(2.0 * torch.abs(gate - 0.5), 0.0, 1.0)
        uq_scalars = torch.stack([
            ident,
            gate,
            gate_confidence,
            change_probability,
            torch.clamp(scale_feature, 0.0, 1.0),
        ], dim=-1)
        uq_h = self.uq_fuse(torch.cat([h, uq_scalars], dim=-1))

        zero = torch.zeros_like(prediction)
        one = torch.ones_like(prediction)
        # Compatibility fields intentionally map old names onto the new
        # semantics so existing result exporters remain usable.
        return {
            "prediction": prediction,
            "raw_prediction": raw_prediction,
            "base_prediction": base_mu,
            "prior_prediction": base_mu,
            "context_prior_prediction": base_mu,
            "candidate_prediction": physics_candidate,
            "inverse_candidate_prediction": physics_candidate,
            "inverse_candidate_raw": physics_candidate_raw,
            "latent": latent,
            "prior_latent": base_latent,
            "evidence_delta": corr,
            "innovation": corr,
            "effective_innovation": applied_correction,
            "fused_delta_mu": applied_correction,
            "reliability": gate,
            "authority": gate,
            "physics_gate": gate,
            "gate_logit": gate_logit,
            "identifiability": ident,
            "acceptance": gate,
            "information_raw": phys["information_raw"],
            "dynamics_residual_prior": phys["residual_base"],
            "dynamics_residual_candidate": residual_final,
            "normalized_improvement": phys["normalized_improvement"],
            "veto_probability": 1.0 - gate,
            "counterfactual_delta_mu": corr,
            "counterfactual_agreement": gate_confidence,
            "information_scale_cv": zero,
            "local_linearity": one,
            "agreement_veto_probability": zero,
            "persistent_state_used": zero,
            "adaptive_persistence": zero,
            "expert_weights": torch.stack([zero, 1.0 - gate, gate], dim=-1),
            "expert_weight_prior": zero,
            "expert_weight_neural": 1.0 - gate,
            "expert_weight_inverse": gate,
            "direction_agreement": torch.tanh(8.0 * corr * applied_correction),
            "magnitude_agreement": gate,
            "change_probability": change_probability,
            "aleatoric_scale": aleatoric_scale,
            "physics_correction": corr,
            "applied_physics_correction": applied_correction,
            "gate_confidence": gate_confidence,
            "features": uq_h,
        }

    def forward(
        self,
        x: torch.Tensor,
        lower: torch.Tensor | None = None,
        upper: float | torch.Tensor = 1.3,
        excitation: torch.Tensor | None = None,
        **kwargs,
    ):
        d = self.forward_details(x, lower=lower, upper=upper, excitation=excitation, **kwargs)
        return d["prediction"], d["latent"], d["reliability"]


class SafeGripCI13Net(SafeGripCI12Net):
    """SafeGrip-CI v1.3: counterfactual-energy guided selective physics.

    v1.3 replaces the single finite-difference/Gauss--Newton physics step with
    an explicit local friction-hypothesis energy landscape.  The dynamics model
    evaluates K nearby friction hypotheses, a soft energy posterior forms the
    physics candidate, entropy measures local identifiability, and two separate
    heads answer the two questions that were entangled in v1.2:

      1. ``benefit_probability``: should the physics candidate be trusted?
      2. ``correction_fraction``: if trusted, how much of it should be applied?

    The final correction gate is their product.  All physics evidence entering
    the point path is detached, so the point-estimation loss cannot make the
    dynamics model look artificially informative.  The dynamics model is meant
    to be trained with the multi-negative contrastive objective implemented in
    :meth:`counterfactual_dynamics_loss`.
    """

    def __init__(
        self,
        d: int,
        dynamics_indices: list[int] | tuple[int, ...] | None = None,
        hidden: int = 96,
        gru_hidden: int = 96,
        dropout: float = 0.10,
        conv_channels: int | None = None,
        gru_layers: int = 2,
        counterfactual_delta: float = 0.08,
        identifiability_lambda: float = 1e-3,
        inverse_dynamics_ridge: float = 1e-3,
        inverse_dynamics_max_step: float = 0.10,
        physics_correction_scale: float = 1.0,
        use_physics_correction: bool = True,
        use_utility_gate: bool = True,
        use_identifiability_feature: bool = True,
        use_bound: bool = True,
        use_regime_head: bool = False,
        use_aleatoric_feature: bool = True,
        aleatoric_floor: float = 0.005,
        energy_grid_points: int = 7,
        energy_grid_radius: float | None = None,
        energy_temperature: float = 0.35,
        use_magnitude_head: bool = True,
        use_energy_improvement_feature: bool = True,
    ):
        super().__init__(
            d=d,
            dynamics_indices=dynamics_indices,
            hidden=hidden,
            gru_hidden=gru_hidden,
            dropout=dropout,
            conv_channels=conv_channels,
            gru_layers=gru_layers,
            counterfactual_delta=counterfactual_delta,
            identifiability_lambda=identifiability_lambda,
            inverse_dynamics_ridge=inverse_dynamics_ridge,
            inverse_dynamics_max_step=inverse_dynamics_max_step,
            physics_correction_scale=physics_correction_scale,
            use_physics_correction=use_physics_correction,
            use_utility_gate=use_utility_gate,
            use_identifiability_feature=use_identifiability_feature,
            use_bound=use_bound,
            use_regime_head=use_regime_head,
            use_aleatoric_feature=use_aleatoric_feature,
            aleatoric_floor=aleatoric_floor,
        )
        points=max(3,int(energy_grid_points))
        if points % 2 == 0:
            points += 1
        self.energy_grid_points=points
        self.energy_grid_radius=max(
            1e-4,
            float(inverse_dynamics_max_step if energy_grid_radius is None else energy_grid_radius),
        )
        self.energy_temperature=max(1e-3,float(energy_temperature))
        self.use_magnitude_head=bool(use_magnitude_head) and self.use_utility_gate
        self.use_energy_improvement_feature=bool(use_energy_improvement_feature)

        # Nine inference-available scalar signals accompany the temporal state:
        # entropy identifiability, energy improvement, base residual, signed and
        # absolute candidate step, posterior entropy, local curvature, optional
        # regime change probability, and aleatoric scale.
        gate_dim=self.hidden+9
        self.benefit_head=nn.Sequential(
            nn.Linear(gate_dim,self.hidden),nn.SiLU(),nn.LayerNorm(self.hidden),
            nn.Dropout(dropout),nn.Linear(self.hidden,1),
        )
        self.magnitude_head=nn.Sequential(
            nn.Linear(gate_dim,self.hidden),nn.SiLU(),nn.LayerNorm(self.hidden),
            nn.Dropout(dropout),nn.Linear(self.hidden,1),
        )
        nn.init.zeros_(self.benefit_head[-1].weight)
        nn.init.constant_(self.benefit_head[-1].bias,-1.25)
        nn.init.zeros_(self.magnitude_head[-1].weight)
        nn.init.constant_(self.magnitude_head[-1].bias,0.0)

        # v1.3 UQ representation uses identifiability + separate gate factors.
        self.uq_fuse=nn.Sequential(
            nn.Linear(self.hidden+6,self.hidden),nn.SiLU(),nn.LayerNorm(self.hidden)
        )

    def _candidate_offsets(self, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        return torch.linspace(
            -self.energy_grid_radius,self.energy_grid_radius,self.energy_grid_points,
            dtype=dtype,device=device,
        )

    def _energy_landscape(
        self, x: torch.Tensor, base_mu: torch.Tensor, upper: float | torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Evaluate local friction hypotheses with the detached dynamics model."""
        idx=torch.as_tensor(self.dynamics_indices,dtype=torch.long,device=x.device)
        observed=torch.index_select(x[:,-1,:],1,idx).detach()
        dyn_h=self._encode_dynamics_context(x)
        dyn_detached=dyn_h.detach()
        upper_t=torch.as_tensor(upper,dtype=base_mu.dtype,device=base_mu.device)

        mu0=base_mu.detach()
        offsets=self._candidate_offsets(mu0.dtype,mu0.device)
        candidates=torch.clamp(mu0[:,None]+offsets[None,:],min=0.0,max=upper_t)
        bsz,k=candidates.shape
        ctx=dyn_detached[:,None,:].expand(-1,k,-1).reshape(bsz*k,-1)
        pred=self.dynamics_prediction_from_context(ctx,candidates.reshape(-1),upper)
        pred=pred.reshape(bsz,k,-1).detach()
        energies=torch.mean((pred-observed[:,None,:]).square(),dim=-1)

        # Dimensionless per-sample scaling prevents arbitrary sensor scale from
        # collapsing the soft energy posterior.  Temperature controls only the
        # relative sharpness of the local landscape.
        emin=energies.min(dim=-1,keepdim=True).values
        spread=torch.mean(torch.abs(energies-emin),dim=-1,keepdim=True).clamp_min(1e-8)
        logits=-(energies-emin)/(self.energy_temperature*spread)
        weights=torch.softmax(logits,dim=-1)
        # Convert posterior imbalance into a displacement from the base rather
        # than averaging clipped hypothesis values directly.  This keeps a
        # flat posterior neutral even when mu_B is near a physical boundary,
        # where clipping would otherwise duplicate one side of the grid and
        # create a spurious inward correction.
        posterior_offset=torch.sum(weights*offsets[None,:],dim=-1)
        candidate_mu=torch.clamp(mu0+posterior_offset,min=0.0,max=upper_t)
        correction=torch.clamp(
            candidate_mu-mu0,
            min=-self.inverse_dynamics_max_step,
            max=self.inverse_dynamics_max_step,
        ).detach()

        entropy=-(weights*torch.log(weights.clamp_min(1e-8))).sum(dim=-1)
        entropy_norm=entropy/max(math.log(float(k)),1e-8)
        identifiability=torch.clamp(1.0-entropy_norm,0.0,1.0).detach()

        center=k//2
        residual_base=energies[:,center]
        # Evaluate the posterior-mean friction as a single interpretable
        # candidate rather than treating posterior expected energy as the final
        # physical residual.
        g_candidate=self.dynamics_prediction_from_context(
            dyn_detached,candidate_mu.detach(),upper
        ).detach()
        residual_candidate=torch.mean((g_candidate-observed).square(),dim=-1)
        normalized_improvement=(residual_base-residual_candidate)/torch.clamp(
            residual_base+residual_candidate,min=1e-8
        )

        if k>=3:
            left=energies[:,center-1]
            right=energies[:,center+1]
            local_scale=torch.clamp(torch.mean(energies,dim=-1),min=1e-8)
            curvature=torch.relu(left+right-2.0*residual_base)/local_scale
        else:
            curvature=torch.zeros_like(residual_base)

        # A bounded separation statistic is retained as ``information_raw`` for
        # backward-compatible exporters.  v1.3 identifiability itself is entropy.
        energy_mean=torch.mean(energies,dim=-1)
        separation=(energy_mean-energies.min(dim=-1).values)/torch.clamp(
            energy_mean+energies.min(dim=-1).values,min=1e-8
        )
        separation=torch.clamp(separation,min=0.0,max=1.0)
        return {
            "context":dyn_h,
            "candidate_grid":candidates.detach(),
            "energy_grid":energies.detach(),
            "energy_weights":weights.detach(),
            "candidate_mu":candidate_mu.detach(),
            "correction":correction,
            "identifiability":identifiability,
            "posterior_entropy":entropy_norm.detach(),
            "information_raw":separation.detach(),
            "residual_base":residual_base.detach(),
            "residual_candidate":residual_candidate.detach(),
            "normalized_improvement":normalized_improvement.detach(),
            "energy_curvature":curvature.detach(),
        }

    def counterfactual_dynamics_loss(
        self,
        x: torch.Tensor,
        true_mu: torch.Tensor,
        upper: float | torch.Tensor = 1.3,
        temperature: float = 0.25,
        negatives: int = 6,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Multi-negative contrastive objective that forces G to use friction.

        The positive hypothesis is the labelled training friction.  Symmetric
        negative hypotheses are generated around it and the energy of the true
        friction is trained to rank below all alternatives.  Labels are used
        only in this training objective; inference remains label free.
        """
        context=self._encode_dynamics_context(x)
        idx=torch.as_tensor(self.dynamics_indices,dtype=torch.long,device=x.device)
        target=torch.index_select(x[:,-1,:],1,idx)
        upper_t=torch.as_tensor(upper,dtype=true_mu.dtype,device=true_mu.device)
        n=max(2,int(negatives))
        # Use evenly-spaced non-zero offsets with a wider range than the local
        # inference grid so the dynamics representation must discriminate mu.
        half=max(1,n//2)
        mag=torch.linspace(
            self.counterfactual_delta,
            max(self.counterfactual_delta,self.energy_grid_radius*1.5),
            half,dtype=true_mu.dtype,device=true_mu.device,
        )
        offsets=torch.cat([-torch.flip(mag,dims=[0]),mag],dim=0)
        if offsets.numel()>n:
            offsets=offsets[:n]
        wrong=torch.clamp(true_mu[:,None]+offsets[None,:],min=0.0,max=upper_t)
        hypotheses=torch.cat([true_mu[:,None],wrong],dim=-1)
        bsz,k=hypotheses.shape
        ctx=context[:,None,:].expand(-1,k,-1).reshape(bsz*k,-1)
        pred=self.dynamics_prediction_from_context(ctx,hypotheses.reshape(-1),upper)
        pred=pred.reshape(bsz,k,-1)
        energy=torch.mean((pred-target[:,None,:]).square(),dim=-1)
        scale=torch.mean(torch.abs(energy-energy[:,:1]),dim=-1,keepdim=True).detach().clamp_min(1e-6)
        logits=-(energy-energy.min(dim=-1,keepdim=True).values)/(max(float(temperature),1e-3)*scale)
        labels=torch.zeros(bsz,dtype=torch.long,device=x.device)
        loss=nn.functional.cross_entropy(logits,labels)
        positive=energy[:,0]
        best_negative=energy[:,1:].min(dim=-1).values
        stats={
            "positive_energy":positive.detach(),
            "best_negative_energy":best_negative.detach(),
            "ranking_accuracy":(positive<best_negative).to(x.dtype).mean().detach(),
        }
        return loss,stats

    def _physics_evidence(
        self, x: torch.Tensor, base_mu: torch.Tensor, upper: float | torch.Tensor
    ) -> dict[str, torch.Tensor]:
        # Compatibility hook: callers of the old private method receive v1.3
        # landscape evidence instead of a Gauss--Newton derivative.
        return self._energy_landscape(x,base_mu,upper)

    def forward_details(
        self,
        x: torch.Tensor,
        lower: torch.Tensor | None = None,
        upper: float | torch.Tensor = 1.3,
        excitation: torch.Tensor | None = None,
        prior_mu: torch.Tensor | None = None,
        prior_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        del excitation,prior_mu,prior_mask
        h=self.encode(x)
        base_mu,base_latent=self._base_prediction(h,upper)
        change_probability=(
            torch.sigmoid(self.change_head(h).squeeze(-1))
            if self.use_regime_head else torch.zeros_like(base_mu)
        )
        aleatoric_scale=(
            nn.functional.softplus(self.aleatoric_head(h).squeeze(-1))+self.aleatoric_floor
            if self.use_aleatoric_feature else torch.full_like(base_mu,self.aleatoric_floor)
        )

        phys=self._energy_landscape(x,base_mu,upper)
        ident=phys["identifiability"]
        corr=phys["correction"] if self.use_physics_correction else torch.zeros_like(base_mu)
        upper_scalar=max(float(torch.as_tensor(upper).detach().cpu()),1e-6)
        ident_feature=ident if self.use_identifiability_feature else torch.zeros_like(ident)
        improve_feature=(
            phys["normalized_improvement"] if self.use_energy_improvement_feature
            else torch.zeros_like(base_mu)
        )
        residual_feature=torch.log1p(torch.clamp(phys["residual_base"],min=0.0))
        corr_feature=corr/max(self.inverse_dynamics_max_step,1e-6)
        abs_corr_feature=torch.abs(corr_feature)
        # Do not feed posterior entropy separately from identifiability: they
        # are complements, so doing both would make the no-identifiability
        # ablation leak the very feature it is meant to remove.  Energy
        # separation is a distinct counterfactual-evidence statistic.
        separation_feature=phys["information_raw"]
        curvature_feature=torch.tanh(torch.clamp(phys["energy_curvature"],min=0.0))
        scale_feature=(
            torch.clamp(aleatoric_scale/upper_scalar,0.0,2.0)
            if self.use_aleatoric_feature else torch.zeros_like(base_mu)
        )
        gate_scalars=torch.stack([
            ident_feature,improve_feature,residual_feature,corr_feature,
            abs_corr_feature,separation_feature,curvature_feature,
            change_probability,scale_feature,
        ],dim=-1)
        gate_input=torch.cat([h,gate_scalars],dim=-1)

        if not self.use_physics_correction:
            benefit_probability=torch.zeros_like(base_mu)
            correction_fraction=torch.zeros_like(base_mu)
            benefit_logit=torch.full_like(base_mu,-20.0)
            magnitude_logit=torch.full_like(base_mu,-20.0)
        elif not self.use_utility_gate:
            benefit_probability=torch.ones_like(base_mu)
            correction_fraction=torch.ones_like(base_mu)
            benefit_logit=torch.full_like(base_mu,20.0)
            magnitude_logit=torch.full_like(base_mu,20.0)
        else:
            benefit_logit=self.benefit_head(gate_input).squeeze(-1)
            benefit_probability=torch.sigmoid(benefit_logit)
            if self.use_magnitude_head:
                magnitude_logit=self.magnitude_head(gate_input).squeeze(-1)
                correction_fraction=torch.sigmoid(magnitude_logit)
            else:
                magnitude_logit=torch.full_like(base_mu,20.0)
                correction_fraction=torch.ones_like(base_mu)

        gate=benefit_probability*correction_fraction
        applied_correction=gate*self.physics_correction_scale*corr
        raw_prediction=base_mu+applied_correction
        prediction=self._project(raw_prediction,lower,upper)
        physics_candidate_raw=base_mu+self.physics_correction_scale*corr
        physics_candidate=self._project(physics_candidate_raw,lower,upper)

        dyn_h_det=phys["context"].detach()
        idx=torch.as_tensor(self.dynamics_indices,dtype=torch.long,device=x.device)
        observed=torch.index_select(x[:,-1,:],1,idx).detach()
        g_final=self.dynamics_prediction_from_context(dyn_h_det,prediction.detach(),upper).detach()
        residual_final=torch.mean((observed-g_final).square(),dim=-1)

        upper_t=torch.as_tensor(upper,dtype=prediction.dtype,device=prediction.device)
        frac=torch.clamp(prediction/torch.clamp(upper_t,min=1e-6),1e-5,1.0-1e-5)
        latent=self._logit(frac)
        benefit_confidence=torch.clamp(2.0*torch.abs(benefit_probability-0.5),0.0,1.0)
        uq_scalars=torch.stack([
            ident,benefit_probability,correction_fraction,gate,
            benefit_confidence,torch.clamp(scale_feature,0.0,1.0),
        ],dim=-1)
        uq_h=self.uq_fuse(torch.cat([h,uq_scalars],dim=-1))

        zero=torch.zeros_like(prediction); one=torch.ones_like(prediction)
        return {
            "prediction":prediction,
            "raw_prediction":raw_prediction,
            "base_prediction":base_mu,
            "prior_prediction":base_mu,
            "context_prior_prediction":base_mu,
            "candidate_prediction":physics_candidate,
            "inverse_candidate_prediction":physics_candidate,
            "inverse_candidate_raw":physics_candidate_raw,
            "latent":latent,
            "prior_latent":base_latent,
            "evidence_delta":corr,
            "innovation":corr,
            "effective_innovation":applied_correction,
            "fused_delta_mu":applied_correction,
            "reliability":gate,
            "authority":gate,
            "physics_gate":gate,
            "gate_logit":benefit_logit,
            "benefit_probability":benefit_probability,
            "benefit_logit":benefit_logit,
            "correction_fraction":correction_fraction,
            "magnitude_logit":magnitude_logit,
            "identifiability":ident,
            "acceptance":benefit_probability,
            "information_raw":phys["information_raw"],
            "posterior_entropy":phys["posterior_entropy"],
            "energy_curvature":phys["energy_curvature"],
            "dynamics_residual_prior":phys["residual_base"],
            "dynamics_residual_candidate":phys["residual_candidate"],
            "dynamics_residual_final":residual_final,
            "normalized_improvement":phys["normalized_improvement"],
            "veto_probability":1.0-benefit_probability,
            "counterfactual_delta_mu":corr,
            "counterfactual_agreement":benefit_confidence,
            "information_scale_cv":phys["posterior_entropy"],
            "local_linearity":torch.clamp(phys["energy_curvature"],0.0,1.0),
            "agreement_veto_probability":1.0-benefit_probability,
            "persistent_state_used":zero,
            "adaptive_persistence":zero,
            "expert_weights":torch.stack([zero,1.0-gate,gate],dim=-1),
            "expert_weight_prior":zero,
            "expert_weight_neural":1.0-gate,
            "expert_weight_inverse":gate,
            "direction_agreement":torch.tanh(8.0*corr*applied_correction),
            "magnitude_agreement":correction_fraction,
            "change_probability":change_probability,
            "aleatoric_scale":aleatoric_scale,
            "physics_correction":corr,
            "applied_physics_correction":applied_correction,
            "gate_confidence":benefit_confidence,
            "features":uq_h,
        }



class SafeGripCI14Net(SafeGripCI13Net):
    """SafeGrip-CI v1.4: strong temporal base + innovation-energy refinement.

    v1.4 keeps the counterfactual-energy idea from v1.3 but changes the parts
    that most directly limit point RMSE:

    * the primary estimator is a two-layer raw-sensor GRU with the same 256-unit
      scale used by the strongest literature GRU comparator in the TRUST setup;
    * the friction target is standardized using *training-only* statistics and
      the primary head is linear in standardized target space (no sigmoid
      compression);
    * the dynamics branch predicts endpoint *innovation* rather than the absolute
      endpoint, which makes temporal persistence a much less useful shortcut;
    * the local friction grid is boundary-aware and masked instead of duplicating
      clipped hypotheses;
    * identifiability combines posterior concentration with absolute energy
      margin, so tiny numerical differences do not look highly informative;
    * one continuous correction controller is the actual prediction gate.
      ``benefit_probability`` remains an auxiliary, interpretable diagnostic and
      is deliberately not multiplied into the point correction.

    These changes preserve the paper's central counterfactual-refinement idea
    while making the base estimator and optimization objective competitive with
    the RMSE-focused baselines.
    """

    def __init__(
        self,
        d: int,
        dynamics_indices: list[int] | tuple[int, ...] | None = None,
        hidden: int = 128,
        gru_hidden: int = 256,
        dropout: float = 0.10,
        conv_channels: int | None = None,
        gru_layers: int = 2,
        counterfactual_delta: float = 0.08,
        identifiability_lambda: float = 1e-3,
        inverse_dynamics_ridge: float = 1e-3,
        inverse_dynamics_max_step: float = 0.10,
        physics_correction_scale: float = 1.0,
        use_physics_correction: bool = True,
        use_utility_gate: bool = True,
        use_identifiability_feature: bool = True,
        use_bound: bool = True,
        use_regime_head: bool = False,
        use_aleatoric_feature: bool = True,
        aleatoric_floor: float = 0.005,
        energy_grid_points: int = 7,
        energy_grid_radius: float | None = None,
        energy_temperature: float = 0.35,
        use_magnitude_head: bool = True,
        use_energy_improvement_feature: bool = True,
        energy_noise_floor: float = 1e-4,
        energy_margin_threshold: float = 0.25,
        energy_margin_temperature: float = 0.15,
    ):
        super().__init__(
            d=d,
            dynamics_indices=dynamics_indices,
            hidden=hidden,
            gru_hidden=gru_hidden,
            dropout=dropout,
            conv_channels=conv_channels,
            gru_layers=gru_layers,
            counterfactual_delta=counterfactual_delta,
            identifiability_lambda=identifiability_lambda,
            inverse_dynamics_ridge=inverse_dynamics_ridge,
            inverse_dynamics_max_step=inverse_dynamics_max_step,
            physics_correction_scale=physics_correction_scale,
            use_physics_correction=use_physics_correction,
            use_utility_gate=use_utility_gate,
            use_identifiability_feature=use_identifiability_feature,
            use_bound=use_bound,
            use_regime_head=use_regime_head,
            use_aleatoric_feature=use_aleatoric_feature,
            aleatoric_floor=aleatoric_floor,
            energy_grid_points=energy_grid_points,
            energy_grid_radius=energy_grid_radius,
            energy_temperature=energy_temperature,
            use_magnitude_head=use_magnitude_head,
            use_energy_improvement_feature=use_energy_improvement_feature,
        )
        self.energy_noise_floor=max(1e-8,float(energy_noise_floor))
        self.energy_margin_threshold=float(energy_margin_threshold)
        self.energy_margin_temperature=max(1e-4,float(energy_margin_temperature))

        # Replace the v1.3 Conv1D/static fusion with a stronger, simpler raw-GRU
        # base.  Keeping the inherited attribute names means the existing staged
        # trainer, exporters and checkpoints remain easy to reason about.
        self.temporal_conv=nn.Identity()
        self.static_encoder=nn.Identity()
        self.temporal_gru=nn.GRU(
            self.d,
            self.gru_hidden,
            num_layers=max(1,int(gru_layers)),
            batch_first=True,
            dropout=float(dropout) if int(gru_layers)>1 else 0.0,
        )
        self.temporal_proj=nn.Sequential(
            nn.Linear(self.gru_hidden,self.hidden),nn.SiLU(),nn.LayerNorm(self.hidden),
            nn.Dropout(dropout),
        )
        self.fuse=nn.Identity()
        inner=max(32,self.hidden//2)
        self.base_head=nn.Sequential(
            nn.Linear(self.hidden,self.hidden),nn.SiLU(),nn.Dropout(dropout),
            nn.Linear(self.hidden,inner),nn.SiLU(),nn.Linear(inner,1),
        )
        self._init_gru(self.temporal_gru)

        # Training-only target statistics are assigned by the fitter.  Defaults
        # keep direct unit tests and old checkpoint loading well defined.
        self.register_buffer("target_mean",torch.tensor(0.0,dtype=torch.float32))
        self.register_buffer("target_std",torch.tensor(1.0,dtype=torch.float32))

    @staticmethod
    def _init_gru(gru: nn.GRU) -> None:
        for name,param in gru.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                # Each gate block is initialized independently.
                for block in param.chunk(3,dim=0):
                    nn.init.orthogonal_(block)
            elif "bias" in name:
                nn.init.zeros_(param)

    def set_target_stats(self, mean: float, std: float) -> None:
        std=max(float(std),1e-4)
        with torch.no_grad():
            self.target_mean.fill_(float(mean))
            self.target_std.fill_(std)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        y,_=self.temporal_gru(x)
        return self.temporal_proj(y[:,-1])

    def _base_prediction(
        self,h: torch.Tensor,upper: float | torch.Tensor
    ) -> tuple[torch.Tensor,torch.Tensor]:
        # z lives in standardized target space.  Mapping back with training-only
        # statistics avoids the strong output compression of mu_upper*sigmoid(z).
        z=self.base_head(h).squeeze(-1)
        raw=self.target_mean.to(z.dtype)+self.target_std.to(z.dtype)*z
        upper_t=torch.as_tensor(upper,dtype=z.dtype,device=z.device)
        mu=torch.clamp(raw,min=0.0,max=upper_t)
        return mu,z

    def standardized_target(self, y: torch.Tensor) -> torch.Tensor:
        return (y-self.target_mean.to(y.dtype))/torch.clamp(self.target_std.to(y.dtype),min=1e-4)

    def _innovation_target(self,x: torch.Tensor) -> torch.Tensor:
        idx=torch.as_tensor(self.dynamics_indices,dtype=torch.long,device=x.device)
        if x.shape[1] <= 1:
            return torch.zeros((x.shape[0],len(self.dynamics_indices)),dtype=x.dtype,device=x.device)
        endpoint=torch.index_select(x[:,-1,:],1,idx)
        previous=torch.index_select(x[:,-2,:],1,idx)
        return endpoint-previous

    def dynamics_prediction(
        self,x: torch.Tensor,mu: torch.Tensor,upper: float | torch.Tensor=1.3
    ) -> tuple[torch.Tensor,torch.Tensor]:
        h=self._encode_dynamics_context(x)
        pred=self.dynamics_prediction_from_context(h,mu,upper)
        return pred,self._innovation_target(x)

    def _boundary_grid(
        self,mu0: torch.Tensor,upper_t: torch.Tensor
    ) -> tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
        """Return offsets, candidates and a valid-hypothesis mask.

        Zero is always the center hypothesis.  At an exact physical boundary the
        impossible side is masked rather than represented by duplicated clipped
        candidates, which keeps posterior entropy and correction neutral.
        """
        half=self.energy_grid_points//2
        neg_span=torch.minimum(mu0,torch.full_like(mu0,self.energy_grid_radius)).clamp_min(0.0)
        pos_span=torch.minimum(upper_t-mu0,torch.full_like(mu0,self.energy_grid_radius)).clamp_min(0.0)
        frac=torch.arange(half,0,-1,dtype=mu0.dtype,device=mu0.device)/float(half)
        neg=-neg_span[:,None]*frac[None,:]
        pos=pos_span[:,None]*torch.flip(frac,dims=[0])[None,:]
        zero=torch.zeros((mu0.shape[0],1),dtype=mu0.dtype,device=mu0.device)
        offsets=torch.cat([neg,zero,pos],dim=1)
        valid_neg=(neg_span[:,None]>1e-8).expand(-1,half)
        valid_pos=(pos_span[:,None]>1e-8).expand(-1,half)
        valid=torch.cat([valid_neg,torch.ones_like(zero,dtype=torch.bool),valid_pos],dim=1)
        candidates=mu0[:,None]+offsets
        return offsets,candidates,valid

    def _energy_landscape(
        self,x: torch.Tensor,base_mu: torch.Tensor,upper: float | torch.Tensor
    ) -> dict[str,torch.Tensor]:
        observed=self._innovation_target(x).detach()
        dyn_h=self._encode_dynamics_context(x)
        dyn_detached=dyn_h.detach()
        upper_t=torch.as_tensor(upper,dtype=base_mu.dtype,device=base_mu.device)
        mu0=base_mu.detach()
        offsets,candidates,valid=self._boundary_grid(mu0,upper_t)
        bsz,k=candidates.shape
        ctx=dyn_detached[:,None,:].expand(-1,k,-1).reshape(bsz*k,-1)
        pred=self.dynamics_prediction_from_context(ctx,candidates.reshape(-1),upper)
        pred=pred.reshape(bsz,k,-1).detach()
        energies=torch.mean((pred-observed[:,None,:]).square(),dim=-1)

        masked_energy=energies.masked_fill(~valid,float("inf"))
        emin=masked_energy.min(dim=-1,keepdim=True).values
        diff=torch.where(valid,energies-emin,torch.zeros_like(energies))
        valid_count=valid.sum(dim=-1,keepdim=True).clamp_min(1)
        spread=(diff.abs().sum(dim=-1,keepdim=True)/valid_count.to(diff.dtype)).clamp_min(self.energy_noise_floor)
        logits=-(energies-emin)/(self.energy_temperature*spread)
        logits=logits.masked_fill(~valid,-1e9)
        weights=torch.softmax(logits,dim=-1)

        uniform=valid.to(weights.dtype)/valid_count.to(weights.dtype)
        posterior_offset=torch.sum((weights-uniform)*offsets,dim=-1)
        correction=torch.clamp(
            posterior_offset,min=-self.inverse_dynamics_max_step,max=self.inverse_dynamics_max_step
        ).detach()
        candidate_mu=torch.clamp(mu0+correction,min=0.0,max=upper_t)

        entropy=-(weights*torch.log(weights.clamp_min(1e-8))*valid.to(weights.dtype)).sum(dim=-1)
        vc=valid_count.squeeze(-1).to(weights.dtype)
        log_vc=torch.log(vc.clamp_min(2.0))
        entropy_norm=torch.where(vc>1.0,entropy/log_vc,torch.ones_like(entropy))
        entropy_ident=torch.clamp(1.0-entropy_norm,0.0,1.0)

        center=k//2
        residual_base=energies[:,center]
        g_candidate=self.dynamics_prediction_from_context(dyn_detached,candidate_mu.detach(),upper).detach()
        residual_candidate=torch.mean((g_candidate-observed).square(),dim=-1)
        normalized_improvement=(residual_base-residual_candidate)/torch.clamp(
            residual_base+residual_candidate,min=self.energy_noise_floor
        )

        best_energy=masked_energy.min(dim=-1).values
        margin=(residual_base-best_energy)/spread.squeeze(-1)
        absolute_strength=torch.sigmoid(
            (margin-self.energy_margin_threshold)/self.energy_margin_temperature
        )
        identifiability=(entropy_ident*absolute_strength).detach()

        if k>=3:
            left=energies[:,center-1]; right=energies[:,center+1]
            both=valid[:,center-1]&valid[:,center+1]
            local_scale=torch.clamp(torch.mean(torch.where(valid,energies,torch.zeros_like(energies)),dim=-1),min=self.energy_noise_floor)
            curvature=torch.where(
                both,torch.relu(left+right-2.0*residual_base)/local_scale,torch.zeros_like(residual_base)
            )
        else:
            curvature=torch.zeros_like(residual_base)

        energy_mean=(torch.where(valid,energies,torch.zeros_like(energies)).sum(dim=-1)/vc)
        separation=(energy_mean-best_energy)/torch.clamp(energy_mean+best_energy,min=self.energy_noise_floor)
        separation=torch.clamp(separation,min=0.0,max=1.0)
        return {
            "context":dyn_h,
            "candidate_grid":candidates.detach(),
            "valid_grid":valid.detach(),
            "energy_grid":energies.detach(),
            "energy_weights":weights.detach(),
            "candidate_mu":candidate_mu.detach(),
            "correction":correction,
            "identifiability":identifiability,
            "posterior_entropy":entropy_norm.detach(),
            "information_raw":separation.detach(),
            "residual_base":residual_base.detach(),
            "residual_candidate":residual_candidate.detach(),
            "normalized_improvement":normalized_improvement.detach(),
            "energy_curvature":curvature.detach(),
            "energy_margin":margin.detach(),
            "energy_strength":absolute_strength.detach(),
        }

    def counterfactual_dynamics_loss(
        self,x: torch.Tensor,true_mu: torch.Tensor,upper: float | torch.Tensor=1.3,
        temperature: float=0.25,negatives: int=6,
    ) -> tuple[torch.Tensor,dict[str,torch.Tensor]]:
        """Contrastive friction discrimination using innovation prediction.

        Wrong hypotheses are generated only on physically feasible sides.  A
        deterministic fallback mirrors the offset when one side is unavailable,
        avoiding repeated clipped negatives at 0 or ``mu_upper``.
        """
        context=self._encode_dynamics_context(x)
        target=self._innovation_target(x)
        upper_t=torch.as_tensor(upper,dtype=true_mu.dtype,device=true_mu.device)
        n=max(2,int(negatives)); half=max(1,n//2)
        mags=torch.linspace(
            self.counterfactual_delta,
            max(self.counterfactual_delta,self.energy_grid_radius*1.5),
            half,dtype=true_mu.dtype,device=true_mu.device,
        )
        raw_offsets=torch.cat([-torch.flip(mags,dims=[0]),mags],dim=0)
        if raw_offsets.numel()>n: raw_offsets=raw_offsets[:n]
        proposed=true_mu[:,None]+raw_offsets[None,:]
        reflected=true_mu[:,None]-raw_offsets[None,:]
        feasible=(proposed>=0.0)&(proposed<=upper_t)
        wrong=torch.where(feasible,proposed,reflected)
        wrong=torch.clamp(wrong,min=0.0,max=upper_t)
        # If the sample sits exactly at a boundary, reflection can still create
        # duplicates. A tiny deterministic rank offset keeps negatives distinct.
        jitter=torch.linspace(-1e-4,1e-4,wrong.shape[1],dtype=wrong.dtype,device=wrong.device)
        wrong=torch.clamp(wrong+jitter[None,:],min=0.0,max=upper_t)
        hypotheses=torch.cat([true_mu[:,None],wrong],dim=-1)
        bsz,k=hypotheses.shape
        ctx=context[:,None,:].expand(-1,k,-1).reshape(bsz*k,-1)
        pred=self.dynamics_prediction_from_context(ctx,hypotheses.reshape(-1),upper).reshape(bsz,k,-1)
        energy=torch.mean((pred-target[:,None,:]).square(),dim=-1)
        scale=torch.mean(torch.abs(energy-energy[:,:1]),dim=-1,keepdim=True).detach().clamp_min(self.energy_noise_floor)
        logits=-(energy-energy.min(dim=-1,keepdim=True).values)/(max(float(temperature),1e-3)*scale)
        labels=torch.zeros(bsz,dtype=torch.long,device=x.device)
        loss=nn.functional.cross_entropy(logits,labels)
        positive=energy[:,0]; best_negative=energy[:,1:].min(dim=-1).values
        return loss,{
            "positive_energy":positive.detach(),
            "best_negative_energy":best_negative.detach(),
            "ranking_accuracy":(positive<best_negative).to(x.dtype).mean().detach(),
        }

    def forward_details(
        self,x: torch.Tensor,lower: torch.Tensor | None=None,
        upper: float | torch.Tensor=1.3,excitation: torch.Tensor | None=None,
        prior_mu: torch.Tensor | None=None,prior_mask: torch.Tensor | None=None,
    ) -> dict[str,torch.Tensor]:
        # Reuse v1.3 feature construction, then replace its product gate by the
        # single continuous correction controller.  Recomputing the few final
        # fields here avoids duplicating the entire exporter-compatible dict.
        d=super().forward_details(x,lower,upper,excitation,prior_mu,prior_mask)
        base=d["base_prediction"]
        corr=d["physics_correction"] if self.use_physics_correction else torch.zeros_like(base)
        if not self.use_physics_correction:
            gate=torch.zeros_like(base)
        elif not self.use_utility_gate:
            gate=torch.ones_like(base)
        elif self.use_magnitude_head:
            gate=d["correction_fraction"]
        else:
            gate=d["benefit_probability"]
        applied=gate*self.physics_correction_scale*corr
        raw=base+applied
        prediction=self._project(raw,lower,upper)
        candidate_raw=base+self.physics_correction_scale*corr
        candidate=self._project(candidate_raw,lower,upper)

        dyn_h=self._encode_dynamics_context(x).detach()
        observed=self._innovation_target(x).detach()
        g_final=self.dynamics_prediction_from_context(dyn_h,prediction.detach(),upper).detach()
        residual_final=torch.mean((observed-g_final).square(),dim=-1)

        # Rebuild latent/UQ features from the *v1.4* final prediction and the
        # actual continuous controller.  The inherited v1.3 dictionary was
        # produced with benefit_probability*correction_fraction, so keeping its
        # latent/UQ fields would make calibration describe a different point
        # model than the one returned here.
        upper_t=torch.as_tensor(upper,dtype=prediction.dtype,device=prediction.device)
        frac=torch.clamp(prediction/torch.clamp(upper_t,min=1e-6),1e-5,1.0-1e-5)
        latent=self._logit(frac)
        h=self.encode(x)
        upper_scalar=max(float(torch.as_tensor(upper).detach().cpu()),1e-6)
        scale_feature=(
            torch.clamp(d["aleatoric_scale"]/upper_scalar,0.0,2.0)
            if self.use_aleatoric_feature else torch.zeros_like(gate)
        )
        controller_confidence=torch.clamp(2.0*torch.abs(gate-0.5),0.0,1.0)
        uq_scalars=torch.stack([
            d["identifiability"],d["benefit_probability"],
            d["correction_fraction"],gate,controller_confidence,
            torch.clamp(scale_feature,0.0,1.0),
        ],dim=-1)
        uq_h=self.uq_fuse(torch.cat([h,uq_scalars],dim=-1))
        gate_logit=(
            d["magnitude_logit"] if self.use_magnitude_head and self.use_utility_gate
            else d["benefit_logit"]
        )

        d.update({
            "prediction":prediction,
            "raw_prediction":raw,
            "candidate_prediction":candidate,
            "inverse_candidate_prediction":candidate,
            "inverse_candidate_raw":candidate_raw,
            "latent":latent,
            "features":uq_h,
            "effective_innovation":applied,
            "fused_delta_mu":applied,
            "reliability":gate,
            "authority":gate,
            "physics_gate":gate,
            "gate_logit":gate_logit,
            "gate_confidence":controller_confidence,
            "acceptance":d["benefit_probability"],
            "veto_probability":1.0-d["benefit_probability"],
            "expert_weights":torch.stack([torch.zeros_like(gate),1.0-gate,gate],dim=-1),
            "expert_weight_prior":torch.zeros_like(gate),
            "expert_weight_neural":1.0-gate,
            "expert_weight_inverse":gate,
            "direction_agreement":torch.tanh(8.0*corr*applied),
            "magnitude_agreement":gate,
            "applied_physics_correction":applied,
            "dynamics_residual_prior":d["dynamics_residual_prior"],
            "dynamics_residual_candidate":d["dynamics_residual_candidate"],
            "dynamics_residual_final":residual_final,
            "identifiability":d["identifiability"],
            "posterior_entropy":d["posterior_entropy"],
            "information_raw":d["information_raw"],
            "normalized_improvement":d["normalized_improvement"],
            "energy_curvature":d["energy_curvature"],
        })
        return d

# Legacy public names remain pinned to v1.1 so old experiments/tests are reproducible.
SafeGripV3Net = SafeGripCI11Net
SafeGripV2Net = SafeGripCI11Net
# v1.2 public alias retained for reproducibility.
SafeGripV4Net = SafeGripCI12Net
# v1.3 alias retained for reproducibility.
SafeGripV5Net = SafeGripCI13Net
# Current v1.4 proposal alias.
SafeGripV6Net = SafeGripCI14Net


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


class FrictionResponseNet(nn.Module):
    """Causal response model used by SafeGrip-FRC.

    The network is intentionally ordinary.  Novelty is not claimed for this
    architecture: a GRU encodes only samples *before* the response target and a
    small MLP predicts the standardized response innovation under a hypothetical
    friction coefficient.
    """

    def __init__(self, d: int, response_dim: int, hidden: int = 128,
                 gru_layers: int = 1, dropout: float = 0.1, mu_upper: float = 1.3):
        super().__init__()
        self.mu_upper = float(mu_upper)
        recurrent_dropout = float(dropout) if int(gru_layers) > 1 else 0.0
        self.encoder = nn.GRU(
            d, int(hidden), num_layers=int(gru_layers), batch_first=True,
            dropout=recurrent_dropout,
        )
        self.head = nn.Sequential(
            nn.Linear(int(hidden) + 1, int(hidden)),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden), int(response_dim)),
        )

    def forward(self, context: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
        if context.ndim != 3:
            raise ValueError("context must have shape [B,C,D]")
        mu = mu.reshape(-1, 1).to(dtype=context.dtype, device=context.device)
        if len(mu) != len(context):
            raise ValueError("mu batch must match context batch")
        z, _ = self.encoder(context)
        mu_scaled = mu / max(self.mu_upper, 1e-6)
        return self.head(torch.cat([z[:, -1], mu_scaled], dim=-1))


class DirectGRUControl(nn.Module):
    """Same-family direct-regression control for the FRC comparison.

    It uses the same GRU depth/hidden width as ``FrictionResponseNet`` but maps
    the encoded observation window directly to friction.  This isolates the
    contribution of response inversion/certification from recurrent capacity.
    """

    def __init__(self, d: int, hidden: int = 128, gru_layers: int = 1,
                 dropout: float = 0.1):
        super().__init__()
        recurrent_dropout = float(dropout) if int(gru_layers) > 1 else 0.0
        self.encoder = nn.GRU(
            d, int(hidden), num_layers=int(gru_layers), batch_first=True,
            dropout=recurrent_dropout,
        )
        self.head = nn.Sequential(
            nn.Linear(int(hidden), int(hidden)),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden), 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z, _ = self.encoder(x)
        return self.head(z[:, -1]).squeeze(-1)


class PFRExcitationGRU(nn.Module):
    """Excitation-aware recurrent encoder for the revised PFR proposal.

    The network keeps the proposal deliberately small: one GRU is followed by
    deterministic physics-guided temporal pooling and two scalar heads.  The
    residual head predicts the standardized friction residual, while the scale
    head predicts a positive physical-unit error scale used only for the
    one-sided conformal safety correction.

    ``excitation_index`` must point to an input channel already expressed on
    ``[0, 1]``.  ``attention_gamma=0`` reduces the pooling to a uniform average
    and is used as the decisive no-attention ablation.
    """

    def __init__(
        self,
        d: int,
        excitation_index: int,
        hidden: int = 128,
        gru_layers: int = 1,
        dropout: float = 0.1,
        attention_gamma: float = 4.0,
        scale_floor: float = 0.005,
    ):
        super().__init__()
        if not 0 <= int(excitation_index) < int(d):
            raise ValueError("excitation_index must refer to an input channel")
        self.excitation_index = int(excitation_index)
        self.attention_gamma = float(attention_gamma)
        self.scale_floor = max(float(scale_floor), 1e-6)
        recurrent_dropout = float(dropout) if int(gru_layers) > 1 else 0.0
        self.encoder = nn.GRU(
            int(d), int(hidden), num_layers=int(gru_layers), batch_first=True,
            dropout=recurrent_dropout,
        )
        inner = max(32, int(hidden) // 2)
        self.residual_head = nn.Sequential(
            nn.Linear(int(hidden), int(hidden)),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden), 1),
        )
        self.scale_head = nn.Sequential(
            nn.Linear(int(hidden), inner),
            nn.SiLU(),
            nn.Linear(inner, 1),
        )

    def attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("x must have shape [B,T,D]")
        excitation = torch.clamp(x[:, :, self.excitation_index], 0.0, 1.0)
        if abs(self.attention_gamma) < 1e-12:
            return torch.full_like(excitation, 1.0 / max(excitation.shape[1], 1))
        return torch.softmax(self.attention_gamma * excitation, dim=1)

    def forward(self, x: torch.Tensor, *, return_attention: bool = False):
        z, _ = self.encoder(x)
        weights = self.attention_weights(x)
        pooled = torch.sum(z * weights.unsqueeze(-1), dim=1)
        residual_z = self.residual_head(pooled).squeeze(-1)
        scale = nn.functional.softplus(self.scale_head(pooled).squeeze(-1)) + self.scale_floor
        if return_attention:
            return residual_z, scale, weights
        return residual_z, scale
