from __future__ import annotations

"""Runnable, paper-structured reproductions for the D2--D4 comparators.

These implementations intentionally distinguish *structural reproduction* from
an exact source-code reproduction.  They reproduce the documented model family,
causal input/output contract and key mathematical mechanism while avoiding any
claim that unpublished MATLAB fuzzy rule parameters or upstream learned weights
are identical to the original papers.

All neural baselines consume ``x`` shaped ``[batch, time, features]``.
"""

from dataclasses import dataclass
import math
from typing import Callable

import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class BaselineBuild:
    model: nn.Module
    target_names: tuple[str, ...]
    implementation_kind: str = "paper_structured_local_reproduction"


class TemporalSummary(nn.Module):
    """Differentiable strain/sequence summary approximating paper feature picking."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("expected [batch,time,features]")
        mean = x.mean(dim=1)
        std = x.std(dim=1, unbiased=False)
        xmax = x.max(dim=1).values
        xmin = x.min(dim=1).values
        first = x[:, 0]
        last = x[:, -1]
        peak = torch.maximum(xmax.abs(), xmin.abs())
        return torch.cat([mean, std, xmax, xmin, last - first, peak], dim=-1)


class TSKFuzzyBlock(nn.Module):
    """Trainable zero/first-order Takagi--Sugeno fuzzy block.

    Gaussian memberships provide smooth fuzzification.  Rule firing strengths
    are normalized and combined with affine consequents.  The paper-specific
    hierarchical wiring is handled by the estimator classes below.
    """

    def __init__(self, in_dim: int, rules: int, out_dim: int = 1):
        super().__init__()
        if in_dim < 1 or rules < 1 or out_dim < 1:
            raise ValueError("invalid fuzzy block dimensions")
        self.in_dim = int(in_dim)
        self.rules = int(rules)
        self.out_dim = int(out_dim)
        self.centers = nn.Parameter(torch.empty(rules, in_dim))
        self.log_widths = nn.Parameter(torch.zeros(rules, in_dim))
        self.consequents = nn.Parameter(torch.empty(rules, out_dim, in_dim + 1))
        nn.init.uniform_(self.centers, -1.0, 1.0)
        nn.init.xavier_uniform_(self.consequents)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        width = F.softplus(self.log_widths) + 1e-3
        z = (x.unsqueeze(1) - self.centers.unsqueeze(0)) / width.unsqueeze(0)
        # log-domain product of Gaussian memberships for numerical stability.
        log_fire = -0.5 * z.square().sum(dim=-1)
        fire = torch.softmax(log_fire, dim=1)
        xb = torch.cat([x, torch.ones_like(x[:, :1])], dim=-1)
        rule_y = torch.einsum("bi,roi->bro", xb, self.consequents)
        return torch.einsum("br,bro->bo", fire, rule_y)


class Mendoza2019Fuzzy(nn.Module):
    """Hierarchical fuzzy tire estimator matching Mendoza-Petit et al. (2019).

    Published order: slip angle -> vertical load -> lateral force ->
    longitudinal force.  The default rule counts match the paper (21, 217,
    460, 145).  Membership centers/consequents are learned on the benchmark
    training split because the paper's exact MATLAB FIS parameter file is not
    public in this repository.
    """

    target_names = ("force_x", "force_y", "force_z", "slip_angle")

    def __init__(self, input_dim: int, *, debug_scale: bool = False):
        super().__init__()
        self.summary = TemporalSummary()
        sdim = 6 * int(input_dim)
        latent = 12 if debug_scale else 24
        self.feature_map = nn.Sequential(nn.LayerNorm(sdim), nn.Linear(sdim, latent), nn.Tanh())
        counts = (7, 15, 21, 15) if debug_scale else (21, 217, 460, 145)
        self.slip = TSKFuzzyBlock(latent, counts[0])
        self.vertical = TSKFuzzyBlock(latent + 1, counts[1])
        self.lateral = TSKFuzzyBlock(latent + 1, counts[2])
        self.longitudinal = TSKFuzzyBlock(latent + 1, counts[3])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.feature_map(self.summary(x))
        slip = self.slip(h)
        fz = self.vertical(torch.cat([h, slip], dim=-1))
        fy = self.lateral(torch.cat([h, slip], dim=-1))
        fx = self.longitudinal(torch.cat([h, fz], dim=-1))
        return torch.cat([fx, fy, fz, slip], dim=-1)


class Yunta2018FuzzyLFC(nn.Module):
    """Fuzzy slip/load/LFC reproduction of Yunta et al. (2018).

    The paper estimates slip angle and vertical load from strain features, then
    estimates lateral-friction demand / grip loss.  For the D2 task interface,
    ``force_y`` is represented as predicted lateral-friction demand times the
    magnitude of predicted vertical load.  The model intentionally does *not*
    fabricate a longitudinal-force output; it is evaluated on the compatible
    targets listed in ``target_names``.
    """

    target_names = ("force_y", "force_z", "slip_angle")

    def __init__(self, input_dim: int, *, debug_scale: bool = False):
        super().__init__()
        self.summary = TemporalSummary()
        sdim = 6 * int(input_dim)
        latent = 10 if debug_scale else 20
        self.feature_map = nn.Sequential(nn.LayerNorm(sdim), nn.Linear(sdim, latent), nn.Tanh())
        rules = 9 if debug_scale else 27
        self.slip = TSKFuzzyBlock(latent, rules)
        self.vertical = TSKFuzzyBlock(latent + 1, rules * 2)
        self.lfc = TSKFuzzyBlock(latent + 2, rules * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.feature_map(self.summary(x))
        slip = self.slip(h)
        fz = self.vertical(torch.cat([h, slip], dim=-1))
        mu_lat = torch.tanh(self.lfc(torch.cat([h, slip, fz], dim=-1)))
        fy = mu_lat * (fz.abs() + 1e-6)
        return torch.cat([fy, fz, slip], dim=-1)


@dataclass(frozen=True)
class VehicleSpec:
    mass: float = 720.0
    lf: float = 1.65
    lr: float = 1.55
    dt: float = 0.04


class PhysicsGuard(nn.Module):
    """Sigmoid min/max parameter guard used by Deep Dynamics."""

    def __init__(self, bounds: list[tuple[float, float]]):
        super().__init__()
        lo = torch.tensor([b[0] for b in bounds], dtype=torch.float32)
        hi = torch.tensor([b[1] for b in bounds], dtype=torch.float32)
        self.register_buffer("lo", lo)
        self.register_buffer("span", hi - lo)

    def forward(self, raw: torch.Tensor) -> torch.Tensor:
        return self.lo + torch.sigmoid(raw) * self.span


class DeepDynamics2024(nn.Module):
    """Local PCNN reproduction of Chrosniak et al. Deep Dynamics.

    Input convention is [vx, vy, yaw_rate, steering, drive] in the first five
    channels.  A historical encoder estimates bounded Pacejka/drivetrain/inertia
    parameters; a differentiable single-track model predicts the next state.
    """

    target_names = ("velocity_x", "velocity_y", "yaw_rate")

    def __init__(self, input_dim: int, *, hidden: int = 96, debug_scale: bool = False, spec: VehicleSpec | None = None):
        super().__init__()
        if input_dim < 3:
            raise ValueError("DeepDynamics2024 requires at least vx, vy, yaw_rate")
        hidden = 24 if debug_scale else hidden
        self.input_dim = int(input_dim)
        self.spec = spec or VehicleSpec()
        self.encoder = nn.GRU(input_dim, hidden, num_layers=1, batch_first=True)
        # Bf,Cf,Df,Ef, Br,Cr,Dr,Er, Cm1,Cm2,Cr0,Cr2, Iz
        bounds = [
            (0.1, 25.0), (0.5, 3.0), (200.0, 20000.0), (-1.5, 1.5),
            (0.1, 25.0), (0.5, 3.0), (200.0, 20000.0), (-1.5, 1.5),
            (0.0, 12000.0), (0.0, 500.0), (0.0, 1500.0), (0.0, 20.0),
            (300.0, 8000.0),
        ]
        self.param_head = nn.Linear(hidden, len(bounds))
        self.guard = PhysicsGuard(bounds)

    @staticmethod
    def _pacejka(alpha, B, C, D, E):
        ba = B * alpha
        return D * torch.sin(C * torch.atan(ba - E * (ba - torch.atan(ba))))

    def _physics(self, last: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        vx, vy, r = last[:, 0], last[:, 1], last[:, 2]
        steer = last[:, 3] if last.shape[1] > 3 else torch.zeros_like(vx)
        drive = last[:, 4] if last.shape[1] > 4 else torch.zeros_like(vx)
        Bf,Cf,Df,Ef,Br,Cr,Dr,Er,Cm1,Cm2,Cr0,Cr2,Iz = params.unbind(dim=-1)
        s = self.spec
        vx_safe = vx.abs().clamp_min(0.5)
        alpha_f = steer - torch.atan2(s.lf * r + vy, vx_safe)
        alpha_r = torch.atan2(s.lr * r - vy, vx_safe)
        fy_f = self._pacejka(alpha_f, Bf,Cf,Df,Ef)
        fy_r = self._pacejka(alpha_r, Br,Cr,Dr,Er)
        fx = (Cm1 - Cm2 * vx) * drive - Cr0 - Cr2 * vx.square()
        dvx = (fx - fy_f * torch.sin(steer)) / s.mass + vy * r
        dvy = (fy_r + fy_f * torch.cos(steer)) / s.mass - vx * r
        dr = (fy_f * s.lf * torch.cos(steer) - fy_r * s.lr) / Iz.clamp_min(1.0)
        return torch.stack([vx + s.dt * dvx, vy + s.dt * dvy, r + s.dt * dr], dim=-1)

    def forward_with_physics(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 3:
            raise ValueError("expected [batch,time,features]")
        _, h = self.encoder(x)
        params = self.guard(self.param_head(h[-1]))
        physics = self._physics(x[:, -1], params)
        return physics, params

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_with_physics(x)[0]


class FTHD2025(DeepDynamics2024):
    """FTHD local reproduction: DDM physics core + fine-tuned residual correction.

    Training can use :meth:`hybrid_loss`, combining next-state supervision with
    a physics-anchor loss.  This mirrors the paper's supervised + unsupervised /
    differential fine-tuning idea without claiming exact upstream weights.
    """

    def __init__(self, input_dim: int, *, hidden: int = 96, debug_scale: bool = False, spec: VehicleSpec | None = None):
        super().__init__(input_dim, hidden=hidden, debug_scale=debug_scale, spec=spec)
        h = 24 if debug_scale else hidden
        self.residual_encoder = nn.GRU(input_dim, h, num_layers=1, batch_first=True)
        self.residual_head = nn.Sequential(nn.Linear(h, h), nn.SiLU(), nn.Linear(h, 3))
        self.residual_scale = nn.Parameter(torch.tensor(-2.0))

    def forward_components(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        physics, params = self.forward_with_physics(x)
        _, h = self.residual_encoder(x)
        residual = torch.tanh(self.residual_head(h[-1])) * F.softplus(self.residual_scale)
        return physics + residual, physics, params

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_components(x)[0]

    def hybrid_loss(self, x: torch.Tensor, y: torch.Tensor, *, physics_weight: float = 0.25) -> torch.Tensor:
        pred, physics, _ = self.forward_components(x)
        supervised = F.mse_loss(pred, y)
        physics_anchor = F.mse_loss(pred, physics.detach())
        return supervised + float(physics_weight) * physics_anchor


class QuaternionLinear(nn.Module):
    """Hamilton-product linear map for quaternion-packed real tensors."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        if in_features % 4 or out_features % 4:
            raise ValueError("quaternion dimensions must be multiples of four")
        self.in_q, self.out_q = in_features // 4, out_features // 4
        self.wr = nn.Parameter(torch.empty(self.out_q, self.in_q))
        self.wi = nn.Parameter(torch.empty(self.out_q, self.in_q))
        self.wj = nn.Parameter(torch.empty(self.out_q, self.in_q))
        self.wk = nn.Parameter(torch.empty(self.out_q, self.in_q))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        for w in (self.wr, self.wi, self.wj, self.wk):
            nn.init.xavier_uniform_(w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xr, xi, xj, xk = x.reshape(*x.shape[:-1], self.in_q, 4).unbind(dim=-1)
        lin: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = lambda a, w: F.linear(a, w)
        yr = lin(xr,self.wr) - lin(xi,self.wi) - lin(xj,self.wj) - lin(xk,self.wk)
        yi = lin(xr,self.wi) + lin(xi,self.wr) + lin(xj,self.wk) - lin(xk,self.wj)
        yj = lin(xr,self.wj) - lin(xi,self.wk) + lin(xj,self.wr) + lin(xk,self.wi)
        yk = lin(xr,self.wk) + lin(xi,self.wj) - lin(xj,self.wi) + lin(xk,self.wr)
        y = torch.stack([yr, yi, yj, yk], dim=-1).reshape(*x.shape[:-1], self.out_q * 4)
        return y if self.bias is None else y + self.bias


class QuaternionGRUCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.x_z, self.h_z = QuaternionLinear(input_size, hidden_size), QuaternionLinear(hidden_size, hidden_size, False)
        self.x_r, self.h_r = QuaternionLinear(input_size, hidden_size), QuaternionLinear(hidden_size, hidden_size, False)
        self.x_n, self.h_n = QuaternionLinear(input_size, hidden_size), QuaternionLinear(hidden_size, hidden_size, False)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        z = torch.sigmoid(self.x_z(x) + self.h_z(h))
        r = torch.sigmoid(self.x_r(x) + self.h_r(h))
        n = torch.tanh(self.x_n(x) + self.h_n(r * h))
        return (1.0 - z) * n + z * h


class Onyekpe2021QGRU(nn.Module):
    """Quaternion-GRU sensor-fusion baseline for IO-VNBD."""

    target_names = ("displacement", "orientation")

    def __init__(self, input_dim: int, *, hidden_size: int = 64, debug_scale: bool = False):
        super().__init__()
        qin = int(math.ceil(input_dim / 4.0) * 4)
        qhidden = int(math.ceil((16 if debug_scale else hidden_size) / 4.0) * 4)
        self.input_dim = int(input_dim)
        self.qin = qin
        self.hidden_size = qhidden
        self.cell = QuaternionGRUCell(qin, qhidden)
        self.head = nn.Linear(qhidden, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] < self.qin:
            x = F.pad(x, (0, self.qin - x.shape[-1]))
        h = torch.zeros(x.shape[0], self.hidden_size, dtype=x.dtype, device=x.device)
        for t in range(x.shape[1]):
            h = self.cell(x[:, t], h)
        return self.head(h)


class SinusoidalPosition(nn.Module):
    def __init__(self, dim: int, max_len: int = 4096):
        super().__init__()
        p = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim))
        pe = torch.zeros(max_len, dim)
        pe[:, 0::2] = torch.sin(p * div)
        pe[:, 1::2] = torch.cos(p * div[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[: x.shape[1]].unsqueeze(0).to(x)


class Wang2023Transformer(nn.Module):
    """Transformer wheel-odometry error predictor for IO-VNBD."""

    target_names = ("displacement", "orientation")

    def __init__(self, input_dim: int, *, d_model: int = 96, heads: int = 4, layers: int = 3, debug_scale: bool = False):
        super().__init__()
        if debug_scale:
            d_model, heads, layers = 32, 4, 1
        self.proj = nn.Linear(input_dim, d_model)
        self.pos = SinusoidalPosition(d_model)
        enc = nn.TransformerEncoderLayer(d_model, heads, dim_feedforward=4*d_model, dropout=0.1, activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc, layers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Linear(d_model // 2, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(self.pos(self.proj(x)))
        return self.head(z[:, -1])


BASELINE_TARGETS: dict[str, tuple[str, ...]] = {
    "mendoza2019_fuzzy": Mendoza2019Fuzzy.target_names,
    "yunta2018_fuzzy_lfc": Yunta2018FuzzyLFC.target_names,
    "chrosniak2024_ddm": DeepDynamics2024.target_names,
    "fang_yu2025_fthd": FTHD2025.target_names,
    "onyekpe2021_qgru": Onyekpe2021QGRU.target_names,
    "wang2023_transformer": Wang2023Transformer.target_names,
}


def make_paper_baseline(name: str, input_dim: int, *, debug_scale: bool = False) -> BaselineBuild:
    key = str(name).strip().lower()
    builders = {
        "mendoza2019_fuzzy": lambda: Mendoza2019Fuzzy(input_dim, debug_scale=debug_scale),
        "yunta2018_fuzzy_lfc": lambda: Yunta2018FuzzyLFC(input_dim, debug_scale=debug_scale),
        "chrosniak2024_ddm": lambda: DeepDynamics2024(input_dim, debug_scale=debug_scale),
        "fang_yu2025_fthd": lambda: FTHD2025(input_dim, debug_scale=debug_scale),
        "onyekpe2021_qgru": lambda: Onyekpe2021QGRU(input_dim, debug_scale=debug_scale),
        "wang2023_transformer": lambda: Wang2023Transformer(input_dim, debug_scale=debug_scale),
    }
    if key not in builders:
        raise ValueError(f"No D2--D4 paper-baseline implementation for {name}")
    return BaselineBuild(builders[key](), BASELINE_TARGETS[key])
