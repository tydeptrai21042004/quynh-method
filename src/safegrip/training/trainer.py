from __future__ import annotations

from dataclasses import dataclass, replace
import torch
import torch.nn.functional as F

from safegrip.model.safegrip_universal import UniversalSafeGrip
from safegrip.universal.innovation import SemanticInnovationNormalizer
from safegrip.universal.dataset import UniversalResearchBatch
from safegrip.training.dropout import sensor_channel_dropout_mask
from safegrip.training.objectives import masked_gaussian_nll, utilization_consistency_loss, friction_inequality_loss


@dataclass(frozen=True)
class ResearchLossConfig:
    huber_beta: float = 0.5
    scale_nll_weight: float = 0.1
    utilization_consistency_weight: float = 0.1
    friction_inequality_weight: float = 0.1
    sensor_dropout: float = 0.2


@dataclass
class ResearchLosses:
    total: torch.Tensor
    task: torch.Tensor
    scale_nll: torch.Tensor
    utilization_consistency: torch.Tensor
    friction_inequality: torch.Tensor


def estimate_target_scales(batches, floor: float = 1e-3) -> torch.Tensor:
    if not batches:
        raise ValueError("batches cannot be empty")
    q = batches[0].target_values.shape[1]
    scales = []
    for j in range(q):
        vals = []
        for b in batches:
            valid = b.target_mask[:, j]
            if torch.any(valid):
                vals.append(b.target_values[valid, j].detach().cpu())
        scales.append(float(torch.cat(vals).std(unbiased=False).clamp_min(floor)) if vals else 1.0)
    return torch.tensor(scales, dtype=torch.float32)



def estimate_innovation_normalizer(batches, floor: float = 1e-3) -> SemanticInnovationNormalizer:
    """Fit training-only NPI residual/direct statistics.

    The returned semantic buffers contain no trainable parameters and never use
    dataset identity.
    """
    return SemanticInnovationNormalizer(floor=floor).fit(batches)

def _scaled_huber(pred, target, mask, scales, beta):
    scaled = (pred - target) / scales.to(pred).unsqueeze(0).clamp_min(1e-8)
    valid = mask.bool() & torch.isfinite(scaled)
    if not torch.any(valid):
        return pred.sum() * 0.0
    return F.smooth_l1_loss(scaled[valid], torch.zeros_like(scaled[valid]), beta=beta, reduction="mean")


def research_losses(model, batch, target_scales, config: ResearchLossConfig | None = None, *, training: bool = True) -> ResearchLosses:
    cfg = config or ResearchLossConfig()
    sensor_batch = batch.sensors
    if training and cfg.sensor_dropout > 0:
        dropped = sensor_channel_dropout_mask(sensor_batch.token_mask, sensor_batch.channel_ids, cfg.sensor_dropout)
        sensor_batch = replace(sensor_batch, token_mask=dropped)
    out = model(sensor_batch, batch.queries, domains=batch.domains)

    # Optimize the dimensionless semantic innovation directly.  Critically,
    # the target coordinate is recomputed after sensor dropout using the same
    # reference mask as the forward pass. Thus a missing anchor switches to the
    # direct-target normalization instead of asking one decoder output to mix a
    # small residual with a full absolute state.
    qids = model.query_tensor(batch.queries, batch.target_values.shape[0], device=out.point.device)
    normalized_target = model.innovation_normalizer.normalize_target(
        batch.target_values, qids, out.physical_anchor, out.anchor_mask
    )
    valid = batch.target_mask.bool() & torch.isfinite(normalized_target) & torch.isfinite(out.innovation)
    if torch.any(valid):
        task = F.smooth_l1_loss(
            out.innovation[valid], normalized_target[valid], beta=cfg.huber_beta, reduction="mean"
        )
    else:
        task = out.innovation.sum() * 0.0
    scale_nll = masked_gaussian_nll(
        out.innovation, out.normalized_scale, normalized_target, batch.target_mask
    )

    names = [q.name or q.quantity for q in batch.queries]
    zero = out.point.sum() * 0.0
    util_consistency = zero
    inequality = zero
    required = {"force_x", "force_y", "force_z", "utilization"}
    if cfg.utilization_consistency_weight > 0 and required.issubset(names):
        fx, fy, fz, u = [out.point[:, names.index(n)] for n in ("force_x", "force_y", "force_z", "utilization")]
        util_consistency = utilization_consistency_loss(u, fx, fy, fz)
    if cfg.friction_inequality_weight > 0 and "friction" in names and "utilization" in names:
        mu = out.point[:, names.index("friction")]
        u = out.point[:, names.index("utilization")]
        mu_mask = batch.target_mask[:, names.index("friction")]
        inequality = friction_inequality_loss(mu, u, mu_mask)

    scale_weight = cfg.scale_nll_weight if model.ablation.learned_scale else 0.0
    total = (
        task + scale_weight * scale_nll
        + cfg.utilization_consistency_weight * util_consistency
        + cfg.friction_inequality_weight * inequality
    )
    return ResearchLosses(total, task, scale_nll, util_consistency, inequality)


def train_step(model, batch, optimizer, target_scales, config: ResearchLossConfig | None = None) -> ResearchLosses:
    model.train(); optimizer.zero_grad(set_to_none=True)
    losses = research_losses(model, batch, target_scales, config, training=True)
    losses.total.backward(); optimizer.step(); return losses
