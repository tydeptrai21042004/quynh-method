from __future__ import annotations

from dataclasses import dataclass, replace
import torch
import torch.nn.functional as F

from safegrip.model.safegrip_universal import UniversalSafeGrip
from safegrip.universal.dataset import UniversalResearchBatch
from safegrip.training.dropout import sensor_channel_dropout_mask
from safegrip.training.objectives import utilization_consistency_loss, friction_inequality_loss


@dataclass(frozen=True)
class ResearchLossConfig:
    huber_beta: float = 0.5
    utilization_consistency_weight: float = 0.1
    friction_inequality_weight: float = 0.1
    sensor_dropout: float = 0.2


@dataclass
class ResearchLosses:
    total: torch.Tensor
    task: torch.Tensor
    utilization_consistency: torch.Tensor
    friction_inequality: torch.Tensor


def estimate_target_scales(batches: list[UniversalResearchBatch] | tuple[UniversalResearchBatch, ...], floor: float = 1e-3) -> torch.Tensor:
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
        if vals:
            v = torch.cat(vals)
            scales.append(float(v.std(unbiased=False).clamp_min(floor)))
        else:
            scales.append(1.0)
    return torch.tensor(scales, dtype=torch.float32)


def _scaled_huber(pred, target, mask, scales, beta):
    scaled = (pred - target) / scales.to(pred).unsqueeze(0).clamp_min(1e-8)
    valid = mask.bool() & torch.isfinite(scaled)
    if not torch.any(valid):
        return pred.sum() * 0.0
    return F.smooth_l1_loss(scaled[valid], torch.zeros_like(scaled[valid]), beta=beta, reduction="mean")


def research_losses(
    model: UniversalSafeGrip,
    batch: UniversalResearchBatch,
    target_scales: torch.Tensor,
    config: ResearchLossConfig | None = None,
    *,
    training: bool = True,
) -> ResearchLosses:
    cfg = config or ResearchLossConfig()
    sensor_batch = batch.sensors
    if training and cfg.sensor_dropout > 0:
        dropped = sensor_channel_dropout_mask(sensor_batch.token_mask, sensor_batch.channel_ids, cfg.sensor_dropout)
        sensor_batch = replace(sensor_batch, token_mask=dropped)
    out = model(sensor_batch, batch.queries)
    task = _scaled_huber(out.point, batch.target_values, batch.target_mask, target_scales, cfg.huber_beta)

    names = [q.name or q.quantity for q in batch.queries]
    util_consistency = out.point.sum() * 0.0
    inequality = out.point.sum() * 0.0
    required = {"force_x", "force_y", "force_z", "utilization"}
    if required.issubset(names):
        fx, fy, fz, u = [out.point[:, names.index(n)] for n in ("force_x", "force_y", "force_z", "utilization")]
        # Consistency is a model-side regularizer and therefore does not require
        # all four labels to be observed on every sample.
        util_consistency = utilization_consistency_loss(u, fx, fy, fz)
    if "friction" in names and "utilization" in names:
        mu = out.point[:, names.index("friction")]
        u = out.point[:, names.index("utilization")]
        # Apply the inequality only where direct friction supervision exists;
        # deployment-specific assumptions can tighten this mask further.
        mu_mask = batch.target_mask[:, names.index("friction")]
        inequality = friction_inequality_loss(mu, u, mu_mask)

    total = task + cfg.utilization_consistency_weight * util_consistency + cfg.friction_inequality_weight * inequality
    return ResearchLosses(total, task, util_consistency, inequality)


def train_step(model, batch, optimizer, target_scales, config: ResearchLossConfig | None = None) -> ResearchLosses:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    losses = research_losses(model, batch, target_scales, config, training=True)
    losses.total.backward()
    optimizer.step()
    return losses
