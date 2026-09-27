from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn.functional as F


@dataclass
class LossBreakdown:
    total: torch.Tensor
    task: torch.Tensor
    utilization_consistency: torch.Tensor
    friction_inequality: torch.Tensor


def masked_huber(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
    if pred.shape != target.shape or pred.shape != mask.shape:
        raise ValueError("pred, target and mask must have identical shapes")
    valid = mask.bool() & torch.isfinite(target) & torch.isfinite(pred)
    if not torch.any(valid):
        return pred.sum() * 0.0
    return F.smooth_l1_loss(pred[valid], target[valid], beta=beta, reduction="mean")


def masked_gaussian_nll(
    point: torch.Tensor,
    scale: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    scale_floor: float = 1e-6,
) -> torch.Tensor:
    """Heteroscedastic Gaussian NLL so the learned scale head receives gradients."""
    if point.shape != target.shape or point.shape != scale.shape or point.shape != mask.shape:
        raise ValueError("point, scale, target and mask must have identical shapes")
    sigma = scale.clamp_min(float(scale_floor))
    valid = mask.bool() & torch.isfinite(point) & torch.isfinite(sigma) & torch.isfinite(target)
    if not torch.any(valid):
        return point.sum() * 0.0
    residual = (target[valid] - point[valid]) / sigma[valid]
    return (0.5 * residual.square() + torch.log(sigma[valid])).mean()


def force_utilization(fx: torch.Tensor, fy: torch.Tensor, fz: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """u = sqrt(Fx^2 + Fy^2) / max(|Fz|, eps)."""
    tangential = torch.sqrt(torch.square(fx) + torch.square(fy))
    normal = torch.clamp(torch.abs(fz), min=float(eps))
    return tangential / normal


def utilization_consistency_loss(u_pred, fx_pred, fy_pred, fz_pred, mask: torch.Tensor | None = None) -> torch.Tensor:
    implied = force_utilization(fx_pred, fy_pred, fz_pred)
    err = torch.abs(u_pred - implied)
    if mask is not None:
        valid = mask.bool() & torch.isfinite(err)
        if not torch.any(valid):
            return err.sum() * 0.0
        err = err[valid]
    return err.mean()


def friction_inequality_loss(mu_pred: torch.Tensor, utilization: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    violation = torch.relu(utilization - mu_pred)
    loss = torch.square(violation)
    if mask is not None:
        valid = mask.bool() & torch.isfinite(loss)
        if not torch.any(valid):
            return loss.sum() * 0.0
        loss = loss[valid]
    return loss.mean()


class TargetNormalizer:
    def __init__(self, eps: float = 1e-6):
        self.eps = float(eps)
        self.mean: torch.Tensor | None = None
        self.std: torch.Tensor | None = None

    def fit(self, values: torch.Tensor, mask: torch.Tensor) -> "TargetNormalizer":
        if values.shape != mask.shape:
            raise ValueError("values and mask must have identical shapes")
        means, stds = [], []
        for j in range(values.shape[1]):
            valid = mask[:, j].bool() & torch.isfinite(values[:, j])
            if torch.any(valid):
                v = values[valid, j]
                means.append(v.mean())
                stds.append(v.std(unbiased=False).clamp_min(self.eps))
            else:
                means.append(torch.tensor(0.0, device=values.device, dtype=values.dtype))
                stds.append(torch.tensor(1.0, device=values.device, dtype=values.dtype))
        self.mean = torch.stack(means)
        self.std = torch.stack(stds)
        return self

    def transform(self, values: torch.Tensor) -> torch.Tensor:
        if self.mean is None or self.std is None:
            raise RuntimeError("TargetNormalizer must be fit first")
        return (values - self.mean.to(values)) / self.std.to(values)

    def inverse(self, values: torch.Tensor) -> torch.Tensor:
        if self.mean is None or self.std is None:
            raise RuntimeError("TargetNormalizer must be fit first")
        return values * self.std.to(values) + self.mean.to(values)
