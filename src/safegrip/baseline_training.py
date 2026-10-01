from __future__ import annotations

"""Small common trainer for the D2--D4 paper-structured baseline reproductions.

The purpose is to make every registered comparator genuinely trainable with the
same split/preprocessing supplied by the dataset benchmark.  Dataset-specific
logic stays outside the models: callers select compatible target columns, while
this module applies one optimization contract to all baselines.
"""

from dataclasses import dataclass
import torch
from torch import nn
import torch.nn.functional as F

from .paper_baselines import FTHD2025, WhONet2021


@dataclass(frozen=True)
class BaselineTrainResult:
    losses: tuple[float, ...]


def baseline_loss(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if isinstance(model, FTHD2025):
        return model.hybrid_loss(x, y)
    pred = model(x)
    if pred.shape != y.shape:
        raise ValueError(f"prediction/target shape mismatch: {tuple(pred.shape)} vs {tuple(y.shape)}")
    if isinstance(model, WhONet2021):
        # WhONet source implementation optimizes absolute displacement error.
        return F.l1_loss(pred, y)
    return F.mse_loss(pred, y)


def train_baseline(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    epochs: int = 1,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
) -> BaselineTrainResult:
    if epochs < 1:
        raise ValueError("epochs must be >= 1")
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    losses: list[float] = []
    for _ in range(int(epochs)):
        opt.zero_grad(set_to_none=True)
        loss = baseline_loss(model, x, y)
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite baseline training loss")
        loss.backward()
        opt.step()
        losses.append(float(loss.detach().cpu()))
    return BaselineTrainResult(tuple(losses))
