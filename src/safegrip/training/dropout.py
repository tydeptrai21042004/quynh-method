from __future__ import annotations

import torch


def sensor_channel_dropout_mask(
    token_mask: torch.Tensor,
    channel_ids: torch.Tensor,
    p: float = 0.2,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Drop complete sensor channels while keeping at least one channel/sample."""
    if not 0.0 <= p < 1.0:
        raise ValueError("p must lie in [0,1)")
    if token_mask.shape != channel_ids.shape:
        raise ValueError("token_mask and channel_ids must have identical shapes")
    out = token_mask.clone()
    if p == 0.0:
        return out
    for i in range(token_mask.shape[0]):
        valid_channels = torch.unique(channel_ids[i][token_mask[i] & (channel_ids[i] >= 0)])
        if len(valid_channels) <= 1:
            continue
        keep = []
        for ch in valid_channels:
            draw = torch.rand((), generator=generator, device=token_mask.device)
            if draw >= p:
                keep.append(int(ch))
        if not keep:
            # Randomly preserve one channel to keep attention numerically valid.
            idx = int(torch.randint(len(valid_channels), (), generator=generator, device=token_mask.device))
            keep = [int(valid_channels[idx])]
        keep_mask = channel_ids[i] < 0  # preserve static/context tokens
        for ch in keep:
            keep_mask |= channel_ids[i] == ch
        out[i] = token_mask[i] & keep_mask
    return out
