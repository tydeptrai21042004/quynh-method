from __future__ import annotations

from dataclasses import dataclass
import torch

from .tokenizer import TokenizedRecord


@dataclass
class UniversalSensorBatch:
    features: torch.Tensor
    token_mask: torch.Tensor
    quantity_ids: torch.Tensor
    axis_ids: torch.Tensor
    location_ids: torch.Tensor
    unit_class_ids: torch.Tensor
    sample_rates_hz: torch.Tensor
    times_sec: torch.Tensor
    channel_ids: torch.Tensor
    sequence_ids: tuple[str, ...]

    def to(self, device):
        return UniversalSensorBatch(
            features=self.features.to(device),
            token_mask=self.token_mask.to(device),
            quantity_ids=self.quantity_ids.to(device),
            axis_ids=self.axis_ids.to(device),
            location_ids=self.location_ids.to(device),
            unit_class_ids=self.unit_class_ids.to(device),
            sample_rates_hz=self.sample_rates_hz.to(device),
            times_sec=self.times_sec.to(device),
            channel_ids=self.channel_ids.to(device),
            sequence_ids=self.sequence_ids,
        )


def pad_tokenized_records(records: list[TokenizedRecord] | tuple[TokenizedRecord, ...]) -> UniversalSensorBatch:
    if not records:
        raise ValueError("at least one TokenizedRecord is required")
    feat_dim = records[0].features.shape[1]
    if any(r.features.shape[1] != feat_dim for r in records):
        raise ValueError("all tokenized records must share feature dimension")
    b = len(records)
    nmax = max(r.num_tokens for r in records)
    features = torch.zeros((b, nmax, feat_dim), dtype=torch.float32)
    mask = torch.zeros((b, nmax), dtype=torch.bool)
    q = torch.zeros((b, nmax), dtype=torch.long)
    a = torch.zeros((b, nmax), dtype=torch.long)
    l = torch.zeros((b, nmax), dtype=torch.long)
    u = torch.zeros((b, nmax), dtype=torch.long)
    fs = torch.zeros((b, nmax), dtype=torch.float32)
    tt = torch.zeros((b, nmax), dtype=torch.float32)
    ch = torch.full((b, nmax), -1, dtype=torch.long)
    for i, r in enumerate(records):
        n = r.num_tokens
        features[i, :n] = torch.from_numpy(r.features)
        mask[i, :n] = True
        q[i, :n] = torch.from_numpy(r.quantity_ids)
        a[i, :n] = torch.from_numpy(r.axis_ids)
        l[i, :n] = torch.from_numpy(r.location_ids)
        u[i, :n] = torch.from_numpy(r.unit_class_ids)
        fs[i, :n] = torch.from_numpy(r.sample_rates_hz)
        tt[i, :n] = torch.from_numpy(r.times_sec)
        ch[i, :n] = torch.from_numpy(r.channel_ids)
    return UniversalSensorBatch(features, mask, q, a, l, u, fs, tt, ch, tuple(r.sequence_id for r in records))
