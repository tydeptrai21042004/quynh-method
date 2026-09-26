from __future__ import annotations

import numpy as np
import torch

from .schema import SensorMeta, SensorChannel, SensorRecord
from .tokenizer import UniversalSensorTokenizer, TokenizerConfig
from .batching import pad_tokenized_records, UniversalSensorBatch
from .queries import FRICTION_QUERY, FORCE_X_QUERY, UTILIZATION_QUERY
from safegrip.model.safegrip_universal import UniversalSafeGrip


def _synthetic_record(n_channels: int, sequence_id: str) -> SensorRecord:
    channels = []
    quantities = [
        ("acceleration", "m/s^2", "longitudinal", "vehicle_body"),
        ("acceleration", "m/s^2", "lateral", "vehicle_body"),
        ("angular_rate", "rad/s", "yaw", "vehicle_body"),
        ("force", "N", "lateral", "front_left"),
        ("strain", "microstrain", "scalar", "tire"),
    ]
    for i in range(n_channels):
        q, unit, axis, loc = quantities[i % len(quantities)]
        rate = [20.0, 100.0, 1000.0][i % 3]
        t = np.arange(0.0, 1.0, 1.0 / rate)
        x = (i + 1) * np.sin(2 * np.pi * (1 + i) * t)
        channels.append(SensorChannel(x, t, SensorMeta(q, unit, axis, loc, rate)))
    return SensorRecord(channels, sequence_id=sequence_id)


def run_universal_smoke(seed: int = 7) -> dict:
    torch.manual_seed(seed)
    tokenizer = UniversalSensorTokenizer(TokenizerConfig())
    records = [_synthetic_record(3, "three"), _synthetic_record(5, "five")]
    tokenized = [tokenizer.tokenize(r) for r in records]
    batch = pad_tokenized_records(tokenized)
    model = UniversalSafeGrip(
        tokenizer.feature_dim,
        token_dim=32,
        latent_dim=48,
        latent_tokens=8,
        cross_attention_heads=4,
        latent_heads=4,
        latent_layers=1,
        dropout=0.0,
    ).eval()
    queries = [FRICTION_QUERY, FORCE_X_QUERY, UTILIZATION_QUERY]
    with torch.no_grad():
        out = model(batch, queries)

    # Explicit set-order invariance audit on the first sample's real tokens.
    n = int(batch.token_mask[0].sum())
    perm = torch.arange(batch.features.shape[1])
    perm[:n] = torch.arange(n - 1, -1, -1)
    pb = UniversalSensorBatch(
        batch.features[[0]][:, perm], batch.token_mask[[0]][:, perm],
        batch.quantity_ids[[0]][:, perm], batch.axis_ids[[0]][:, perm],
        batch.location_ids[[0]][:, perm], batch.unit_class_ids[[0]][:, perm],
        batch.sample_rates_hz[[0]][:, perm], batch.times_sec[[0]][:, perm],
        batch.channel_ids[[0]][:, perm], (batch.sequence_ids[0],),
    )
    with torch.no_grad():
        a = model(UniversalSensorBatch(
            batch.features[[0]], batch.token_mask[[0]], batch.quantity_ids[[0]], batch.axis_ids[[0]],
            batch.location_ids[[0]], batch.unit_class_ids[[0]], batch.sample_rates_hz[[0]],
            batch.times_sec[[0]], batch.channel_ids[[0]], (batch.sequence_ids[0],),
        ), queries).point
        b = model(pb, queries).point
    return {
        "status": "ok",
        "feature_dim": tokenizer.feature_dim,
        "token_counts": [int(z.num_tokens) for z in tokenized],
        "prediction_shape": list(out.point.shape),
        "all_scales_positive": bool(torch.all(out.scale > 0)),
        "max_order_invariance_abs_diff": float(torch.max(torch.abs(a - b))),
    }
