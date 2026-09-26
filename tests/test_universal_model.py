import numpy as np
import torch

from safegrip.universal.schema import SensorMeta, SensorChannel, SensorRecord
from safegrip.universal.tokenizer import UniversalSensorTokenizer
from safegrip.universal.batching import pad_tokenized_records, UniversalSensorBatch
from safegrip.universal.queries import FRICTION_QUERY, FORCE_X_QUERY
from safegrip.model.safegrip_universal import UniversalSafeGrip


def _record(n_channels: int, suffix="x"):
    channels = []
    for i in range(n_channels):
        rate = 20.0 + 5.0 * i
        t = np.arange(0.0, 1.0, 1.0 / rate)
        x = np.sin(2 * np.pi * (1.0 + i) * t)
        channels.append(SensorChannel(x, t, SensorMeta("acceleration", "m/s^2", "lateral", "vehicle_body", rate)))
    return SensorRecord(channels, sequence_id=f"{suffix}-{n_channels}")


def _permute_batch(batch, perm):
    return UniversalSensorBatch(
        features=batch.features[:, perm],
        token_mask=batch.token_mask[:, perm],
        quantity_ids=batch.quantity_ids[:, perm],
        axis_ids=batch.axis_ids[:, perm],
        location_ids=batch.location_ids[:, perm],
        unit_class_ids=batch.unit_class_ids[:, perm],
        sample_rates_hz=batch.sample_rates_hz[:, perm],
        times_sec=batch.times_sec[:, perm],
        channel_ids=batch.channel_ids[:, perm],
        sequence_ids=batch.sequence_ids,
    )


def test_variable_sensor_counts_one_model():
    torch.manual_seed(1)
    tok = UniversalSensorTokenizer()
    zs = [tok.tokenize(_record(n)) for n in (3, 8, 25)]
    batch = pad_tokenized_records(zs)
    model = UniversalSafeGrip(tok.feature_dim, token_dim=32, latent_dim=48, latent_tokens=8, cross_attention_heads=4, latent_heads=4, latent_layers=1, dropout=0.0)
    model.eval()
    with torch.no_grad():
        out = model(batch, [FRICTION_QUERY, FORCE_X_QUERY])
    assert out.point.shape == (3, 2)
    assert out.scale.shape == (3, 2)
    assert torch.all(out.scale > 0)


def test_token_order_invariance():
    torch.manual_seed(2)
    tok = UniversalSensorTokenizer()
    batch = pad_tokenized_records([tok.tokenize(_record(4))])
    model = UniversalSafeGrip(tok.feature_dim, token_dim=32, latent_dim=48, latent_tokens=8, cross_attention_heads=4, latent_heads=4, latent_layers=1, dropout=0.0)
    model.eval()
    perm = torch.randperm(batch.features.shape[1])
    pbatch = _permute_batch(batch, perm)
    with torch.no_grad():
        a = model(batch, [FRICTION_QUERY]).point
        b = model(pbatch, [FRICTION_QUERY]).point
    assert torch.allclose(a, b, atol=1e-6, rtol=1e-6)
