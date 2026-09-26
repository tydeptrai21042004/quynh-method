import numpy as np
import torch

from safegrip.universal.schema import SensorMeta, SensorChannel, SensorRecord, TargetValue
from safegrip.universal.tokenizer import UniversalSensorTokenizer
from safegrip.universal.dataset import collate_sensor_records
from safegrip.model.safegrip_universal import UniversalSafeGrip
from safegrip.training.trainer import estimate_target_scales, research_losses, ResearchLossConfig


def _record(domain, mu=None, fx=None, fy=None, fz=None, u=None):
    t = np.linspace(0, 1, 21)
    ch = SensorChannel(np.sin(t), t, SensorMeta("acceleration", "m/s^2", "lateral", "vehicle_body", 20.0))
    targets = {}
    for name, val in {"friction": mu, "force_x": fx, "force_y": fy, "force_z": fz, "utilization": u}.items():
        if val is not None:
            targets[name] = TargetValue(name, float(val))
    return SensorRecord([ch], targets, sequence_id=domain, source_domain=domain)


def test_partial_target_collate_and_research_loss():
    tok = UniversalSensorTokenizer()
    b1 = collate_sensor_records([_record("lira", mu=0.8), _record("kit", fx=100, fy=50, fz=1000, u=0.12)], tok)
    assert b1.target_mask[0].sum() == 1
    assert b1.target_mask[1].sum() == 4
    scales = estimate_target_scales([b1])
    model = UniversalSafeGrip(tok.feature_dim, token_dim=24, latent_dim=48, latent_tokens=4, cross_attention_heads=4, latent_heads=4, latent_layers=1, dropout=0.0)
    losses = research_losses(model, b1, scales, ResearchLossConfig(sensor_dropout=0.0), training=False)
    assert torch.isfinite(losses.total)
