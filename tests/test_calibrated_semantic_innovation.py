import numpy as np
import torch

from safegrip.model.safegrip_universal import UniversalSafeGrip
from safegrip.training.trainer import (
    estimate_innovation_normalizer,
    estimate_semantic_feature_normalizer,
)
from safegrip.universal.dataset import collate_sensor_records
from safegrip.universal.queries import VELOCITY_X_QUERY
from safegrip.universal.schema import SensorChannel, SensorMeta, SensorRecord, TargetValue
from safegrip.universal.tokenizer import UniversalSensorTokenizer


def _record(seq: str, reference: float, target: float, amplitude: float = 1.0) -> SensorRecord:
    t = np.asarray([0.0, 0.25, 0.5, 0.75], dtype=float)
    vx = np.asarray([reference - 0.3, reference - 0.2, reference - 0.1, reference])
    ay = amplitude * np.sin(2.0 * np.pi * t)
    return SensorRecord(
        [
            SensorChannel(vx, t, SensorMeta("velocity", "m/s", "longitudinal", "vehicle_body", 4.0)),
            SensorChannel(ay, t, SensorMeta("acceleration", "m/s^2", "lateral", "vehicle_body", 4.0)),
        ],
        {"velocity_x": TargetValue("velocity_x", target)},
        sequence_id=seq,
        source_domain="deep_dynamics_iac",
    )


def test_reference_calibration_recovers_affine_coordinate_without_trainable_parameters():
    tok = UniversalSensorTokenizer()
    refs = [10.0, 20.0, 30.0, 40.0]
    records = [_record(str(i), x, 1.5 * x + 2.0) for i, x in enumerate(refs)]
    batch = collate_sensor_records(records, tok, (VELOCITY_X_QUERY,))
    normalizer = estimate_innovation_normalizer([batch], floor=1e-5)
    qids = torch.tensor([[VELOCITY_X_QUERY.ids()]] * len(records), dtype=torch.long)

    gain, bias = normalizer.reference_parameters_for(qids)
    assert torch.allclose(gain, torch.full_like(gain, 1.5), atol=1e-4)
    assert torch.allclose(bias, torch.full_like(bias, 2.0), atol=1e-3)
    assert sum(p.numel() for p in normalizer.parameters()) == 0

    anchor = torch.tensor(refs, dtype=torch.float32).unsqueeze(1)
    mask = torch.ones_like(anchor, dtype=torch.bool)
    z = normalizer.normalize_target(batch.target_values, qids, anchor, mask)
    assert torch.isfinite(z).all()
    assert float(torch.max(torch.abs(z))) < 1e-3


def test_semantic_feature_normalizer_is_train_only_parameter_free_and_order_equivariant():
    tok = UniversalSensorTokenizer()
    records = [
        _record("a", 10.0, 10.2, amplitude=1e-4),
        _record("b", 20.0, 20.2, amplitude=2e-4),
        _record("c", 30.0, 30.2, amplitude=3e-4),
    ]
    batch = collate_sensor_records(records, tok, (VELOCITY_X_QUERY,))
    norm = estimate_semantic_feature_normalizer([batch], tok.feature_dim)
    assert sum(p.numel() for p in norm.parameters()) == 0

    s = batch.sensors
    a = norm(s.features, s.quantity_ids, s.axis_ids, s.location_ids, s.unit_class_ids)
    perm = torch.randperm(s.features.shape[1])
    b = norm(
        s.features[:, perm], s.quantity_ids[:, perm], s.axis_ids[:, perm],
        s.location_ids[:, perm], s.unit_class_ids[:, perm],
    )
    assert torch.allclose(a[:, perm], b, atol=1e-6, rtol=1e-6)
    assert torch.isfinite(a).all()


def test_full_model_keeps_dataset_id_conditioning_disabled_with_both_normalizers():
    tok = UniversalSensorTokenizer()
    records = [_record("a", 10.0, 10.3), _record("b", 20.0, 20.5)]
    batch = collate_sensor_records(records, tok, (VELOCITY_X_QUERY,))
    feature_norm = estimate_semantic_feature_normalizer([batch], tok.feature_dim)
    innovation_norm = estimate_innovation_normalizer([batch])
    model = UniversalSafeGrip(
        tok.feature_dim,
        token_dim=24,
        latent_dim=48,
        latent_tokens=4,
        cross_attention_heads=4,
        latent_heads=4,
        latent_layers=1,
        dropout=0.0,
        feature_normalizer=feature_norm,
        innovation_normalizer=innovation_norm,
    )
    assert model.is_dataset_agnostic
    assert model.dataset_embedding is None
