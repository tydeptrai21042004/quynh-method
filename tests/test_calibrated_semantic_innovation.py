import numpy as np
import torch

from safegrip.model.safegrip_universal import UniversalSafeGrip
from safegrip.training.trainer import (
    estimate_innovation_normalizer,
    estimate_semantic_feature_normalizer,
)
from safegrip.universal.dataset import collate_sensor_records
from safegrip.universal.feature_normalization import SemanticRelationalContrast
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


def test_ra_npi_gain_is_identity_anchored_and_parameter_free():
    tok = UniversalSensorTokenizer()
    refs = [10.0, 20.0, 30.0, 40.0]
    records = [_record(str(i), x, 1.5 * x + 2.0) for i, x in enumerate(refs)]
    batch = collate_sensor_records(records, tok, (VELOCITY_X_QUERY,))
    normalizer = estimate_innovation_normalizer([batch], floor=1e-5)
    qids = torch.tensor([[VELOCITY_X_QUERY.ids()]] * len(records), dtype=torch.long)

    gain, bias = normalizer.reference_parameters_for(qids)
    # The unconstrained synthetic slope is 1.5.  RA-NPI deliberately remains
    # between the v3 identity gain and that free fit because v4 evidence showed
    # that unconstrained calibration can hurt an already useful reference.
    assert torch.all(gain >= 1.0)
    assert torch.all(gain < 1.5)
    assert torch.allclose(bias, torch.zeros_like(bias), atol=0.0, rtol=0.0)
    assert sum(p.numel() for p in normalizer.parameters()) == 0

    anchor = torch.tensor(refs, dtype=torch.float32).unsqueeze(1)
    mask = torch.ones_like(anchor, dtype=torch.bool)
    z = normalizer.normalize_target(batch.target_values, qids, anchor, mask)
    assert torch.isfinite(z).all()
    assert float(torch.max(torch.abs(z))) < 3.0


def test_semantic_relational_contrast_is_mean_preserving_singleton_identity_and_order_equivariant():
    transform = SemanticRelationalContrast()
    features = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [10.0, 20.0]]])
    mask = torch.tensor([[True, True, True]])
    # First two tokens are the same physical semantic type at the same patch
    # time; the third is at another time and is therefore a singleton group.
    q = torch.zeros((1, 3), dtype=torch.long)
    a = torch.zeros((1, 3), dtype=torch.long)
    l = torch.zeros((1, 3), dtype=torch.long)
    u = torch.zeros((1, 3), dtype=torch.long)
    times = torch.tensor([[0.25, 0.25, 0.50]])
    channels = torch.tensor([[0, 1, 2]])

    out = transform(features, mask, q, a, l, u, times, channels)
    assert torch.allclose(out[0, :2].mean(dim=0), features[0, :2].mean(dim=0))
    assert torch.allclose(out[0, 2], features[0, 2])  # singleton => exact identity
    assert torch.allclose(out[0, 0], torch.tensor([0.0, 1.0]))
    assert torch.allclose(out[0, 1], torch.tensor([4.0, 5.0]))
    assert sum(p.numel() for p in transform.parameters()) == 0

    perm = torch.tensor([2, 0, 1])
    perm_out = transform(
        features[:, perm], mask[:, perm], q[:, perm], a[:, perm], l[:, perm], u[:, perm],
        times[:, perm], channels[:, perm],
    )
    assert torch.allclose(out[:, perm], perm_out, atol=1e-6, rtol=1e-6)


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
    first = norm(s.features, s.quantity_ids, s.axis_ids, s.location_ids, s.unit_class_ids)
    perm = torch.randperm(s.features.shape[1])
    second = norm(
        s.features[:, perm], s.quantity_ids[:, perm], s.axis_ids[:, perm],
        s.location_ids[:, perm], s.unit_class_ids[:, perm],
    )
    assert torch.allclose(first[:, perm], second, atol=1e-6, rtol=1e-6)
    assert torch.isfinite(first).all()


def test_full_model_keeps_dataset_id_conditioning_disabled_with_ra_npi_transforms():
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
    assert sum(p.numel() for p in model.relational_transform.parameters()) == 0
