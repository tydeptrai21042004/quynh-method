import math
import numpy as np
import torch

from safegrip.model.safegrip_universal import UniversalSafeGrip
from safegrip.training.trainer import estimate_innovation_normalizer
from safegrip.universal.dataset import collate_sensor_records
from safegrip.universal.queries import VELOCITY_X_QUERY, DISPLACEMENT_QUERY
from safegrip.universal.schema import SensorChannel, SensorMeta, SensorRecord, TargetValue
from safegrip.universal.tokenizer import UniversalSensorTokenizer


def _vx_record(seq: str, history_last: float, target: float) -> SensorRecord:
    t = np.asarray([0.0, 0.25, 0.5, 0.75], dtype=float)
    values = np.asarray([history_last - 0.3, history_last - 0.2, history_last - 0.1, history_last])
    return SensorRecord(
        [SensorChannel(values, t, SensorMeta("velocity", "m/s", "longitudinal", "vehicle_body", 4.0))],
        {"velocity_x": TargetValue("velocity_x", target)},
        sequence_id=seq,
        source_domain="deep_dynamics_iac",
    )


def test_npi_uses_small_residual_coordinate_when_reference_exists():
    tok = UniversalSensorTokenizer()
    records = [
        _vx_record("a", 20.0, 20.2),
        _vx_record("b", 30.0, 30.4),
        _vx_record("c", 40.0, 40.6),
    ]
    batch = collate_sensor_records(records, tok, (VELOCITY_X_QUERY,))
    normalizer = estimate_innovation_normalizer([batch], floor=1e-4)
    qids = torch.tensor([[VELOCITY_X_QUERY.ids()]] * len(records), dtype=torch.long)
    anchor = torch.tensor([[20.0], [30.0], [40.0]])
    mask = torch.ones_like(anchor, dtype=torch.bool)
    center, scale = normalizer.parameters_for(qids, mask)

    # Residuals are [0.2, 0.4, 0.6], so NPI learns a small residual coordinate
    # rather than the absolute 20--40 m/s target scale.
    assert torch.allclose(center, torch.full_like(center, 0.4), atol=1e-5)
    assert torch.all(scale > 0)
    assert float(scale.max()) < 1.0

    z = normalizer.normalize_target(batch.target_values, qids, anchor, mask)
    assert torch.isfinite(z).all()
    assert float(torch.max(torch.abs(z))) < 3.0


def test_npi_switches_to_direct_coordinate_if_reference_channel_is_missing():
    tok = UniversalSensorTokenizer()
    records = [
        _vx_record("a", 20.0, 20.2),
        _vx_record("b", 30.0, 30.4),
        _vx_record("c", 40.0, 40.6),
    ]
    batch = collate_sensor_records(records, tok, (VELOCITY_X_QUERY,))
    normalizer = estimate_innovation_normalizer([batch], floor=1e-4)
    qids = torch.tensor([[VELOCITY_X_QUERY.ids()]] * len(records), dtype=torch.long)

    no_ref = torch.zeros((len(records), 1), dtype=torch.bool)
    zero_anchor = torch.zeros((len(records), 1))
    z_direct = normalizer.normalize_target(batch.target_values, qids, zero_anchor, no_ref)

    # Missing-reference samples remain normalized O(1), rather than becoming
    # target/reference-scale outliers in the residual coordinate.
    assert torch.isfinite(z_direct).all()
    assert float(torch.max(torch.abs(z_direct))) < 3.0


def test_npi_reconstructs_exact_reference_plus_center_plus_scaled_innovation():
    tok = UniversalSensorTokenizer()
    records = [
        _vx_record("a", 20.0, 20.2),
        _vx_record("b", 30.0, 30.4),
        _vx_record("c", 40.0, 40.6),
    ]
    batch = collate_sensor_records(records, tok, (VELOCITY_X_QUERY,))
    normalizer = estimate_innovation_normalizer([batch], floor=1e-4)
    qids = torch.tensor([[VELOCITY_X_QUERY.ids()]] * len(records), dtype=torch.long)
    anchor = torch.tensor([[20.0], [30.0], [40.0]])
    mask = torch.ones_like(anchor, dtype=torch.bool)
    z = torch.tensor([[-1.0], [0.0], [1.0]])

    point, center, scale = normalizer.reconstruct(z, qids, anchor, mask)
    expected = anchor + center + scale * z
    assert torch.allclose(point, expected, atol=1e-7)


def test_zero_initialized_decoder_preserves_npi_reference_coordinate():
    tok = UniversalSensorTokenizer()
    records = [
        _vx_record("a", 20.0, 20.2),
        _vx_record("b", 30.0, 30.4),
        _vx_record("c", 40.0, 40.6),
    ]
    batch = collate_sensor_records(records, tok, (VELOCITY_X_QUERY,))
    normalizer = estimate_innovation_normalizer([batch], floor=1e-4)
    model = UniversalSafeGrip(
        tok.feature_dim,
        token_dim=24,
        latent_dim=48,
        latent_tokens=4,
        cross_attention_heads=4,
        latent_heads=4,
        latent_layers=1,
        dropout=0.0,
        innovation_normalizer=normalizer,
    )
    assert torch.count_nonzero(model.decoder.point_head.weight) == 0
    assert torch.count_nonzero(model.decoder.point_head.bias) == 0

    model.eval()
    with torch.no_grad():
        out = model(batch.sensors, batch.queries)
    expected = torch.where(out.anchor_mask, out.physical_anchor, torch.zeros_like(out.physical_anchor)) + out.innovation_center
    assert torch.allclose(out.innovation, torch.zeros_like(out.innovation), atol=0, rtol=0)
    assert torch.allclose(out.point, expected, atol=1e-6)


def test_curved_displacement_reference_is_endpoint_chord_not_path_length():
    tok = UniversalSensorTokenizer()
    t = np.linspace(0.0, 1.0, 21)
    speed = 10.0
    yaw_rate = math.pi / 2.0  # 90 degrees over one second
    record = SensorRecord(
        [
            SensorChannel(
                np.full_like(t, speed), t,
                SensorMeta("velocity", "m/s", "scalar", "vehicle_body", 20.0),
            ),
            SensorChannel(
                np.full_like(t, yaw_rate), t,
                SensorMeta("angular_rate", "rad/s", "yaw", "vehicle_body", 20.0),
            ),
        ],
        sequence_id="quarter-circle",
    )
    batch = collate_sensor_records([record], tok, (DISPLACEMENT_QUERY,))
    ref, mask = UniversalSafeGrip._physical_query_anchor(batch.sensors, (DISPLACEMENT_QUERY,))
    radius = speed / yaw_rate
    expected_chord = 2.0 * radius * math.sin((yaw_rate * 1.0) / 2.0)
    assert bool(mask.item())
    assert abs(float(ref.item()) - expected_chord) < 1e-4
    assert float(ref.item()) < speed * 1.0


def test_core_npi_ablation_coordinates_are_algebraically_distinct():
    tok = UniversalSensorTokenizer()
    records = [
        _vx_record("a", 20.0, 20.2),
        _vx_record("b", 30.0, 30.4),
        _vx_record("c", 40.0, 40.6),
    ]
    batch = collate_sensor_records(records, tok, (VELOCITY_X_QUERY,))
    qids = torch.tensor([[VELOCITY_X_QUERY.ids()]] * len(records), dtype=torch.long)
    anchor = torch.tensor([[20.0], [30.0], [40.0]])
    mask = torch.ones_like(anchor, dtype=torch.bool)
    z = torch.ones_like(anchor)

    outputs = {}
    for mode in (
        "npi_v3", "direct_normalized", "reference_only",
        "old_physical_innovation", "npi_no_center", "npi_no_scale",
    ):
        normalizer = estimate_innovation_normalizer([batch], floor=1e-4, mode=mode)
        point, center, scale = normalizer.reconstruct(z, qids, anchor, mask)
        outputs[mode] = point
        assert torch.isfinite(point).all()
        assert torch.isfinite(center).all()
        assert torch.isfinite(scale).all()

    # Reference-only is exactly P_q(X); old innovation is exactly P_q(X)+z.
    assert torch.allclose(outputs["reference_only"], anchor, atol=1e-7)
    assert torch.allclose(outputs["old_physical_innovation"], anchor + z, atol=1e-7)
    # Direct normalized prediction must not inherit the physical reference.
    assert not torch.allclose(outputs["direct_normalized"], outputs["npi_v3"])
    # Location- and scale-removal controls must genuinely change the coordinate.
    assert not torch.allclose(outputs["npi_no_center"], outputs["npi_v3"])
    assert not torch.allclose(outputs["npi_no_scale"], outputs["npi_v3"])


def test_direct_normalized_ignores_reference_even_when_observable():
    tok = UniversalSensorTokenizer()
    records = [
        _vx_record("a", 20.0, 20.2),
        _vx_record("b", 30.0, 30.4),
        _vx_record("c", 40.0, 40.6),
    ]
    batch = collate_sensor_records(records, tok, (VELOCITY_X_QUERY,))
    normalizer = estimate_innovation_normalizer(
        [batch], floor=1e-4, mode="direct_normalized"
    )
    qids = torch.tensor([[VELOCITY_X_QUERY.ids()]] * len(records), dtype=torch.long)
    mask = torch.ones((len(records), 1), dtype=torch.bool)
    a1 = torch.tensor([[20.0], [30.0], [40.0]])
    a2 = a1 + 1000.0
    z = torch.zeros_like(a1)
    p1, _, _ = normalizer.reconstruct(z, qids, a1, mask)
    p2, _, _ = normalizer.reconstruct(z, qids, a2, mask)
    assert torch.allclose(p1, p2, atol=1e-7)


def test_path_length_reference_control_differs_from_curvature_reference():
    from safegrip.universal.reference import PhysicalReferenceOperator

    tok = UniversalSensorTokenizer()
    t = np.linspace(0.0, 1.0, 21)
    speed = 10.0
    yaw_rate = math.pi / 2.0
    record = SensorRecord(
        [
            SensorChannel(
                np.full_like(t, speed), t,
                SensorMeta("velocity", "m/s", "scalar", "vehicle_body", 20.0),
            ),
            SensorChannel(
                np.full_like(t, yaw_rate), t,
                SensorMeta("angular_rate", "rad/s", "yaw", "vehicle_body", 20.0),
            ),
        ],
        sequence_id="quarter-circle-ablation",
    )
    batch = collate_sensor_records([record], tok, (DISPLACEMENT_QUERY,))
    curvature, m1 = PhysicalReferenceOperator(localization_mode="curvature")(
        batch.sensors, (DISPLACEMENT_QUERY,)
    )
    path_length, m2 = PhysicalReferenceOperator(localization_mode="path_length")(
        batch.sensors, (DISPLACEMENT_QUERY,)
    )
    assert bool(m1.item()) and bool(m2.item())
    assert abs(float(path_length.item()) - speed) < 1e-4
    assert float(curvature.item()) < float(path_length.item())
