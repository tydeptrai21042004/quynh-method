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
        last_values=None if batch.last_values is None else batch.last_values[:, perm],
        rms_values=None if batch.rms_values is None else batch.rms_values[:, perm],
        integral_values=None if batch.integral_values is None else batch.integral_values[:, perm],
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


def _state_record():
    t = np.asarray([0.0, 0.25, 0.5, 0.75], dtype=float)
    return SensorRecord([
        SensorChannel(
            np.asarray([10.0, 11.0, 12.0, 13.0]), t,
            SensorMeta("velocity", "m/s", "longitudinal", "vehicle_body", 4.0),
        ),
        SensorChannel(
            np.asarray([0.1, 0.2, 0.3, 0.4]), t,
            SensorMeta("velocity", "m/s", "lateral", "vehicle_body", 4.0),
        ),
        SensorChannel(
            np.asarray([0.01, 0.02, 0.03, 0.04]), t,
            SensorMeta("angular_rate", "rad/s", "yaw", "vehicle_body", 4.0),
        ),
    ], sequence_id="state")


def test_query_matched_physical_anchor_is_latest_canonical_sensor_value():
    from safegrip.universal.queries import VELOCITY_X_QUERY, VELOCITY_Y_QUERY, YAW_RATE_QUERY

    tok = UniversalSensorTokenizer()
    batch = pad_tokenized_records([tok.tokenize(_state_record())])
    anchor, mask = UniversalSafeGrip._physical_query_anchor(
        batch, [VELOCITY_X_QUERY, VELOCITY_Y_QUERY, YAW_RATE_QUERY]
    )
    assert mask.tolist() == [[True, True, True]]
    assert torch.allclose(anchor, torch.tensor([[13.0, 0.4, 0.04]]), atol=1e-6)


def test_query_anchor_is_strict_and_unobserved_query_is_original_predictor():
    from safegrip.universal.queries import FRICTION_QUERY

    torch.manual_seed(7)
    tok = UniversalSensorTokenizer()
    batch = pad_tokenized_records([tok.tokenize(_state_record())])
    model = UniversalSafeGrip(
        tok.feature_dim, token_dim=32, latent_dim=48, latent_tokens=8,
        cross_attention_heads=4, latent_heads=4, latent_layers=1, dropout=0.0,
    )
    model.eval()
    with torch.no_grad():
        out = model(batch, [FRICTION_QUERY])
    assert not bool(out.anchor_mask.item())
    assert float(out.physical_anchor.item()) == 0.0
    # Non-interference: without a semantic observation match, the final point is
    # exactly the pre-existing decoder output.
    assert torch.equal(out.point, out.innovation)


def test_query_anchor_reparameterizes_decoder_as_innovation_without_new_parameters():
    from safegrip.universal.queries import VELOCITY_X_QUERY

    torch.manual_seed(8)
    tok = UniversalSensorTokenizer()
    batch = pad_tokenized_records([tok.tokenize(_state_record())])
    model = UniversalSafeGrip(
        tok.feature_dim, token_dim=32, latent_dim=48, latent_tokens=8,
        cross_attention_heads=4, latent_heads=4, latent_layers=1, dropout=0.0,
    )
    n_params_before = sum(p.numel() for p in model.parameters())
    model.eval()
    with torch.no_grad():
        out = model(batch, [VELOCITY_X_QUERY])
    n_params_after = sum(p.numel() for p in model.parameters())
    assert n_params_before == n_params_after
    assert bool(out.anchor_mask.item())
    assert torch.allclose(out.physical_anchor, torch.tensor([[13.0]]), atol=1e-6)
    assert torch.allclose(out.point, out.physical_anchor + out.innovation, atol=0.0, rtol=0.0)


def test_query_anchor_preserves_token_permutation_invariance():
    from safegrip.universal.queries import VELOCITY_X_QUERY

    torch.manual_seed(9)
    tok = UniversalSensorTokenizer()
    batch = pad_tokenized_records([tok.tokenize(_state_record())])
    perm = torch.randperm(batch.features.shape[1])
    pbatch = _permute_batch(batch, perm)
    model = UniversalSafeGrip(
        tok.feature_dim, token_dim=32, latent_dim=48, latent_tokens=8,
        cross_attention_heads=4, latent_heads=4, latent_layers=1, dropout=0.0,
    )
    model.eval()
    with torch.no_grad():
        a = model(batch, [VELOCITY_X_QUERY])
        b = model(pbatch, [VELOCITY_X_QUERY])
    assert torch.allclose(a.physical_anchor, b.physical_anchor, atol=0.0, rtol=0.0)
    assert torch.allclose(a.point, b.point, atol=1e-6, rtol=1e-6)


def test_next_state_anchor_uses_history_only_not_target_row():
    import pandas as pd
    from safegrip.sensor_io.deep_dynamics_iac import deep_dynamics_iac_frame_to_record
    from safegrip.universal.queries import VELOCITY_X_QUERY

    frame = pd.DataFrame({
        "time": [0.00, 0.04, 0.08],
        "vx": [10.0, 11.0, 15.0],
        "vy": [0.1, 0.2, 0.4],
        "omega": [0.01, 0.02, 0.05],
        "delta": [0.0, 0.0, 0.0],
    })
    record = deep_dynamics_iac_frame_to_record(frame, sequence_id="next-state")
    assert float(record.targets["velocity_x"].value) == 15.0
    # The matching input channel ends one step earlier by adapter construction.
    assert float(record.channels[0].values[-1]) == 11.0

    tok = UniversalSensorTokenizer()
    batch = pad_tokenized_records([tok.tokenize(record)])
    anchor, mask = UniversalSafeGrip._physical_query_anchor(batch, [VELOCITY_X_QUERY])
    assert bool(mask.item())
    assert float(anchor.item()) == 11.0
    assert float(anchor.item()) != float(record.targets["velocity_x"].value)


def test_friction_reference_is_horizontal_specific_force_and_dataset_agnostic():
    from safegrip.universal.queries import FRICTION_QUERY

    t = np.asarray([0.0, 0.25, 0.5, 0.75], dtype=float)
    record = SensorRecord([
        SensorChannel(
            np.full_like(t, 3.0), t,
            SensorMeta("acceleration", "m/s^2", "longitudinal", "vehicle_body", 4.0),
        ),
        SensorChannel(
            np.full_like(t, 4.0), t,
            SensorMeta("acceleration", "m/s^2", "lateral", "vehicle_body", 4.0),
        ),
    ], sequence_id="friction-physics")
    tok = UniversalSensorTokenizer()
    batch = pad_tokenized_records([tok.tokenize(record)])
    ref, mask = UniversalSafeGrip._physical_query_anchor(batch, [FRICTION_QUERY])
    assert bool(mask.item())
    assert torch.allclose(ref, torch.tensor([[5.0 / 9.80665]]), atol=1e-6)


def test_displacement_reference_integrates_measured_body_speed():
    from safegrip.universal.queries import DISPLACEMENT_QUERY

    t = np.linspace(0.0, 1.0, 11)
    record = SensorRecord([
        SensorChannel(
            np.full_like(t, 10.0), t,
            SensorMeta("velocity", "m/s", "scalar", "vehicle_body", 10.0),
        ),
    ], sequence_id="odometry")
    tok = UniversalSensorTokenizer()
    batch = pad_tokenized_records([tok.tokenize(record)])
    ref, mask = UniversalSafeGrip._physical_query_anchor(batch, [DISPLACEMENT_QUERY])
    assert bool(mask.item())
    assert torch.allclose(ref, torch.tensor([[10.0]]), atol=1e-5)
