import numpy as np
import pandas as pd
import pytest
import torch

from safegrip.ablations import ABLATIONS, get_ablation
from safegrip.universal.queries import queries_for_dataset
from safegrip.universal.units import canonicalize_values
from safegrip.sensor_io.uc3m_tire import uc3m_tire_frame_to_record
from safegrip.sensor_io.deep_dynamics_iac import deep_dynamics_iac_frame_to_record
from safegrip.sensor_io.io_vnbd import io_vnbd_frame_to_record
from safegrip.universal.tokenizer import UniversalSensorTokenizer, TokenizerConfig
from safegrip.universal.dataset import collate_paper_records
from safegrip.model.safegrip_universal import UniversalSafeGrip


def test_dataset_specific_queries_are_not_one_friction_task():
    assert [q.name for q in queries_for_dataset("lira_cd")] == ["friction"]
    assert [q.name for q in queries_for_dataset("uc3m_tire")] == ["slip_angle"]
    assert [q.name for q in queries_for_dataset("deep_dynamics_iac")] == ["velocity_x", "velocity_y", "yaw_rate"]
    assert [q.name for q in queries_for_dataset("io_vnbd")] == ["displacement"]


def test_uc3m_adapter_does_not_manufacture_missing_forces():
    f = pd.DataFrame({"strain_1": [10, 12], "strain_2": [20, 22], "strain_3": [30, 32]})
    r = uc3m_tire_frame_to_record(f, slip_angle_deg=6)
    assert set(r.targets) == {"slip_angle"}
    assert r.source_domain == "uc3m_tire"


def test_deep_dynamics_adapter_uses_last_state_as_target_not_input_history():
    f = pd.DataFrame({"vx": [10.0, 11.0, 12.0], "vy": [1.0, 1.2, 1.4], "omega": [0.1, 0.2, 0.3], "delta": [0.0, 0.01, 0.02]})
    r = deep_dynamics_iac_frame_to_record(f)
    assert r.targets["velocity_x"].value == 12.0
    assert len(r.channels[0].values) == 2


def test_io_vnbd_wheel_speed_remains_angular():
    values, tr = canonicalize_values([60.0], "wheel_speed", "rpm")
    assert tr.canonical_unit == "rad/s"
    assert np.allclose(values[0], 2 * np.pi)
    f = pd.DataFrame({"wheel_speed_fl": [1.0, 2.0], "ax": [0.0, 0.1]})
    r = io_vnbd_frame_to_record(f)
    wheel = next(c for c in r.channels if c.meta.quantity == "wheel_speed")
    assert wheel.meta.canonical_unit == "rad/s"


def test_all_ablation_configs_build_and_change_declared_switch():
    assert set(ABLATIONS) == {
        "npi_v3", "direct_normalized", "reference_only", "old_physical_innovation",
        "npi_no_center", "npi_no_scale", "mean_pool", "channel_id_only",
        "no_physical_metadata", "no_spectrum", "no_sensor_dropout",
        "dataset_id_conditioning", "path_length_reference", "no_learned_scale",
    }
    assert get_ablation("npi_v3").innovation_mode == "npi_v3"
    assert get_ablation("old_physical_innovation").innovation_mode == "old_physical_innovation"
    assert get_ablation("path_length_reference").localization_reference == "path_length"
    assert get_ablation("mean_pool").aggregation == "mean_pool"
    assert get_ablation("channel_id_only").channel_id_embedding is True
    assert get_ablation("dataset_id_conditioning").dataset_id_conditioning is True


def test_dataset_id_ablation_requires_domain_and_runs_with_domain():
    f = pd.DataFrame({"strain_1": [10, 12, 11], "strain_2": [20, 22, 21], "strain_3": [30, 32, 31], "fx": [1, 2, 3]})
    record = uc3m_tire_frame_to_record(f)
    ab = get_ablation("dataset_id_conditioning")
    tok = UniversalSensorTokenizer(TokenizerConfig(use_spectrum=ab.spectrum))
    batch = collate_paper_records([record], tok, "uc3m_tire")
    model = UniversalSafeGrip(tok.feature_dim, token_dim=24, latent_dim=48, latent_tokens=4, cross_attention_heads=4, latent_heads=4, latent_layers=1, dropout=0.0, ablation=ab)
    with pytest.raises(ValueError, match="domains are required"):
        model(batch.sensors, batch.queries)
    with torch.no_grad():
        out = model(batch.sensors, batch.queries, domains=batch.domains)
    assert out.point.shape == (1, 1)
