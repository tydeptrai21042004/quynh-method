import numpy as np
import torch

from safegrip.ablations import get_ablation
from safegrip.datasets import PAPER_DATASETS
from safegrip.model.safegrip_universal import UniversalSafeGrip
from safegrip.universal.dataset import collate_sensor_records
from safegrip.universal.queries import queries_for_dataset
from safegrip.universal.schema import SensorMeta, SensorChannel, SensorRecord
from safegrip.universal.tokenizer import UniversalSensorTokenizer, TokenizerConfig


def _record(domain: str):
    t = np.linspace(0.0, 1.0, 25)
    return SensorRecord(
        [SensorChannel(np.sin(2*np.pi*t), t, SensorMeta("acceleration", "m/s^2", "lateral", "vehicle_body", 24.0))],
        sequence_id=f"unified-{domain}",
        source_domain=domain,
    )


def test_full_proposal_has_no_dataset_specific_parameters():
    tok = UniversalSensorTokenizer()
    model = UniversalSafeGrip(tok.feature_dim, token_dim=24, latent_dim=48, latent_tokens=4,
                              cross_attention_heads=4, latent_heads=4, latent_layers=1,
                              dropout=0.0)
    assert model.is_dataset_agnostic
    assert model.dataset_embedding is None
    assert not any("dataset_embedding" in n for n, _ in model.named_parameters())


def test_one_model_instance_runs_all_four_dataset_query_sets_without_domain_ids():
    torch.manual_seed(9)
    tok = UniversalSensorTokenizer()
    model = UniversalSafeGrip(tok.feature_dim, token_dim=24, latent_dim=48, latent_tokens=4,
                              cross_attention_heads=4, latent_heads=4, latent_layers=1,
                              dropout=0.0)
    model.eval()
    with torch.no_grad():
        for dataset in PAPER_DATASETS:
            queries = queries_for_dataset(dataset)
            batch = collate_sensor_records([_record(dataset)], tok, queries)
            # Deliberately omit domains: the same parameters must work for every dataset.
            out = model(batch.sensors, queries)
            assert out.point.shape == (1, len(queries))
            assert out.scale.shape == (1, len(queries))
            assert torch.isfinite(out.point).all()


def test_dataset_id_conditioning_exists_only_as_ablation():
    tok = UniversalSensorTokenizer(TokenizerConfig())
    model = UniversalSafeGrip(tok.feature_dim, token_dim=24, latent_dim=48, latent_tokens=4,
                              cross_attention_heads=4, latent_heads=4, latent_layers=1,
                              dropout=0.0, ablation=get_ablation("dataset_id_conditioning"))
    assert not model.is_dataset_agnostic
    assert model.dataset_embedding is not None


def test_one_shared_model_can_train_on_mixed_d1_d4_partial_supervision():
    from safegrip.universal.schema import TargetValue
    from safegrip.universal.dataset import DEFAULT_RESEARCH_QUERIES, collate_sensor_records
    from safegrip.training.trainer import estimate_target_scales, research_losses, ResearchLossConfig

    t = np.linspace(0.0, 1.0, 25)
    def rec(domain, targets):
        return SensorRecord(
            [SensorChannel(np.cos(2*np.pi*t), t, SensorMeta("acceleration", "m/s^2", "lateral", "vehicle_body", 24.0))],
            {k: TargetValue(k, float(v)) for k, v in targets.items()},
            sequence_id=f"mixed-{domain}", source_domain=domain,
        )

    records = [
        rec("lira_cd", {"friction": 0.8}),
        rec("uc3m_tire", {"force_x": 100.0, "force_y": 50.0, "force_z": 1200.0, "slip_angle": 0.03}),
        rec("deep_dynamics_iac", {"velocity_x": 12.0, "velocity_y": 0.3, "yaw_rate": 0.05}),
        rec("io_vnbd", {"displacement": 0.2, "orientation": 0.01}),
    ]
    tok = UniversalSensorTokenizer()
    batch = collate_sensor_records(records, tok, DEFAULT_RESEARCH_QUERIES)
    model = UniversalSafeGrip(tok.feature_dim, token_dim=24, latent_dim=48, latent_tokens=4,
                              cross_attention_heads=4, latent_heads=4, latent_layers=1,
                              dropout=0.0)
    scales = estimate_target_scales([batch])
    losses = research_losses(model, batch, scales, ResearchLossConfig(sensor_dropout=0.0), training=True)
    losses.total.backward()
    assert torch.isfinite(losses.total)
    assert model.is_dataset_agnostic
    assert model.dataset_embedding is None
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
