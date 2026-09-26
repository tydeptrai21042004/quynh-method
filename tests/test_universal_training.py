import torch

from safegrip.training.objectives import masked_huber, force_utilization, friction_inequality_loss, TargetNormalizer
from safegrip.training.sampler import multidataset_sample_weights
from safegrip.training.dropout import sensor_channel_dropout_mask


def test_partial_supervision_mask_and_normalizer():
    y = torch.tensor([[0.8, 1000.0], [0.7, float("nan")], [0.9, 1200.0]])
    mask = torch.tensor([[1, 1], [1, 0], [1, 1]], dtype=torch.bool)
    norm = TargetNormalizer().fit(y, mask)
    z = norm.transform(torch.nan_to_num(y))
    assert torch.isfinite(z).all()
    pred = torch.zeros_like(y)
    loss = masked_huber(pred, torch.nan_to_num(y), mask)
    assert loss > 0


def test_physics_inequality_penalizes_only_violation():
    mu = torch.tensor([0.8, 0.4])
    u = torch.tensor([0.5, 0.6])
    loss = friction_inequality_loss(mu, u)
    assert torch.allclose(loss, torch.tensor((0.2 ** 2) / 2), atol=1e-6)
    assert torch.allclose(force_utilization(torch.tensor([3.0]), torch.tensor([4.0]), torch.tensor([10.0])), torch.tensor([0.5]), atol=1e-5)


def test_sampler_rebalances_large_domain():
    ids = ["large"] * 100 + ["small"] * 4
    w = multidataset_sample_weights(ids, tau=0.0)
    assert torch.isclose(w[:100].sum(), w[100:].sum())


def test_sensor_dropout_drops_whole_channels_and_keeps_one():
    token_mask = torch.tensor([[1, 1, 1, 1, 1, 1]], dtype=torch.bool)
    channels = torch.tensor([[0, 0, 1, 1, 2, 2]])
    gen = torch.Generator().manual_seed(3)
    out = sensor_channel_dropout_mask(token_mask, channels, p=0.9, generator=gen)
    kept = torch.unique(channels[0][out[0]])
    assert len(kept) >= 1
    for ch in kept:
        assert torch.all(out[0][channels[0] == ch])
