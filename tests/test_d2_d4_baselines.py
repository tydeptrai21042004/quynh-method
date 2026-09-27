import torch

from safegrip.paper_baselines import (
    BASELINE_TARGETS, FTHD2025, Mendoza2019Fuzzy, Onyekpe2021QGRU,
    make_paper_baseline,
)


def test_all_missing_d2_d4_baselines_are_locally_buildable_and_finite():
    cases = {
        "mendoza2019_fuzzy": (3, 4),
        "yunta2018_fuzzy_lfc": (3, 3),
        "chrosniak2024_ddm": (5, 3),
        "fang_yu2025_fthd": (5, 3),
        "onyekpe2021_qgru": (8, 2),
        "wang2023_transformer": (8, 2),
    }
    for name, (d, q) in cases.items():
        built = make_paper_baseline(name, d, debug_scale=True)
        y = built.model(torch.randn(2, 12, d))
        assert y.shape == (2, q)
        assert torch.isfinite(y).all()
        assert len(BASELINE_TARGETS[name]) == q


def test_mendoza_hierarchical_paper_rule_counts():
    m = Mendoza2019Fuzzy(3, debug_scale=False)
    assert (m.slip.rules, m.vertical.rules, m.lateral.rules, m.longitudinal.rules) == (21, 217, 460, 145)


def test_fthd_exposes_hybrid_physics_loss():
    m = FTHD2025(5, debug_scale=True)
    x = torch.randn(4, 10, 5)
    y = torch.randn(4, 3)
    loss = m.hybrid_loss(x, y)
    assert torch.isfinite(loss) and loss.ndim == 0


def test_qgru_uses_quaternion_packed_hidden_state():
    m = Onyekpe2021QGRU(7, debug_scale=True)
    assert m.qin % 4 == 0 and m.hidden_size % 4 == 0
