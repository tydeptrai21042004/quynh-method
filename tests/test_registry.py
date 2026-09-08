from safegrip.datasets import DATASET_REGISTRY
from safegrip.benchmark import PROPOSAL_VARIANTS


def test_dataset_registry_has_multiple_real_sources():
    expected={"lira","kuleuven","kit","deep_dynamics","comma2k19","extreme_road","bicycle_tire","mendeley_friction"}
    assert expected.issubset(DATASET_REGISTRY)
    for name in expected:
        assert DATASET_REGISTRY[name]["role"]
        assert DATASET_REGISTRY[name]["auto_download"]


def test_ablation_covers_safegrip_v2_components():
    expected={
        "safegrip_data_only","safegrip_static_only","safegrip_no_gate",
        "safegrip_no_bound","safegrip_no_uq","safegrip_no_calibration","safegrip",
    }
    assert expected.issubset(PROPOSAL_VARIANTS)
