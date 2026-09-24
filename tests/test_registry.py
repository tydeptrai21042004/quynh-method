from safegrip.datasets import DATASET_REGISTRY
from safegrip.literature import PAPER_BASELINES, LITERATURE_BASELINES, validate_paper_baselines


def test_dataset_registry_has_multiple_real_sources():
    expected={"lira","kuleuven","kit","deep_dynamics","comma2k19","extreme_road","bicycle_tire","mendeley_friction"}
    assert expected.issubset(DATASET_REGISTRY)


def test_only_requested_primary_paper_baselines_are_registered():
    expected=(
        "du2023_inceptiontime",
        "todorovic2022_cnn",
        "lampe2023_gru",
        "levenberg2023_stft",
    )
    assert PAPER_BASELINES==expected
    validate_paper_baselines(PAPER_BASELINES)
    assert set(LITERATURE_BASELINES)==set(expected)
    for name in expected:
        assert LITERATURE_BASELINES[name]["doi"]
        assert "friction" in (LITERATURE_BASELINES[name]["task"]+LITERATURE_BASELINES[name]["title"]).lower() or "grip" in (LITERATURE_BASELINES[name]["task"]+LITERATURE_BASELINES[name]["title"]).lower()


def test_levenberg_sampling_limitation_is_explicit():
    meta=LITERATURE_BASELINES["levenberg2023_stft"]
    assert "low-rate" in meta["fidelity"]
    assert "20 Hz" in meta["common_benchmark_note"]
    assert "70-125 Hz" in meta["common_benchmark_note"]
