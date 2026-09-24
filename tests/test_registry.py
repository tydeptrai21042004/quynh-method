from safegrip.datasets import DATASET_REGISTRY, PRIMARY_FRICTION_DATASETS
from safegrip.literature import PAPER_BASELINES, LITERATURE_BASELINES, validate_paper_baselines


def test_dataset_registry_has_multiple_real_sources():
    expected={"lira","kuleuven","kit","deep_dynamics","comma2k19","extreme_road","bicycle_tire","mendeley_friction","mssp2023_friction"}
    assert expected.issubset(DATASET_REGISTRY)



def test_primary_friction_benchmarks_are_real_only():
    assert PRIMARY_FRICTION_DATASETS == ("lira", "mssp2023_friction")
    assert all("synthetic" not in name and "simulat" not in name for name in PRIMARY_FRICTION_DATASETS)

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


def test_only_verified_paper_baselines_are_accepted():
    from safegrip.literature import PAPER_BASELINES, validate_paper_baselines
    assert PAPER_BASELINES == (
        "du2023_inceptiontime", "todorovic2022_cnn",
        "lampe2023_gru", "levenberg2023_stft",
    )
    for bad in ["direct_gru_control", "lampe2023_lstm", "schaefke2023_transformer", "chen2025_svdkl"]:
        try:
            validate_paper_baselines([bad])
        except ValueError:
            pass
        else:
            raise AssertionError(f"non-primary baseline accepted: {bad}")


def test_source_settings_protocol_rejects_nonreproducible_adaptations():
    from safegrip.literature import validate_source_settings
    validate_source_settings(["du2023_inceptiontime", "lampe2023_gru"])
    for name in ["todorovic2022_cnn", "levenberg2023_stft"]:
        try:
            validate_source_settings([name])
        except ValueError:
            pass
        else:
            raise AssertionError(f"source-settings incorrectly allowed for {name}")
