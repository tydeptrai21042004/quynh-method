import pytest

from safegrip.datasets import DATASET_REGISTRY, PAPER_DATASETS, PRIMARY_FRICTION_DATASETS
from safegrip.literature import (
    PAPER_BASELINES, LITERATURE_BASELINES, DATASET_BASELINES,
    baselines_for_dataset, validate_paper_baselines,
)


def test_public_registry_is_closed_to_four_paper_datasets():
    assert PAPER_DATASETS == ("lira_cd", "uc3m_tire", "deep_dynamics_iac", "io_vnbd")
    assert tuple(DATASET_REGISTRY) == PAPER_DATASETS
    assert PRIMARY_FRICTION_DATASETS == ("lira_cd",)
    for old in ("mssp2023_friction", "kit", "kuleuven", "comma2k19"):
        assert old not in DATASET_REGISTRY


def test_dataset_scoped_paper_baselines_are_exactly_frozen_set():
    expected = {
        "lira_cd": ("du2023_inceptiontime", "levenberg2023_stft"),
        "uc3m_tire": ("mendoza2019_fuzzy", "yunta2018_fuzzy_lfc"),
        "deep_dynamics_iac": ("chrosniak2024_ddm", "fang_yu2025_fthd"),
        "io_vnbd": ("onyekpe2021_qgru", "wang2023_transformer"),
    }
    assert DATASET_BASELINES == expected
    assert set(PAPER_BASELINES) == {x for xs in expected.values() for x in xs}
    validate_paper_baselines(PAPER_BASELINES)
    for name in PAPER_BASELINES:
        assert LITERATURE_BASELINES[name]["doi"]
        assert LITERATURE_BASELINES[name]["dataset"] in expected


def test_cross_dataset_baseline_is_rejected():
    with pytest.raises(ValueError, match="not allowed"):
        validate_paper_baselines(["du2023_inceptiontime"], dataset="io_vnbd")


def test_d2_yunta_provenance_is_not_claimed_exact():
    assert LITERATURE_BASELINES["mendoza2019_fuzzy"]["exact_dataset"] is True
    assert LITERATURE_BASELINES["yunta2018_fuzzy_lfc"]["exact_dataset"] is False


def test_unimplemented_reproductions_cannot_be_claimed_runnable():
    validate_paper_baselines(baselines_for_dataset("lira_cd"), require_runnable=True)
    with pytest.raises(ValueError, match="not implemented"):
        validate_paper_baselines(baselines_for_dataset("io_vnbd"), require_runnable=True)


def test_levenberg_low_rate_limitation_remains_explicit():
    meta = LITERATURE_BASELINES["levenberg2023_stft"]
    assert "low-rate" in meta["fidelity"]
    assert "70--125 Hz" in meta["limitation"]
