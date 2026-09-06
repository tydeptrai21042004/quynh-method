import torch
from safegrip.models import make_literature_baseline, SafeGripNet, SafeGripNoTemporal, DeterministicTCN
from safegrip.literature import PAPER_BASELINES, LITERATURE_BASELINES, validate_paper_baselines


def test_literature_model_shapes():
    x=torch.randn(3,16,8)
    for n in ["todorovic2022_cnn","lampe2023_lstm","lampe2023_gru","schaefke2023_transformer"]:
        y=make_literature_baseline(n,8,debug_scale=True)(x)
        assert y.shape==(3,)


def test_proposal_shapes():
    x=torch.randn(3,16,8)
    mu,ls=SafeGripNet(8,hidden=16,blocks=2)(x)
    assert mu.shape==(3,) and ls.shape==(3,)
    mu,ls=SafeGripNoTemporal(8,hidden=16)(x)
    assert mu.shape==(3,) and ls.shape==(3,)
    assert DeterministicTCN(8,hidden=16,blocks=2)(x).shape==(3,)


def test_every_paper_baseline_has_real_citation():
    validate_paper_baselines(PAPER_BASELINES)
    for name in PAPER_BASELINES:
        assert LITERATURE_BASELINES[name]["doi"]
        assert LITERATURE_BASELINES[name]["title"]
        assert LITERATURE_BASELINES[name]["runnable"] is True
