import torch
from safegrip.models import make_literature_baseline, SafeGripNet, SafeGripNoTemporal, DeterministicTCN
from safegrip.literature import PAPER_BASELINES, LITERATURE_BASELINES, validate_paper_baselines
from safegrip.benchmark import literature_hparams


def test_literature_model_shapes():
    x=torch.randn(3,16,8)
    for n in ["todorovic2022_cnn","lampe2023_lstm","lampe2023_gru","schaefke2023_transformer"]:
        y=make_literature_baseline(n,8,sequence_length=16,debug_scale=True)(x)
        assert y.shape==(3,)


def test_todorovic_paper_architecture_materializes_source_shape():
    model=make_literature_baseline("todorovic2022_cnn",7,sequence_length=100,dropout=0.0)
    y=model(torch.randn(2,100,7))
    conv=[m for m in model.modules() if isinstance(m,torch.nn.Conv1d)]
    assert [(m.out_channels,m.kernel_size[0]) for m in conv]==[(128,14),(128,10),(256,10)]
    linear=[m for m in model.modules() if isinstance(m,torch.nn.Linear)]
    assert linear[0].in_features==3072 and linear[0].out_features==400
    assert y.shape==(2,)


def test_lampe_source_defaults_and_scaler():
    cfg={
        "sequence_length":64,
        "training":{"lr":1e-3,"weight_decay":1e-4,"batch_size":256,"dropout":.1,"patience":10,
                    "epochs_quick":3,"epochs_paper":60},
        "baseline":{
            "lampe2023_lstm":{"sequence_length":100,"scaler":"minmax","optimizer":"adam","lr":1e-3,
                              "weight_decay":1e-4,"batch_size":64,"dropout":0.0,"patience":500,"epochs":500}
        },
    }
    hp=literature_hparams("lampe2023_lstm",cfg,preset="paper")
    assert hp["scaler"]=="minmax" and hp["optimizer"]=="adam"
    assert hp["batch_size"]==64 and hp["epochs"]==500 and hp["dropout"]==0.0


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
        assert LITERATURE_BASELINES[name]["fidelity"]
        assert LITERATURE_BASELINES[name]["runnable"] is True
