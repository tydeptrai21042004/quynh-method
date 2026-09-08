import torch
from safegrip.models import make_literature_baseline, SafeGripV2Net, ResidualScaleHead
from safegrip.literature import LITERATURE_BASELINES, PAPER_BASELINES, validate_paper_baselines
from safegrip.benchmark import literature_hparams


def test_literature_model_shapes():
    x=torch.randn(2,100,8)
    for name in ("todorovic2022_cnn","lampe2023_lstm","lampe2023_gru"):
        model=make_literature_baseline(name,8,sequence_length=100,debug_scale=True)
        y=model(x)
        assert y.shape==(2,)


def test_transformer_shape():
    x=torch.randn(2,16,8)
    model=make_literature_baseline("schaefke2023_transformer",8,sequence_length=16,debug_scale=True,hidden=32,heads=4)
    assert model(x).shape==(2,)


def test_lampe_paper_hparams_preserve_source_preprocessing():
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


def test_safegrip_v2_is_bound_parameterized():
    x=torch.randn(3,16,8)
    lo=torch.tensor([0.1,0.4,0.8])
    model=SafeGripV2Net(8,excitation_index=2,hidden=16,gru_hidden=8,dropout=0.0)
    p,z,g=model(x,lo,1.3)
    assert p.shape==(3,) and z.shape==(3,) and g.shape==(3,)
    assert torch.all(p>=lo-1e-7)
    assert torch.all(p<=1.3+1e-7)
    assert torch.all((g>=0)&(g<=1))


def test_safegrip_v2_global_support_without_lower_bound():
    x=torch.randn(3,8,5)
    model=SafeGripV2Net(5,excitation_index=None,hidden=16,gru_hidden=8,use_gate=False,use_bound=False)
    p,_,_=model(x,None,1.3)
    assert torch.all(p>=0) and torch.all(p<=1.3)


def test_residual_scale_positive():
    h=torch.randn(4,16)
    s=ResidualScaleHead(16,floor=0.01)(h)
    assert s.shape==(4,)
    assert torch.all(s>=0.01)


def test_every_paper_baseline_has_real_citation():
    validate_paper_baselines(PAPER_BASELINES)
    for name in PAPER_BASELINES:
        assert LITERATURE_BASELINES[name]["doi"]
        assert LITERATURE_BASELINES[name]["title"]
        assert LITERATURE_BASELINES[name]["fidelity"]
        assert LITERATURE_BASELINES[name]["runnable"] is True
