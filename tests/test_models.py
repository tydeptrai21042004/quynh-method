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


def test_safegrip_ci_legacy_excitation_proxy_is_monotone_only_for_proxy_ablation():
    from safegrip.models import SafeGripV3Net
    model=SafeGripV3Net(6,excitation_index=1,hidden=16,gru_hidden=8,dropout=0.0,
                        gate_init_slope=5.0,gate_init_threshold=0.3)
    e=torch.linspace(0,1,21)
    r=model.reliability(e)
    assert torch.all(r[1:]>=r[:-1]-1e-8)
    assert float(r[-1].detach())>float(r[0].detach())


def test_safegrip_ci_reports_counterfactual_components_and_respects_bound():
    from safegrip.models import SafeGripV3Net
    x=torch.randn(4,16,7)
    lo=torch.tensor([0.1,0.2,0.3,0.4])
    model=SafeGripV3Net(7,hidden=16,gru_hidden=8,dropout=0.0,dynamics_indices=[0,1,2])
    d=model.forward_details(x,lo,1.3)
    required={"prediction","prior_prediction","candidate_prediction","innovation","authority",
              "identifiability","acceptance","information_raw","latent"}
    assert required <= set(d)
    assert d["prediction"].shape==(4,)
    assert torch.all(d["prediction"]>=lo-1e-7)
    assert torch.all(d["prediction"]<=1.3+1e-7)
    assert torch.all((d["authority"]>=0)&(d["authority"]<=1))
    assert torch.all((d["identifiability"]>=0)&(d["identifiability"]<=1))
    assert torch.all((d["acceptance"]>=0)&(d["acceptance"]<=1))


def test_safegrip_ci_zero_dynamics_sensitivity_blocks_update():
    from safegrip.models import SafeGripV3Net
    torch.manual_seed(0)
    model=SafeGripV3Net(5,hidden=16,gru_hidden=8,dropout=0.0,dynamics_indices=[0,1])
    for p in model.dynamics_head.parameters():
        torch.nn.init.zeros_(p)
    x=torch.randn(3,12,5); lo=torch.tensor([0.1,0.1,0.1])
    d=model.forward_details(x,lo,1.3)
    assert torch.allclose(d["information_raw"],torch.zeros_like(d["information_raw"]),atol=1e-8)
    assert torch.allclose(d["identifiability"],torch.zeros_like(d["identifiability"]),atol=1e-8)
    assert torch.allclose(d["authority"],torch.zeros_like(d["authority"]),atol=1e-8)
    assert torch.allclose(d["prediction"],d["prior_prediction"],atol=1e-7)


def test_safegrip_ci_persistent_prior_is_used_when_supplied():
    from safegrip.models import SafeGripV3Net
    model=SafeGripV3Net(5,hidden=16,gru_hidden=8,dropout=0.0,use_innovation=False)
    x=torch.randn(2,10,5); lo=torch.tensor([0.1,0.2]); prior=torch.tensor([0.55,0.75])
    mask=torch.tensor([True,True])
    d_state=model.forward_details(x,lo,1.3,prior_mu=prior,prior_mask=mask)
    d_ctx=model.forward_details(x,lo,1.3)
    expected=model.state_persistence*prior+(1.0-model.state_persistence)*d_ctx["context_prior_prediction"]
    assert torch.allclose(d_state["prior_prediction"],expected,atol=2e-5)
    assert torch.all(d_state["persistent_state_used"]==1)


def test_residual_scale_can_initialize_near_error_scale():
    from safegrip.models import ResidualScaleHead
    h=torch.zeros(5,16)
    head=ResidualScaleHead(16,floor=0.005,initial_scale=0.04)
    s=head(h)
    assert torch.allclose(s,torch.full_like(s,0.04),atol=1e-5)


def test_static_ablation_is_endpoint_only():
    from safegrip.models import SafeGripV3Net
    torch.manual_seed(0)
    model=SafeGripV3Net(4,excitation_index=None,hidden=16,gru_hidden=8,dropout=0.0,
                        use_temporal=False,use_gate=False,use_innovation=False)
    model.eval()
    x1=torch.randn(2,8,4); x2=x1.clone(); x2[:,:-1,:]=torch.randn_like(x2[:,:-1,:])*100.0
    lo=torch.tensor([0.1,0.2])
    with torch.no_grad():
        p1,_,_=model(x1,lo,1.3); p2,_,_=model(x2,lo,1.3)
    assert torch.allclose(p1,p2,atol=1e-7)


def test_safegrip_ci_neutral_counterfactual_evidence_does_not_halve_authority():
    from safegrip.models import SafeGripV3Net
    import types
    model=SafeGripV3Net(
        3, hidden=16, gru_hidden=8, dropout=0.0, dynamics_indices=[0],
        acceptance_temperature=12.0, acceptance_tolerance=0.10, acceptance_strength=0.35,
    )
    # Deterministic local dynamics: G(mu)=mu.  Candidate == prior gives exactly
    # neutral residual improvement, which should now preserve most authority.
    def dyn(self, context_h, mu, upper=1.3):
        return mu.reshape(-1,1)
    model.dynamics_prediction_from_context=types.MethodType(dyn,model)
    x=torch.zeros(2,6,3); x[:,-1,0]=0.5
    h=torch.zeros(2,model.hidden); prior=torch.tensor([0.5,0.5]); cand=prior.clone()
    out=model._counterfactual_authority(x,h,prior,cand,1.3)
    acceptance=out[2]; veto=out[7]
    assert torch.all(acceptance>0.85)
    assert torch.all(veto<0.5)


def test_safegrip_ci_counterfactual_veto_reacts_asymmetrically_to_harm():
    from safegrip.models import SafeGripV3Net
    import types
    model=SafeGripV3Net(
        3, hidden=16, gru_hidden=8, dropout=0.0, dynamics_indices=[0],
        acceptance_temperature=12.0, acceptance_tolerance=0.10, acceptance_strength=0.35,
    )
    def dyn(self, context_h, mu, upper=1.3):
        return mu.reshape(-1,1)
    model.dynamics_prediction_from_context=types.MethodType(dyn,model)
    x=torch.zeros(1,6,3); x[:,-1,0]=0.5
    h=torch.zeros(1,model.hidden); prior=torch.tensor([0.5])
    neutral=model._counterfactual_authority(x,h,prior,prior,1.3)
    harmful=model._counterfactual_authority(x,h,prior,torch.tensor([1.0]),1.3)
    assert float(harmful[7])>float(neutral[7])
    assert float(harmful[2])<float(neutral[2])


def test_safegrip_ci_inverse_dynamics_correction_has_observed_direction():
    from safegrip.models import SafeGripV3Net
    import types
    model=SafeGripV3Net(
        3, hidden=16, gru_hidden=8, dropout=0.0, dynamics_indices=[0],
        inverse_dynamics_ridge=1e-4, inverse_dynamics_max_step=0.12,
    )
    def dyn(self, context_h, mu, upper=1.3):
        return mu.reshape(-1,1)
    model.dynamics_prediction_from_context=types.MethodType(dyn,model)
    x=torch.zeros(1,6,3); x[:,-1,0]=0.7
    h=torch.zeros(1,model.hidden); prior=torch.tensor([0.5]); cand=torch.tensor([0.55])
    out=model._counterfactual_authority(x,h,prior,cand,1.3)
    cf_delta=out[8]; agreement=out[9]
    assert float(cf_delta)>0.0
    assert float(cf_delta)<=0.120001
    assert 0.5<float(agreement)<=1.0


def test_safegrip_ci_multiscale_counterfactual_is_consistent_for_linear_dynamics():
    from safegrip.models import SafeGripV3Net
    import types
    model=SafeGripV3Net(
        3, hidden=16, gru_hidden=8, dropout=0.0, dynamics_indices=[0],
        counterfactual_delta=0.08, counterfactual_scale_span=2.0,
        use_multiscale_counterfactual=True, use_linearity_consistency=True,
    )
    def dyn(self, context_h, mu, upper=1.3):
        return mu.reshape(-1,1)
    model.dynamics_prediction_from_context=types.MethodType(dyn,model)
    x=torch.zeros(2,6,3); x[:,-1,0]=0.6
    h=torch.zeros(2,model.hidden); prior=torch.tensor([0.5,0.7]); cand=prior.clone()
    out=model._counterfactual_authority(x,h,prior,cand,1.3)
    info_cv, linearity = out[10], out[11]
    assert torch.all(info_cv < 1e-5)
    assert torch.all(linearity > 0.9999)


def test_safegrip_ci_multiscale_linearity_discount_detects_curvature():
    from safegrip.models import SafeGripV3Net
    import types
    model=SafeGripV3Net(
        3, hidden=16, gru_hidden=8, dropout=0.0, dynamics_indices=[0],
        counterfactual_delta=0.15, counterfactual_scale_span=2.5,
        use_multiscale_counterfactual=True, use_linearity_consistency=True,
        linearity_penalty=1.0,
    )
    def dyn(self, context_h, mu, upper=1.3):
        return (mu**3).reshape(-1,1)
    model.dynamics_prediction_from_context=types.MethodType(dyn,model)
    x=torch.zeros(1,6,3); x[:,-1,0]=0.3
    h=torch.zeros(1,model.hidden); prior=torch.tensor([0.45]); cand=torch.tensor([0.50])
    out=model._counterfactual_authority(x,h,prior,cand,1.3)
    assert float(out[10]) > 0.0
    assert float(out[11]) < 1.0


def test_safegrip_ci_agreement_veto_attenuates_opposite_inverse_update():
    from safegrip.models import SafeGripV3Net
    import types
    common=dict(
        d=3, hidden=16, gru_hidden=8, dropout=0.0, dynamics_indices=[0],
        acceptance_temperature=12.0, acceptance_tolerance=0.10, acceptance_strength=0.20,
        agreement_temperature=12.0, agreement_threshold=0.40, agreement_strength=0.50,
    )
    with_veto=SafeGripV3Net(**common, use_agreement_veto=True)
    no_veto=SafeGripV3Net(**common, use_agreement_veto=False)
    def dyn(self, context_h, mu, upper=1.3):
        return mu.reshape(-1,1)
    with_veto.dynamics_prediction_from_context=types.MethodType(dyn,with_veto)
    no_veto.dynamics_prediction_from_context=types.MethodType(dyn,no_veto)
    x=torch.zeros(1,6,3); x[:,-1,0]=0.75
    h=torch.zeros(1,with_veto.hidden); prior=torch.tensor([0.50]); bad_candidate=torch.tensor([0.35])
    a=with_veto._counterfactual_authority(x,h,prior,bad_candidate,1.3)
    b=no_veto._counterfactual_authority(x,h,prior,bad_candidate,1.3)
    assert float(a[9]) < 0.1  # learned and inverse-dynamics corrections disagree
    assert float(a[12]) > 0.5
    assert float(a[2]) < float(b[2])


def test_safegrip_v11_dual_expert_weights_are_valid_and_bounded():
    from safegrip.models import SafeGripV3Net
    model=SafeGripV3Net(5,hidden=16,gru_hidden=8,dropout=0.0,dynamics_indices=[0,1],
        use_dual_expert=True,use_learned_arbitration=True,use_inverse_expert=True,
        use_adaptive_persistence=True,use_split_innovation=True,
        use_multiscale_counterfactual=False,use_linearity_consistency=False,use_agreement_veto=False)
    x=torch.randn(4,10,5); lo=torch.full((4,),0.1); prior=torch.full((4,),0.6)
    d=model.forward_details(x,lo,1.3,prior_mu=prior)
    assert d["expert_weights"].shape==(4,3)
    assert torch.allclose(d["expert_weights"].sum(-1),torch.ones(4),atol=1e-6)
    assert torch.all((d["expert_weights"]>=0)&(d["expert_weights"]<=1))
    assert torch.all(d["prediction"]>=lo-1e-7) and torch.all(d["prediction"]<=1.3+1e-7)


def test_safegrip_v11_inverse_expert_is_unavailable_without_observability():
    from safegrip.models import SafeGripV3Net
    model=SafeGripV3Net(4,hidden=16,gru_hidden=8,dropout=0.0,dynamics_indices=[0],
        use_dual_expert=True,use_learned_arbitration=True,use_inverse_expert=True,
        use_adaptive_persistence=True,use_split_innovation=True)
    for p in model.dynamics_head.parameters():
        torch.nn.init.zeros_(p)
    x=torch.randn(3,8,4); lo=torch.full((3,),0.1); prior=torch.full((3,),0.6)
    d=model.forward_details(x,lo,1.3,prior_mu=prior)
    assert torch.allclose(d["identifiability"],torch.zeros_like(d["identifiability"]),atol=1e-8)
    assert torch.allclose(d["expert_weight_inverse"],torch.zeros_like(d["expert_weight_inverse"]),atol=1e-8)


def test_safegrip_v11_adaptive_persistence_stays_in_configured_range():
    from safegrip.models import SafeGripV3Net
    model=SafeGripV3Net(4,hidden=16,gru_hidden=8,dropout=0.0,
        use_adaptive_persistence=True,persistence_min=0.1,persistence_max=0.9)
    x=torch.randn(5,8,4); lo=torch.full((5,),0.1); prior=torch.full((5,),0.5)
    d=model.forward_details(x,lo,1.3,prior_mu=prior)
    rho=d["adaptive_persistence"]
    assert torch.all(rho>=0.1-1e-6) and torch.all(rho<=0.9+1e-6)
