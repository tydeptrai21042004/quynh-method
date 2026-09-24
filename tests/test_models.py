import torch
from safegrip.models import make_literature_baseline, SafeGripV2Net, ResidualScaleHead
from safegrip.literature import LITERATURE_BASELINES, PAPER_BASELINES, validate_paper_baselines
from safegrip.benchmark import literature_hparams


def test_primary_neural_paper_baseline_shapes():
    x=torch.randn(2,100,8)
    for name in ("du2023_inceptiontime","todorovic2022_cnn","lampe2023_gru"):
        model=make_literature_baseline(name,8,sequence_length=100,debug_scale=True)
        y=model(x)
        assert y.shape==(2,)


def test_paper_hparams_preserve_du_and_lampe_source_settings():
    cfg={
        "sequence_length":64,
        "training":{"lr":1e-3,"weight_decay":1e-4,"batch_size":256,"dropout":.1,"patience":10,
                    "epochs_quick":3,"epochs_paper":60},
        "baseline":{
            "du2023_inceptiontime":{"sequence_length":100,"scaler":"standard","optimizer":"sgd","lr":1e-3,
                                      "momentum":0.98,"weight_decay":0.0,"batch_size":128,"dropout":0.0,"patience":20,"epochs":200},
            "lampe2023_gru":{"sequence_length":100,"scaler":"minmax","optimizer":"adam","lr":1e-3,
                              "weight_decay":1e-4,"batch_size":64,"dropout":0.0,"patience":500,"epochs":500}
        },
    }
    du=literature_hparams("du2023_inceptiontime",cfg,preset="paper")
    gru=literature_hparams("lampe2023_gru",cfg,preset="paper")
    assert du["optimizer"]=="sgd" and du["momentum"]==0.98
    assert gru["scaler"]=="minmax" and gru["optimizer"]=="adam"
    assert gru["batch_size"]==64 and gru["epochs"]==500 and gru["dropout"]==0.0


def test_levenberg_low_rate_stft_adaptation_is_finite_and_positive_slope():
    import numpy as np
    from types import SimpleNamespace
    from sklearn.preprocessing import StandardScaler
    from safegrip.levenberg import fit_levenberg2023_stft

    rng=np.random.default_rng(4); n=40; t=20
    raw=rng.normal(size=(n,t,3)).astype(np.float32)
    amp=np.linspace(0.5,2.0,n).astype(np.float32)
    phase=np.linspace(0,2*np.pi,t,endpoint=False)
    raw[:,:,1]+=amp[:,None]*np.sin(2*phase)[None,:]
    scaler=StandardScaler().fit(raw.reshape(-1,3))
    X=scaler.transform(raw.reshape(-1,3)).reshape(n,t,3).astype(np.float32)
    y=(0.6+0.04*np.log(np.maximum(amp,1e-6))).astype(np.float32)
    b=SimpleNamespace(features=["ax","ay","yaw_rate"],scaler=scaler,Xtr=X,ytr=y)
    model=fit_levenberg2023_stft(b,{"lira":{"resample_hz":20.0}})
    pred=model.predict(X)
    assert pred.shape==(n,) and np.isfinite(pred).all()
    assert model.slope>=0.0 and 0.0<model.frequency_hz<=10.0


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


def test_safegrip_v12_outputs_selective_physics_components_and_bound():
    from safegrip.models import SafeGripV4Net
    torch.manual_seed(4)
    model=SafeGripV4Net(6,hidden=24,gru_hidden=16,conv_channels=16,dropout=0.0,dynamics_indices=[0,1])
    x=torch.randn(5,12,6); lo=torch.linspace(0.05,0.25,5)
    d=model.forward_details(x,lo,1.3)
    required={"prediction","base_prediction","physics_gate","identifiability","physics_correction",
              "applied_physics_correction","change_probability","aleatoric_scale","features"}
    assert required <= set(d)
    assert torch.all(d["prediction"]>=lo-1e-7)
    assert torch.all(d["prediction"]<=1.3+1e-7)
    assert torch.all((d["physics_gate"]>=0)&(d["physics_gate"]<=1))
    assert torch.all((d["identifiability"]>=0)&(d["identifiability"]<=1))
    assert torch.all((d["change_probability"]>=0)&(d["change_probability"]<=1))
    assert torch.all(d["aleatoric_scale"]>0)


def test_safegrip_v12_zero_dynamics_sensitivity_cannot_change_base_prediction():
    from safegrip.models import SafeGripV4Net
    torch.manual_seed(5)
    model=SafeGripV4Net(5,hidden=24,gru_hidden=12,conv_channels=16,dropout=0.0,dynamics_indices=[0,1])
    for p in model.dynamics_head.parameters():
        torch.nn.init.zeros_(p)
    x=torch.randn(4,10,5); lo=torch.zeros(4)
    d=model.forward_details(x,lo,1.3)
    assert torch.allclose(d["identifiability"],torch.zeros_like(d["identifiability"]),atol=1e-8)
    assert torch.allclose(d["physics_correction"],torch.zeros_like(d["physics_correction"]),atol=1e-8)
    assert torch.allclose(d["prediction"],d["base_prediction"],atol=1e-7)


def test_safegrip_v12_unconditional_physics_ablation_uses_full_correction():
    from safegrip.models import SafeGripV4Net
    torch.manual_seed(6)
    model=SafeGripV4Net(4,hidden=16,gru_hidden=8,conv_channels=8,dropout=0.0,dynamics_indices=[0],
                        use_utility_gate=False,use_physics_correction=True,use_bound=False)
    x=torch.randn(3,8,4)
    d=model.forward_details(x,None,1.3)
    assert torch.allclose(d["physics_gate"],torch.ones_like(d["physics_gate"]))
    expected=torch.clamp(d["base_prediction"]+model.physics_correction_scale*d["physics_correction"],0.0,1.3)
    assert torch.allclose(d["prediction"],expected,atol=1e-6)


def test_safegrip_v12_physics_step_is_bounded():
    from safegrip.models import SafeGripV4Net
    torch.manual_seed(7)
    max_step=0.07
    model=SafeGripV4Net(5,hidden=16,gru_hidden=8,conv_channels=8,dropout=0.0,dynamics_indices=[0,1],
                        inverse_dynamics_max_step=max_step)
    d=model.forward_details(torch.randn(8,10,5),torch.zeros(8),1.3)
    assert torch.all(torch.abs(d["physics_correction"])<=max_step+1e-7)


def test_safegrip_v12_no_aleatoric_feature_is_deterministic_constant():
    from safegrip.models import SafeGripV4Net
    torch.manual_seed(8)
    floor=0.007
    model=SafeGripV4Net(5,hidden=16,gru_hidden=8,conv_channels=8,dropout=0.0,dynamics_indices=[0],
                        use_aleatoric_feature=False,aleatoric_floor=floor)
    d=model.forward_details(torch.randn(6,9,5),torch.zeros(6),1.3)
    assert torch.allclose(d["aleatoric_scale"],torch.full_like(d["aleatoric_scale"],floor),atol=1e-8)


def test_safegrip_v12_raw_inverse_candidate_is_pre_projection_correction():
    from safegrip.models import SafeGripV4Net
    torch.manual_seed(9)
    model=SafeGripV4Net(4,hidden=16,gru_hidden=8,conv_channels=8,dropout=0.0,dynamics_indices=[0],
                        use_bound=True,physics_correction_scale=0.5)
    x=torch.randn(4,8,4); lo=torch.full((4,),0.8)
    d=model.forward_details(x,lo,1.3)
    expected=d["base_prediction"]+0.5*d["physics_correction"]
    assert torch.allclose(d["inverse_candidate_raw"],expected,atol=1e-7)
    assert torch.all(d["inverse_candidate_prediction"]>=lo-1e-7)


def test_safegrip_v13_energy_landscape_outputs_two_stage_selector():
    from safegrip.models import SafeGripV5Net
    torch.manual_seed(10)
    model=SafeGripV5Net(6,hidden=24,gru_hidden=16,conv_channels=16,dropout=0.0,
                        dynamics_indices=[0,1],energy_grid_points=7,energy_grid_radius=0.1)
    x=torch.randn(5,12,6); lo=torch.linspace(0.05,0.25,5)
    d=model.forward_details(x,lo,1.3)
    required={"prediction","base_prediction","benefit_probability","correction_fraction",
              "physics_gate","posterior_entropy","energy_curvature","normalized_improvement",
              "identifiability","physics_correction"}
    assert required <= set(d)
    assert torch.all((d["benefit_probability"]>=0)&(d["benefit_probability"]<=1))
    assert torch.all((d["correction_fraction"]>=0)&(d["correction_fraction"]<=1))
    assert torch.allclose(d["physics_gate"],d["benefit_probability"]*d["correction_fraction"],atol=1e-7)
    assert torch.all((d["identifiability"]>=0)&(d["identifiability"]<=1))
    assert torch.all((d["posterior_entropy"]>=0)&(d["posterior_entropy"]<=1+1e-6))
    assert torch.all(d["prediction"]>=lo-1e-7)
    assert torch.all(d["prediction"]<=1.3+1e-7)


def test_safegrip_v13_flat_energy_landscape_has_zero_identifiability_and_no_step():
    from safegrip.models import SafeGripV5Net
    torch.manual_seed(11)
    model=SafeGripV5Net(5,hidden=20,gru_hidden=12,conv_channels=12,dropout=0.0,
                        dynamics_indices=[0,1])
    # Zeroing the dynamics head makes every friction hypothesis identical.
    for p in model.dynamics_head.parameters():
        torch.nn.init.zeros_(p)
    x=torch.randn(4,10,5); lo=torch.zeros(4)
    d=model.forward_details(x,lo,1.3)
    assert torch.allclose(d["identifiability"],torch.zeros_like(d["identifiability"]),atol=1e-6)
    assert torch.allclose(d["physics_correction"],torch.zeros_like(d["physics_correction"]),atol=1e-6)
    assert torch.allclose(d["prediction"],d["base_prediction"],atol=1e-6)


def test_safegrip_v13_unconditional_ablation_applies_full_energy_candidate():
    from safegrip.models import SafeGripV5Net
    torch.manual_seed(12)
    model=SafeGripV5Net(4,hidden=16,gru_hidden=8,conv_channels=8,dropout=0.0,
                        dynamics_indices=[0],use_utility_gate=False,use_bound=False,
                        physics_correction_scale=0.75)
    d=model.forward_details(torch.randn(3,9,4),None,1.3)
    assert torch.allclose(d["benefit_probability"],torch.ones_like(d["benefit_probability"]))
    assert torch.allclose(d["correction_fraction"],torch.ones_like(d["correction_fraction"]))
    expected=torch.clamp(d["base_prediction"]+0.75*d["physics_correction"],0.0,1.3)
    assert torch.allclose(d["prediction"],expected,atol=1e-6)


def test_safegrip_v13_contrastive_dynamics_loss_is_finite_and_differentiable():
    from safegrip.models import SafeGripV5Net
    torch.manual_seed(13)
    model=SafeGripV5Net(5,hidden=20,gru_hidden=12,conv_channels=12,dropout=0.0,
                        dynamics_indices=[0,1])
    x=torch.randn(6,10,5); y=torch.rand(6)*1.2
    loss,stats=model.counterfactual_dynamics_loss(x,y,1.3,temperature=0.25,negatives=6)
    assert torch.isfinite(loss)
    assert 0.0 <= float(stats["ranking_accuracy"]) <= 1.0
    loss.backward()
    grads=[p.grad for p in model.dynamics_head.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_safegrip_v13_flat_energy_is_neutral_near_upper_boundary():
    """Clipped hypothesis grids must not create a fake inward correction."""
    from safegrip.models import SafeGripV5Net
    torch.manual_seed(14)
    model=SafeGripV5Net(4,hidden=16,gru_hidden=8,conv_channels=8,dropout=0.0,
                        dynamics_indices=[0],energy_grid_points=7,energy_grid_radius=0.2)
    # Make the direct estimate very close to mu_upper and make all energy
    # hypotheses identical.  A direct average of clipped grid values would be
    # biased inward here; the v1.3 posterior-offset construction stays neutral.
    for p in model.base_head.parameters():
        torch.nn.init.zeros_(p)
    torch.nn.init.constant_(model.base_head[-1].bias,8.0)
    for p in model.dynamics_head.parameters():
        torch.nn.init.zeros_(p)
    d=model.forward_details(torch.randn(5,9,4),torch.zeros(5),1.3)
    assert torch.all(d["base_prediction"]>1.29)
    assert torch.allclose(d["identifiability"],torch.zeros_like(d["identifiability"]),atol=1e-6)
    assert torch.allclose(d["physics_correction"],torch.zeros_like(d["physics_correction"]),atol=1e-6)


def test_safegrip_v13_no_identifiability_does_not_leak_entropy_into_selector():
    """The no-identifiability ablation must remove both I and its entropy complement."""
    from safegrip.models import SafeGripV5Net
    import types
    torch.manual_seed(15)
    model=SafeGripV5Net(5,hidden=20,gru_hidden=12,conv_channels=12,dropout=0.0,
                        dynamics_indices=[0,1],use_identifiability_feature=False)
    model.eval()
    x=torch.randn(6,10,5); lo=torch.zeros(6)
    with torch.no_grad():
        ref=model.forward_details(x,lo,1.3)["benefit_probability"].clone()
    original=model._energy_landscape
    def altered(self,x_,base_,upper_):
        out=original(x_,base_,upper_)
        out["identifiability"]=torch.ones_like(out["identifiability"])
        out["posterior_entropy"]=torch.zeros_like(out["posterior_entropy"])
        return out
    model._energy_landscape=types.MethodType(altered,model)
    with torch.no_grad():
        changed=model.forward_details(x,lo,1.3)["benefit_probability"]
    assert torch.allclose(ref,changed,atol=1e-7)


def test_safegrip_v14_uses_single_continuous_controller():
    from safegrip.models import SafeGripV6Net
    torch.manual_seed(21)
    model=SafeGripV6Net(6,hidden=24,gru_hidden=16,dropout=0.0,dynamics_indices=[0,1])
    model.set_target_stats(0.7,0.1)
    d=model.forward_details(torch.randn(5,12,6),torch.zeros(5),1.3)
    assert torch.allclose(d["physics_gate"],d["correction_fraction"],atol=1e-7)
    # The diagnostic benefit probability must not silently re-enter the gate.
    product=d["benefit_probability"]*d["correction_fraction"]
    assert not torch.allclose(d["physics_gate"],product,atol=1e-5)


def test_safegrip_v14_dynamics_target_is_endpoint_innovation():
    from safegrip.models import SafeGripV6Net
    torch.manual_seed(22)
    model=SafeGripV6Net(4,hidden=16,gru_hidden=8,dropout=0.0,dynamics_indices=[0,2])
    x=torch.randn(3,7,4); mu=torch.full((3,),0.7)
    _,target=model.dynamics_prediction(x,mu,1.3)
    expected=x[:,-1,[0,2]]-x[:,-2,[0,2]]
    assert torch.allclose(target,expected,atol=1e-7)


def test_safegrip_v14_flat_energy_is_boundary_neutral():
    from safegrip.models import SafeGripV6Net
    torch.manual_seed(23)
    model=SafeGripV6Net(5,hidden=20,gru_hidden=12,dropout=0.0,dynamics_indices=[0,1])
    model.set_target_stats(1.299,0.001)
    for p in model.dynamics_head.parameters():
        torch.nn.init.zeros_(p)
    x=torch.randn(4,10,5)
    with torch.no_grad():
        h=model.encode(x); base,_=model._base_prediction(h,1.3)
        e=model._energy_landscape(x,base,1.3)
    assert torch.allclose(e["correction"],torch.zeros_like(e["correction"]),atol=1e-7)
    assert torch.allclose(e["identifiability"],torch.zeros_like(e["identifiability"]),atol=1e-7)


def test_safegrip_v14_target_standardization_maps_back_to_physical_units():
    from safegrip.models import SafeGripV6Net
    torch.manual_seed(24)
    model=SafeGripV6Net(3,hidden=12,gru_hidden=8,dropout=0.0,dynamics_indices=[0])
    model.set_target_stats(0.72,0.08)
    for p in model.base_head.parameters():
        torch.nn.init.zeros_(p)
    x=torch.randn(2,6,3)
    with torch.no_grad():
        base,z=model._base_prediction(model.encode(x),1.3)
    assert torch.allclose(z,torch.zeros_like(z),atol=1e-7)
    assert torch.allclose(base,torch.full_like(base,0.72),atol=1e-6)


def test_frc_models_shapes_and_mu_conditioning():
    from safegrip.models import FrictionResponseNet, DirectGRUControl
    import torch
    response = FrictionResponseNet(6, 3, hidden=16, gru_layers=1, dropout=0.0, mu_upper=1.3)
    x = torch.randn(5, 8, 6)
    mu1 = torch.full((5,), 0.3)
    mu2 = torch.full((5,), 0.9)
    y1 = response(x, mu1); y2 = response(x, mu2)
    assert y1.shape == (5,3)
    assert y2.shape == (5,3)
    # The friction hypothesis is a real model input, not an unused config knob.
    assert not torch.allclose(y1, y2)
    direct = DirectGRUControl(6, hidden=16, gru_layers=1, dropout=0.0)
    assert direct(x).shape == (5,)


def test_pfr_excitation_gru_attention_favors_more_excited_timesteps():
    import torch
    from safegrip.models import PFRExcitationGRU

    model = PFRExcitationGRU(d=3, excitation_index=2, hidden=8, dropout=0.0, attention_gamma=4.0)
    x = torch.zeros(2, 4, 3)
    x[0, :, 2] = torch.tensor([0.0, 0.1, 0.2, 0.9])
    x[1, :, 2] = 0.25
    _, scale, weights = model(x, return_attention=True)
    assert weights.shape == (2, 4)
    assert torch.allclose(weights.sum(dim=1), torch.ones(2), atol=1e-6)
    assert int(torch.argmax(weights[0]).item()) == 3
    assert torch.allclose(weights[1], torch.full((4,), 0.25), atol=1e-6)
    assert torch.all(scale > 0)


def test_pfr_excitation_gru_gamma_zero_is_uniform_pooling():
    import torch
    from safegrip.models import PFRExcitationGRU

    model = PFRExcitationGRU(d=2, excitation_index=1, hidden=8, dropout=0.0, attention_gamma=0.0)
    x = torch.randn(3, 5, 2)
    weights = model.attention_weights(x)
    assert torch.allclose(weights, torch.full_like(weights, 0.2), atol=1e-7)
