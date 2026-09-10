class DummyTrial:
    number=0
    def suggest_categorical(self,name,choices): return choices[0]
    def suggest_int(self,name,a,b): return a
    def suggest_float(self,name,a,b,log=False): return a


def test_proposal_hyperparameter_space_is_explicit():
    from safegrip.tuning import suggest_safegrip
    hp=suggest_safegrip(DummyTrial(),{"tuning":{"space":{}}})
    for k in ["sequence_length","hidden","gru_hidden","dropout","lr","weight_decay","batch_size","huber_beta",
              "state_persistence","counterfactual_delta","counterfactual_scale_span","identifiability_lambda",
              "linearity_penalty","acceptance_temperature","acceptance_margin","agreement_strength",
              "inverse_dynamics_ridge","inverse_dynamics_max_step","innovation_loss_weight",
              "state_update_loss_weight","candidate_loss_weight","cf_agreement_loss_weight","direction_loss_weight",
              "dynamics_loss_weight","counterfactual_loss_weight","counterfactual_margin",
              "dynamics_pretrain_epochs","do_no_harm_weight","information_beta"]:
        assert k in hp


def test_baseline_search_keeps_source_constraints():
    from safegrip.tuning import suggest_literature
    cfg={"baseline_tuning":{"sequence_length":[16,32,64,100,128],"space":{}}}
    tod=suggest_literature(DummyTrial(),cfg,"todorovic2022_cnn")
    lstm=suggest_literature(DummyTrial(),cfg,"lampe2023_lstm")
    assert tod["sequence_length"]==100
    assert lstm["dropout"]==0.0


def test_global_tuning_eval_start_includes_all_search_spaces():
    from safegrip.benchmark import tuning_eval_start
    cfg={
        "sequence_length":16,
        "tuning":{"space":{"sequence_length":[8,16,64]}},
        "baseline_tuning":{"sequence_length":[16,128],"space":{"x":{"sequence_length":[100]}}},
        "baseline":{"foo":{"sequence_length":100}},
    }
    assert tuning_eval_start(cfg)==127
    assert tuning_eval_start(cfg,include_baselines=False)==127
