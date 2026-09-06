class DummyTrial:
    number=0
    def suggest_categorical(self,name,choices): return choices[0]
    def suggest_int(self,name,a,b): return a
    def suggest_float(self,name,a,b,log=False): return a


def test_proposal_hyperparameter_space_is_explicit():
    from safegrip.tuning import suggest_safegrip
    hp=suggest_safegrip(DummyTrial(),{"tuning":{"space":{}}})
    for k in ["sequence_length","hidden","tcn_blocks","kernel_size","dropout","lr","weight_decay","batch_size","lambda_mse","lambda_physics"]:
        assert k in hp


def test_baseline_search_keeps_source_constraints():
    from safegrip.tuning import suggest_literature
    cfg={"baseline_tuning":{"sequence_length":[16,32,64,100,128],"space":{}}}
    tod=suggest_literature(DummyTrial(),cfg,"todorovic2022_cnn")
    lstm=suggest_literature(DummyTrial(),cfg,"lampe2023_lstm")
    assert tod["sequence_length"]==100
    assert lstm["dropout"]==0.0
