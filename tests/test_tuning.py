class DummyTrial:
    def suggest_categorical(self,name,choices): return choices[0]
    def suggest_int(self,name,a,b): return a
    def suggest_float(self,name,a,b,log=False): return a


def test_proposal_hyperparameter_space_is_explicit():
    from safegrip.tuning import suggest_safegrip
    hp=suggest_safegrip(DummyTrial(),{"tuning":{"space":{}}})
    for k in ["sequence_length","hidden","tcn_blocks","kernel_size","dropout","lr","weight_decay","batch_size","lambda_mse","lambda_physics"]:
        assert k in hp
