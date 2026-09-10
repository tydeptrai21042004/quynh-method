from safegrip.datasets import DATASET_REGISTRY
from safegrip.benchmark import PROPOSAL_VARIANTS


def test_dataset_registry_has_multiple_real_sources():
    expected={"lira","kuleuven","kit","deep_dynamics","comma2k19","extreme_road","bicycle_tire","mendeley_friction"}
    assert expected.issubset(DATASET_REGISTRY)
    for name in expected:
        assert DATASET_REGISTRY[name]["role"]
        assert DATASET_REGISTRY[name]["auto_download"]


def test_ablation_covers_safegrip_v2_components():
    expected={
        "safegrip_data_only","safegrip_static_only","safegrip_no_gate",
        "safegrip_no_bound","safegrip_no_uq","safegrip_no_calibration","safegrip",
    }
    assert expected.issubset(PROPOSAL_VARIANTS)


def test_no_excitation_ablation_is_registered():
    from safegrip.benchmark import PROPOSAL_VARIANTS, _proposal_flags
    assert "safegrip_no_excitation" in PROPOSAL_VARIANTS
    flags=_proposal_flags("safegrip_no_excitation")
    assert flags["raw_features_only"] is True
    assert flags["use_gate"] is False
    assert flags["use_excitation_regularizer"] is False
    assert flags["use_excitation_uq_inflation"] is False



def test_primary_ci_ablations_have_distinct_semantics():
    from safegrip.benchmark import PRIMARY_ABLATION_VARIANTS, _proposal_flags
    specs={v:tuple(sorted((k,str(val)) for k,val in _proposal_flags(v).items()
                          if k not in {"canonical_variant","raw_features_only","use_gate","use_excitation_regularizer","use_excitation_uq_inflation"}))
           for v in PRIMARY_ABLATION_VARIANTS}
    assert len(set(specs.values()))==len(PRIMARY_ABLATION_VARIANTS)


def test_full_ci_uses_raw_features_and_counterfactual_authority():
    from safegrip.benchmark import _proposal_flags
    f=_proposal_flags("safegrip")
    assert f["feature_mode"]=="raw"
    assert f["use_persistent_state"] is True
    assert f["use_identifiability"] is True
    assert f["use_acceptance"] is True
    assert f["use_excitation_proxy"] is False
    assert f["use_innovation_supervision"] is True


def test_excitation_proxy_is_explicit_comparator_not_full_method():
    from safegrip.benchmark import _proposal_flags
    proxy=_proposal_flags("safegrip_excitation_proxy")
    full=_proposal_flags("safegrip")
    assert proxy["use_excitation_proxy"] is True
    assert proxy["feature_mode"]=="raw"
    assert proxy["use_acceptance"] is True
    assert full["use_excitation_proxy"] is False


def test_v10_counterfactual_ablation_family_is_registered():
    from safegrip.benchmark import PROPOSAL_VARIANTS, _proposal_flags
    expected={
        "safegrip_single_scale_cf", "safegrip_no_linearity_consistency",
        "safegrip_no_agreement_veto", "safegrip_no_counterfactual_ranking",
        "safegrip_no_state_update_loss", "safegrip_no_direction_loss",
        "safegrip_no_dynamics_pretrain",
    }
    assert expected.issubset(PROPOSAL_VARIANTS)
    assert _proposal_flags("safegrip")["use_multiscale_counterfactual"] is True
    assert _proposal_flags("safegrip_single_scale_cf")["use_multiscale_counterfactual"] is False
    assert _proposal_flags("safegrip_no_agreement_veto")["use_agreement_veto"] is False
