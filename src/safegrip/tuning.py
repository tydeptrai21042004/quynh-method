from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

from .benchmark import (
    make_bundle, common_eval_start, tuning_eval_start, fit_proposal, predict_proposal, predict_proposal_details, regression_metrics,
    fit_literature, predict_literature, literature_hparams,
)
from .literature import PAPER_BASELINES, validate_paper_baselines
from .utils import ensure_dir, seed_everything


def require_optuna():
    try:
        import optuna
        return optuna
    except ImportError as e:
        raise RuntimeError("Hyperparameter search requires Optuna. Install with: pip install -e '.[paper]'") from e


def suggest_safegrip(trial, cfg):
    """Validation-only search space for the SafeGrip-CI v1.4 point path."""
    space=cfg.get("tuning",{}).get("space",{})
    p=cfg.get("proposal",{})
    return {
        "sequence_length":trial.suggest_categorical("sequence_length",space.get("sequence_length",[64,100,128])),
        "scaler":trial.suggest_categorical("scaler",space.get("scaler",["minmax","standard"])),
        "hidden":trial.suggest_categorical("hidden",space.get("hidden",[96,128,160])),
        "gru_hidden":trial.suggest_categorical("gru_hidden",space.get("gru_hidden",[128,192,256])),
        # v1.4 replaces the legacy convolutional stem with Identity(), so this
        # must stay fixed rather than acting as a hidden RNG selector.
        "conv_channels":int(p.get("conv_channels",64)),
        "gru_layers":trial.suggest_categorical("gru_layers",space.get("gru_layers",[1,2])),
        "dropout":trial.suggest_float("dropout",*space.get("dropout",[0.0,0.3])),
        "lr":trial.suggest_float("lr",*space.get("lr",[1e-4,3e-3]),log=True),
        "weight_decay":trial.suggest_float("weight_decay",*space.get("weight_decay",[1e-6,1e-3]),log=True),
        "batch_size":trial.suggest_categorical("batch_size",space.get("batch_size",[64,128,256])),
        # Active v1.4 point training uses MSE, not Huber.
        "huber_beta":float(p.get("huber_beta",0.05)),
        "counterfactual_delta":trial.suggest_categorical("counterfactual_delta",space.get("counterfactual_delta",[0.04,0.08,0.12])),
        # Historical v1.2 parameters are kept in the returned dictionary for
        # checkpoint/config compatibility but are not Optuna dimensions in
        # v1.3 because entropy identifiability and the energy posterior do not
        # use the old Jacobian regularizer or Gauss--Newton ridge.
        "identifiability_lambda":float(p.get("identifiability_lambda",1e-3)),
        "inverse_dynamics_ridge":float(p.get("inverse_dynamics_ridge",1e-3)),
        "inverse_dynamics_max_step":trial.suggest_categorical("inverse_dynamics_max_step",space.get("inverse_dynamics_max_step",[0.06,0.10,0.14])),
        "physics_correction_scale":trial.suggest_categorical("physics_correction_scale",space.get("physics_correction_scale",[0.5,0.75,1.0])),
        "energy_grid_points":trial.suggest_categorical("energy_grid_points",space.get("energy_grid_points",[5,7,9])),
        "energy_grid_radius":trial.suggest_categorical("energy_grid_radius",space.get("energy_grid_radius",[0.06,0.10,0.14])),
        "energy_temperature":trial.suggest_categorical("energy_temperature",space.get("energy_temperature",[0.20,0.35,0.50])),
        "energy_noise_floor":float(p.get("energy_noise_floor",1e-4)),
        "energy_margin_threshold":trial.suggest_categorical("energy_margin_threshold",space.get("energy_margin_threshold",[0.10,0.25,0.50])),
        "energy_margin_temperature":float(p.get("energy_margin_temperature",0.15)),
        "base_pretrain_epochs":trial.suggest_categorical("base_pretrain_epochs",space.get("base_pretrain_epochs",[10,15,20])),
        "selector_warmup_epochs":trial.suggest_categorical("selector_warmup_epochs",space.get("selector_warmup_epochs",[2,4,6])),
        "base_joint_lr_scale":trial.suggest_categorical("base_joint_lr_scale",space.get("base_joint_lr_scale",[0.05,0.10,0.20])),
        "base_loss_weight":trial.suggest_categorical("base_loss_weight",space.get("base_loss_weight",[0.10,0.25,0.50])),
        "dynamic_loss_weight":0.0,
        "safety_loss_weight":float(p.get("safety_loss_weight",0.0)),
        "unsafe_margin":float(p.get("unsafe_margin",0.05)),
        "benefit_gate_loss_weight":trial.suggest_categorical("benefit_gate_loss_weight",space.get("benefit_gate_loss_weight",[0.10,0.20,0.35])),
        "correction_fraction_loss_weight":trial.suggest_categorical("correction_fraction_loss_weight",space.get("correction_fraction_loss_weight",[0.25,0.35,0.50])),
        "do_no_harm_weight":trial.suggest_categorical("do_no_harm_weight",space.get("do_no_harm_weight",[0.0,0.05,0.10])),
        "utility_gate_loss_weight":float(p.get("utility_gate_loss_weight",0.35)),
        # Regime/change/smoothness heads are disabled in the active v1.3 path;
        # keep their config values fixed instead of wasting tuning budget.
        "change_loss_weight":0.0,
        "change_threshold":float(p.get("change_threshold",0.02)),
        "smooth_loss_weight":0.0,
        "heteroscedastic_loss_weight":float(p.get("heteroscedastic_loss_weight",0.0)),
        "dynamics_loss_weight":trial.suggest_categorical("dynamics_loss_weight",space.get("dynamics_loss_weight",[0.05,0.10,0.20])),
        "counterfactual_loss_weight":trial.suggest_categorical("counterfactual_loss_weight",space.get("counterfactual_loss_weight",[0.1,0.2,0.35])),
        "counterfactual_margin":float(p.get("counterfactual_margin",0.02)),
        "contrastive_temperature":trial.suggest_categorical("contrastive_temperature",space.get("contrastive_temperature",[0.15,0.25,0.40])),
        "contrastive_negatives":trial.suggest_categorical("contrastive_negatives",space.get("contrastive_negatives",[4,6,8])),
        "dynamics_pretrain_epochs":trial.suggest_categorical("dynamics_pretrain_epochs",space.get("dynamics_pretrain_epochs",[5,8,12])),
        # UQ-only parameters remain fixed during point-RMSE selection.
        "information_beta":float(p.get("information_beta",0.5)),
        "disagreement_beta":float(p.get("disagreement_beta",0.25)),
        "uq_scale_floor":float(p.get("uq_scale_floor",0.005)),
        "uq_lr":float(p.get("uq_lr",1e-3)),
        "uq_epochs":int(p.get("uq_epochs",100)),
        "aleatoric_floor":float(p.get("aleatoric_floor",0.005)),
        "utility_gate_beta":float(p.get("utility_gate_beta",0.02)),
        "unsafe_temperature":float(p.get("unsafe_temperature",0.02)),
        "benefit_margin":float(p.get("benefit_margin",0.0)),
        "benefit_temperature":float(p.get("benefit_temperature",0.01)),
        "do_no_harm_margin":float(p.get("do_no_harm_margin",0.002)),
        "correction_fraction_beta":float(p.get("correction_fraction_beta",0.05)),
        "freeze_dynamics_after_pretrain":bool(p.get("freeze_dynamics_after_pretrain",True)),
    }

def _gaussian_nll(y,p,s):
    s=np.maximum(np.asarray(s),1e-5)
    return float(np.mean(.5*((np.asarray(y)-np.asarray(p))/s)**2+np.log(s)))


def tune_safegrip(csv_path, out_dir, cfg, trials=None, epochs=None, evaluate_test=True):
    """Tune SafeGrip on validation RMSE only; the test partition stays locked."""
    optuna=require_optuna(); out=ensure_dir(out_dir); seed_everything(cfg["seed"])
    trials=int(trials or cfg.get("tuning",{}).get("trials",30))
    epochs=int(epochs or cfg.get("tuning",{}).get("epochs",cfg["training"]["epochs_paper"]))
    start=tuning_eval_start(cfg,include_baselines=True); cache={}

    def bundle(seq,scaler):
        key=(int(seq),str(scaler))
        if key not in cache: cache[key]=make_bundle(csv_path,cfg,sequence_length=int(seq),scaler_kind=str(scaler),eval_start=start,feature_mode="raw")
        return cache[key]

    def objective(trial):
        hp=suggest_safegrip(trial,cfg); b=bundle(hp["sequence_length"],hp["scaler"])
        seed_everything(int(cfg["seed"])+trial.number)
        model,_=fit_proposal("safegrip_no_uq",b,cfg,epochs,hp)
        d=predict_proposal_details(model,"safegrip_no_uq",b.Xv,b.lov,b.raw_lov,cfg["mu_upper"],excitation=b.ev,ids=b.idv)
        p=d["prediction"]
        rmse=float(np.sqrt(np.mean((b.yv-p)**2)))
        lower_violation=float(np.mean(b.lov>b.yv))
        trial.set_user_attr("val_rmse",rmse); trial.set_user_attr("lower_violation_rate",lower_violation)
        # Main benchmark ranks point estimators by RMSE; use the same validation
        # selection metric for the proposal and every literature comparator.
        return rmse

    db=out/"optuna.sqlite3"
    study=optuna.create_study(direction="minimize",study_name="safegrip_ci_v140_rmse",storage=f"sqlite:///{db}",load_if_exists=True,
                              sampler=optuna.samplers.TPESampler(seed=cfg["seed"]))
    remaining=max(0,trials-len(study.trials))
    if remaining: study.optimize(objective,n_trials=remaining)
    best={**study.best_trial.params}
    (out/"best_hparams.yaml").write_text(yaml.safe_dump(best,sort_keys=False),encoding="utf-8")
    pd.DataFrame(study.trials_dataframe()).to_csv(out/"trials.csv",index=False)
    try:
        importance=optuna.importance.get_param_importances(study)
        (out/"param_importance.json").write_text(json.dumps(importance,indent=2),encoding="utf-8")
    except Exception:
        importance={}
    ref_bundle=bundle(int(best["sequence_length"]),best.get("scaler",cfg.get("proposal",{}).get("scaler","minmax")))
    endpoint_ids=ref_bundle.idv.astype(str).tolist()
    endpoint_hash=__import__("hashlib").sha256("\n".join(endpoint_ids).encode()).hexdigest()
    (out/"tuning_endpoint_manifest.json").write_text(json.dumps({"eval_start":start,"n":len(endpoint_ids),"sha256":endpoint_hash,"endpoint_ids":endpoint_ids},indent=2),encoding="utf-8")
    summary={"objective":"validation RMSE","best_value":study.best_value,"best_trial":study.best_trial.number,"best_params":best,
             "test_used_during_search":False,"n_trials_total":len(study.trials),"param_importance":importance,
             "common_eval_start":start,"validation_endpoint_sha256":endpoint_hash,
             "fixed_not_tuned":{"mu_upper":cfg["mu_upper"],"alpha":cfg["alpha"],
                                "information_beta":cfg.get("proposal",{}).get("information_beta",1.0),
                                "identifiability_lambda":"historical v1.2 compatibility only; entropy identifiability has no lambda",
                                "inverse_dynamics_ridge":"historical v1.2 compatibility only; v1.4 has no Gauss--Newton solve",
                                "dynamic_change_regime_losses":"disabled in active v1.4 point path",
                                "uq_reason":"point-model search uses validation RMSE; UQ-only parameters are not searched against a point metric"}}
    if evaluate_test:
        b=bundle(int(best["sequence_length"]),best.get("scaler",cfg.get("proposal",{}).get("scaler","minmax"))); seed_everything(cfg["seed"]); model,_=fit_proposal("safegrip",b,cfg,epochs,best)
        d=predict_proposal_details(model,"safegrip",b.Xt,b.lot,b.raw_lot,cfg["mu_upper"],excitation=b.et,ids=b.idt)
        p,s,bound=d["prediction"],d["sigma"],d["bound"]
        summary["final_test_metrics"]=regression_metrics(
            b.yt,p,bound,cfg["mu_upper"],s,raw_mean=d["raw_mean"],
            interval_low=d.get("pi95_low_physics"),interval_high=d.get("pi95_high_physics"),
            interval_low_raw=d.get("pi95_low_raw"),interval_high_raw=d.get("pi95_high_raw"),
        )
        pd.DataFrame({"y_true":b.yt,"prediction":p,"sigma":s,"physics_lower":bound,
                      "pi95_low":d.get("pi95_low_physics"),"pi95_high":d.get("pi95_high_physics"),
                      "excitation":b.et}).to_csv(out/"best_test_predictions.csv",index=False)
    (out/"tuning_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    return summary


SENSITIVITY_DEFAULT_PARAMETERS = (
    "physics_correction_scale",
    "energy_grid_points",
    "energy_grid_radius",
    "energy_temperature",
    "energy_margin_threshold",
    "base_loss_weight",
    "selector_warmup_epochs",
    "base_joint_lr_scale",
    "benefit_gate_loss_weight",
    "correction_fraction_loss_weight",
    "do_no_harm_weight",
    "counterfactual_loss_weight",
    "contrastive_temperature",
    "dynamics_pretrain_epochs",
)


def run_hyperparameter_sensitivity(csv_path, out_dir, cfg, best_hparams, parameters=None, epochs=None):
    """One-factor-at-a-time sensitivity analysis around selected SafeGrip settings.

    This is deliberately *not* another hyperparameter selection loop.  The
    selected configuration is held fixed and each scientifically important
    proposal parameter is swept over the predeclared tuning grid on the locked
    validation endpoints.  The resulting table is suitable for a robustness
    figure/table without touching test labels.
    """
    out=ensure_dir(out_dir)
    base=dict(best_hparams or {})
    hp0={**cfg.get("proposal",{}), **base}
    seq=int(base.get("sequence_length",cfg.get("sequence_length",16)))
    start=tuning_eval_start(cfg,include_baselines=True)
    b=make_bundle(csv_path,cfg,sequence_length=seq,scaler_kind=str(base.get("scaler",cfg.get("proposal",{}).get("scaler","minmax"))),eval_start=start,feature_mode="raw")
    search_space=cfg.get("tuning",{}).get("space",{})
    parameters=list(parameters or SENSITIVITY_DEFAULT_PARAMETERS)
    bad=[k for k in parameters if k not in search_space]
    if bad:
        raise ValueError("No declared tuning grid for sensitivity parameters: "+", ".join(bad))
    epochs=int(epochs or cfg.get("tuning",{}).get("epochs",cfg["training"]["epochs_paper"]))

    rows=[]
    for pi,name in enumerate(parameters):
        values=list(search_space[name])
        # Always include the selected value if it is not already on the declared grid.
        selected=base.get(name,hp0.get(name))
        if selected is not None and selected not in values:
            values.append(selected)
        for value in values:
            hp=dict(base); hp[name]=value; hp["sequence_length"]=seq
            # Same seed for all values: isolate the hyperparameter rather than RNG.
            seed_everything(int(cfg["seed"]))
            model,_=fit_proposal("safegrip_no_uq",b,cfg,epochs,hp)
            d=predict_proposal_details(model,"safegrip_no_uq",b.Xv,b.lov,b.raw_lov,cfg["mu_upper"],excitation=b.ev,ids=b.idv)
            y=np.asarray(b.yv,float); pred=np.asarray(d["prediction"],float)
            row={
                "parameter":name,
                "value":value,
                "is_selected":bool(selected is not None and value==selected),
                "validation_rmse":float(np.sqrt(np.mean((y-pred)**2))),
                "validation_mae":float(np.mean(np.abs(y-pred))),
                "mean_authority":float(np.mean(d.get("authority",np.nan))),
                "mean_identifiability":float(np.mean(d.get("identifiability",np.nan))),
                "mean_acceptance":float(np.mean(d.get("acceptance",np.nan))),
                "mean_information_scale_cv":float(np.mean(d.get("information_scale_cv",np.nan))),
                "mean_local_linearity":float(np.mean(d.get("local_linearity",np.nan))),
                "mean_counterfactual_agreement":float(np.mean(d.get("counterfactual_agreement",np.nan))),
            }
            rows.append(row)
    table=pd.DataFrame(rows)
    table.to_csv(out/"hyperparameter_sensitivity.csv",index=False)
    endpoint_ids=b.idv.astype(str).tolist()
    endpoint_hash=__import__("hashlib").sha256("\n".join(endpoint_ids).encode()).hexdigest()
    summary={
        "protocol":"one-factor-at-a-time validation sensitivity around selected SafeGrip hyperparameters",
        "test_labels_used":False,
        "same_validation_endpoints":True,
        "common_eval_start":start,
        "validation_endpoint_sha256":endpoint_hash,
        "sequence_length_fixed":seq,
        "parameters":parameters,
        "selected_hparams":base,
        "epochs":epochs,
    }
    (out/"hyperparameter_sensitivity.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    return table


def _baseline_space(cfg,name):
    common=cfg.get("baseline_tuning",{})
    custom=common.get("space",{}).get(name,{})
    return common,custom


def suggest_literature(trial,cfg,name):
    """Dataset-adaptation search while preserving recoverable core architectures."""
    common,space=_baseline_space(cfg,name)
    seq_default=common.get("sequence_length",[16,32,64,100,128])
    hp={
        "sequence_length":trial.suggest_categorical("sequence_length",space.get("sequence_length",seq_default)),
        "lr":trial.suggest_float("lr",*space.get("lr",common.get("lr",[1e-4,3e-3])),log=True),
        "weight_decay":trial.suggest_float("weight_decay",*space.get("weight_decay",common.get("weight_decay",[1e-6,1e-3])),log=True),
        "batch_size":trial.suggest_categorical("batch_size",space.get("batch_size",common.get("batch_size",[64,128,256]))),
    }
    # Architecture unknown publicly: tune only the adapted implementation rather
    # than presenting arbitrary fixed values as if they were source parameters.
    if name=="schaefke2023_transformer":
        hp.update({
            "hidden":trial.suggest_categorical("hidden",space.get("hidden",[64,128])),
            "layers":trial.suggest_int("layers",*space.get("layers",[1,3])),
            "heads":trial.suggest_categorical("heads",space.get("heads",[2,4,8])),
            "ff_mult":trial.suggest_categorical("ff_mult",space.get("ff_mult",[2,4])),
            "dropout":trial.suggest_float("dropout",*space.get("dropout",[0.0,0.3])),
        })
    elif name=="chen2025_svdkl":
        hp.update({
            "hidden":trial.suggest_categorical("hidden",space.get("hidden",[32,64,128])),
            "feature_dim":trial.suggest_categorical("feature_dim",space.get("feature_dim",[8,16,32])),
            "inducing":trial.suggest_categorical("inducing",space.get("inducing",[64,128,256])),
            "dropout":trial.suggest_float("dropout",*space.get("dropout",[0.0,0.3])),
        })
    elif name=="todorovic2022_cnn":
        # Keep the published 100-sample context and architecture fixed.
        hp["sequence_length"]=100
        hp["dropout"]=trial.suggest_float("dropout",*space.get("dropout",[0.0,0.2]))
    elif name in ("lampe2023_lstm","lampe2023_gru"):
        # Architecture, MinMax scaling and source initialization remain fixed.
        hp["dropout"]=0.0
    return hp


def tune_literature_baselines(csv_path,out_dir,cfg,names=None,trials=None,epochs=None,protocol="controlled"):
    """Give every literature comparator the same validation-trial budget.

    Core architecture/preprocessing constraints from the papers remain fixed.
    Only dataset-dependent settings (and unknown architecture details for the
    explicitly adapted Transformer/SV-DKL comparators) are selected on validation.
    """
    optuna=require_optuna(); out=ensure_dir(out_dir)
    names=list(names or PAPER_BASELINES); validate_paper_baselines(names)
    protocol=str(protocol).replace("-","_")
    if protocol not in {"controlled","source_faithful"}: raise ValueError("protocol must be controlled or source_faithful")
    controlled=cfg.get("comparison",{}).get("controlled",{})
    trials=int(trials or (controlled.get("tuning_trials",30) if protocol=="controlled" else cfg.get("baseline_tuning",{}).get("trials",30)))
    tuning_seeds=[int(x) for x in controlled.get("tuning_seeds",[1101,2202])] if protocol=="controlled" else [int(cfg.get("seed",0))]
    start=tuning_eval_start(cfg,include_baselines=True); all_best={}; summaries={}

    for model_index,name in enumerate(names):
        model_out=ensure_dir(out/name); cache={}
        source=literature_hparams(name,cfg,None,"paper")
        def bundle(seq):
            key=(int(seq),str(source["scaler"]))
            if key not in cache:
                cache[key]=make_bundle(csv_path,cfg,sequence_length=key[0],scaler_kind=key[1],eval_start=start)
            return cache[key]

        def objective(trial):
            hp=suggest_literature(trial,cfg,name); full=literature_hparams(name,cfg,hp,"paper")
            b=bundle(full["sequence_length"])
            if protocol=="controlled":
                full.update({
                    "epochs":int(epochs or controlled.get("epochs",cfg["training"]["epochs_paper"])),
                    "patience":int(controlled.get("patience",cfg["training"].get("patience",10))),
                    "batch_size":int(controlled.get("batch_size",128)),
                })
                if bool(controlled.get("common_optimizer_hparams",True)):
                    full["lr"]=float(controlled.get("lr",1e-3)); full["weight_decay"]=float(controlled.get("weight_decay",1e-4))
            run_epochs=int(epochs or full["epochs"])
            rmses=[]
            for tuning_seed in tuning_seeds:
                seed_everything(int(tuning_seed))
                model=fit_literature(name,b,cfg,run_epochs,"paper",full)
                p,_=predict_literature(model,name,b.Xv)
                rmses.append(float(np.sqrt(np.mean((b.yv-p)**2))))
            rmse=float(np.mean(rmses))
            trial.set_user_attr("seed_rmse",rmses); return rmse

        db=model_out/"optuna.sqlite3"
        study=optuna.create_study(direction="minimize",study_name=name,storage=f"sqlite:///{db}",load_if_exists=True,
                                  sampler=optuna.samplers.TPESampler(seed=int(cfg["seed"])+model_index))
        remaining=max(0,trials-len(study.trials))
        if remaining: study.optimize(objective,n_trials=remaining)
        best={**study.best_trial.params}
        if name=="todorovic2022_cnn": best["sequence_length"]=100
        if name in ("lampe2023_lstm","lampe2023_gru"): best["dropout"]=0.0
        all_best[name]=best
        (model_out/"best_hparams.yaml").write_text(yaml.safe_dump(best,sort_keys=False),encoding="utf-8")
        pd.DataFrame(study.trials_dataframe()).to_csv(model_out/"trials.csv",index=False)
        rb=bundle(best["sequence_length"]); ids=rb.idv.astype(str).tolist()
        eh=__import__("hashlib").sha256("\n".join(ids).encode()).hexdigest()
        (model_out/"tuning_endpoint_manifest.json").write_text(json.dumps({"eval_start":start,"n":len(ids),"sha256":eh,"endpoint_ids":ids},indent=2),encoding="utf-8")
        summaries[name]={"objective":"mean validation RMSE across fixed tuning seeds","best_value":study.best_value,"best_trial":study.best_trial.number,
                         "best_params":best,"n_trials_total":len(study.trials),"test_used_during_search":False,
                         "protocol":protocol,"tuning_seeds":tuning_seeds,
                         "common_eval_start":start,"validation_endpoint_sha256":eh}
        (model_out/"tuning_summary.json").write_text(json.dumps(summaries[name],indent=2),encoding="utf-8")

    (out/"best_hparams.yaml").write_text(yaml.safe_dump(all_best,sort_keys=False),encoding="utf-8")
    (out/"tuning_summary.json").write_text(json.dumps({"same_trial_budget":trials,"common_eval_start":start,
                                                         "primary_selection_metric":"validation RMSE","protocol":protocol,
                                                         "tuning_seeds":tuning_seeds,"models":summaries},indent=2),encoding="utf-8")
    return {"same_trial_budget":trials,"common_eval_start":start,"protocol":protocol,"tuning_seeds":tuning_seeds,"models":summaries,"best_params":all_best}


def tune_ablation_variants(csv_path, out_dir, cfg, variants=None, trials=15, epochs=None):
    """Retune each SafeGrip ablation on the identical validation endpoints.

    This is a supplementary robustness analysis.  The main controlled ablation
    should keep the full model hyperparameters fixed; this routine answers the
    separate objection that an ablated architecture might simply require a
    different optimum.
    """
    from .benchmark import PROPOSAL_VARIANTS, _proposal_flags
    optuna=require_optuna(); out=ensure_dir(out_dir)
    variants=list(variants or PROPOSAL_VARIANTS)
    bad=[v for v in variants if v not in PROPOSAL_VARIANTS]
    if bad: raise ValueError("Unknown proposal variants: "+", ".join(bad))
    trials=int(trials); epochs=int(epochs or cfg.get("tuning",{}).get("epochs",cfg["training"]["epochs_paper"]))
    start=tuning_eval_start(cfg,include_baselines=True)
    summaries={}; best_all={}
    for vi,variant in enumerate(variants):
        vo=ensure_dir(out/variant); cache={}
        mode=_proposal_flags(variant)["feature_mode"]
        def bundle(seq,scaler):
            key=(int(seq),str(scaler))
            if key not in cache:
                cache[key]=make_bundle(csv_path,cfg,sequence_length=key[0],scaler_kind=key[1],eval_start=start,feature_mode=mode)
            return cache[key]
        def objective(trial):
            hp=suggest_safegrip(trial,cfg); b=bundle(hp["sequence_length"],hp.get("scaler",cfg.get("proposal",{}).get("scaler","minmax")))
            seed_everything(int(cfg["seed"])+100000*vi+trial.number)
            model,_=fit_proposal(variant,b,cfg,epochs,hp)
            d=predict_proposal_details(model,variant,b.Xv,b.lov,b.raw_lov,cfg["mu_upper"],excitation=b.ev,ids=b.idv)
            return float(np.sqrt(np.mean((b.yv-d["prediction"])**2)))
        study=optuna.create_study(direction="minimize",study_name=f"{variant}_retuned_rmse",
            storage=f"sqlite:///{vo/'optuna.sqlite3'}",load_if_exists=True,
            sampler=optuna.samplers.TPESampler(seed=int(cfg["seed"])+vi))
        remaining=max(0,trials-len(study.trials))
        if remaining: study.optimize(objective,n_trials=remaining)
        best=dict(study.best_trial.params); best_all[variant]=best
        rb=bundle(best["sequence_length"],best.get("scaler",cfg.get("proposal",{}).get("scaler","minmax"))); ids=rb.idv.astype(str).tolist()
        eh=__import__("hashlib").sha256("\n".join(ids).encode()).hexdigest()
        (vo/"best_hparams.yaml").write_text(yaml.safe_dump(best,sort_keys=False),encoding="utf-8")
        pd.DataFrame(study.trials_dataframe()).to_csv(vo/"trials.csv",index=False)
        (vo/"tuning_endpoint_manifest.json").write_text(json.dumps({"eval_start":start,"n":len(ids),"sha256":eh,"endpoint_ids":ids},indent=2),encoding="utf-8")
        summaries[variant]={"best_validation_rmse":float(study.best_value),"best_trial":int(study.best_trial.number),
                            "best_params":best,"n_trials_total":len(study.trials),"validation_endpoint_sha256":eh}
    (out/"best_hparams.yaml").write_text(yaml.safe_dump(best_all,sort_keys=False),encoding="utf-8")
    (out/"retuned_ablation_summary.json").write_text(json.dumps({"protocol":"retuned supplementary ablation","same_validation_endpoints":True,"common_eval_start":start,"models":summaries},indent=2),encoding="utf-8")
    return summaries


def tune_frc(csv_path, out_dir, cfg, trials=None, epochs=None):
    """Tune only ordinary FRC network-training hyperparameters.

    Mathematical quantities (candidate grid, certificate deltas, horizons and
    theorem definition) are intentionally fixed and are not optimized against
    validation RMSE. Every candidate is evaluated with the same tuning seeds.
    """
    from .frc_benchmark import (
        make_frc_bundle, fit_frc_response_model, calibrate_residual_radii,
        evaluate_frc_split, _common_eval_start_frc,
    )
    optuna=require_optuna(); out=ensure_dir(out_dir)
    tc=cfg.get("frc_tuning",{}); space=tc.get("space",{})
    trials=int(trials or tc.get("trials",30)); run_epochs=int(epochs or tc.get("epochs",40))
    seeds=[int(x) for x in tc.get("seeds",[1101,2202])]
    start=tuning_eval_start(cfg,include_baselines=True)
    b=make_frc_bundle(csv_path,cfg,eval_start=start)
    from .friction_resolution import make_mu_grid
    grid=make_mu_grid(float(cfg.get("frc",{}).get("mu_min",0.0)),
                      float(cfg.get("frc",{}).get("mu_max",cfg.get("mu_upper",1.3))),
                      float(cfg.get("frc",{}).get("mu_grid_step",0.02)))

    def objective(trial):
        hp={
            "hidden":trial.suggest_categorical("hidden",space.get("hidden",[96,128,160])),
            "gru_layers":trial.suggest_categorical("gru_layers",space.get("gru_layers",[1,2])),
            "dropout":trial.suggest_float("dropout",*space.get("dropout",[0.0,0.3])),
            "lr":trial.suggest_float("lr",*space.get("lr",[1e-4,3e-3]),log=True),
            "weight_decay":trial.suggest_float("weight_decay",*space.get("weight_decay",[1e-6,1e-3]),log=True),
            "batch_size":trial.suggest_categorical("batch_size",space.get("batch_size",[64,128,256])),
            "epochs":run_epochs,
        }
        vals=[]
        for seed in seeds:
            seed_everything(seed)
            model,_,_=fit_frc_response_model(b,cfg,hp)
            radii=calibrate_residual_radii(model,b,cfg,grid)
            pred,*_=evaluate_frc_split(model,b.Xv,b.Rv,b.yv,b,cfg,grid,radii)
            vals.append(float(np.sqrt(np.mean((b.yv-pred)**2))))
        score=float(np.mean(vals)); trial.set_user_attr("seed_rmse",vals); return score

    db=Path(out)/"optuna.sqlite3"
    study=optuna.create_study(direction="minimize",study_name="safegrip_frc",storage=f"sqlite:///{db}",load_if_exists=True,
                              sampler=optuna.samplers.TPESampler(seed=int(cfg.get("seed",0))))
    remaining=max(0,trials-len(study.trials))
    if remaining: study.optimize(objective,n_trials=remaining)
    best=dict(study.best_trial.params); best["epochs"]=run_epochs
    (Path(out)/"best_hparams.yaml").write_text(yaml.safe_dump(best,sort_keys=False),encoding="utf-8")
    pd.DataFrame(study.trials_dataframe()).to_csv(Path(out)/"trials.csv",index=False)
    ids=b.idv.astype(str).tolist(); eh=__import__("hashlib").sha256("\n".join(ids).encode()).hexdigest()
    (Path(out)/"tuning_endpoint_manifest.json").write_text(json.dumps({"eval_start":start,"n":len(ids),"sha256":eh,"endpoint_ids":ids},indent=2),encoding="utf-8")
    summary={
        "method":"safegrip_frc","objective":"mean validation RMSE across fixed tuning seeds",
        "best_value":float(study.best_value),"best_trial":int(study.best_trial.number),"best_params":best,
        "n_trials_total":len(study.trials),"tuning_seeds":seeds,"test_used_during_search":False,
        "mathematical_parameters_tuned":False,"common_eval_start":start,
    }
    (Path(out)/"tuning_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    return summary


def tune_pfr(csv_path, out_dir, cfg, trials=None, epochs=None):
    """Validation-only tuning for the small SafeGrip-PFR neural residual model.

    Only ordinary approximation/optimization hyperparameters are searched.
    The theorem parameters ``alpha``, the conformal quantile rule, the mechanics
    lower bound, and ``mu_upper`` are fixed by the protocol and never optimized.
    """
    from .pfr_benchmark import _fit_scalar_gru, _predict_scalar
    from .pfr import compose_raw_prediction, pfr_project
    from .physics import conformal_lower_correction, apply_lower_correction

    optuna = require_optuna()
    out = ensure_dir(out_dir)
    tc = cfg.get("pfr_tuning", {})
    space = tc.get("space", {})
    trials = int(trials or tc.get("trials", 20))
    run_epochs = int(epochs or tc.get("epochs", 40))
    seeds = [int(x) for x in tc.get("seeds", [1101, 2202])]
    start = tuning_eval_start(cfg, include_baselines=True)

    def objective(trial):
        hp = {
            "sequence_length": trial.suggest_categorical(
                "sequence_length", space.get("sequence_length", [16, 32, 64])
            ),
            "scaler": trial.suggest_categorical("scaler", space.get("scaler", ["standard", "minmax"])),
            "hidden": trial.suggest_categorical("hidden", space.get("hidden", [32, 64, 96])),
            "gru_layers": trial.suggest_categorical("gru_layers", space.get("gru_layers", [1, 2])),
            "dropout": trial.suggest_float("dropout", *space.get("dropout", [0.0, 0.3])),
            "lr": trial.suggest_float("lr", *space.get("lr", [1e-4, 3e-3]), log=True),
            "weight_decay": trial.suggest_float(
                "weight_decay", *space.get("weight_decay", [1e-6, 1e-3]), log=True
            ),
            "batch_size": trial.suggest_categorical("batch_size", space.get("batch_size", [64, 128, 256])),
            "epochs": run_epochs,
            "patience": int(cfg.get("training", {}).get("patience", 10)),
        }
        b = make_bundle(
            csv_path,
            cfg,
            sequence_length=int(hp["sequence_length"]),
            scaler_kind=str(hp["scaler"]),
            eval_start=start,
            feature_mode="raw",
        )
        q = conformal_lower_correction(b.raw_loc, b.yc, float(cfg.get("alpha", 0.05)))
        lower_v = apply_lower_correction(b.raw_lov, q)
        vals = []
        for seed in seeds:
            seed_everything(seed)
            fit = _fit_scalar_gru(b, cfg, hp, target_mode="residual")
            residual = _predict_scalar(fit, b.Xv)
            raw = compose_raw_prediction(b.raw_lov, residual)
            pred = pfr_project(raw, lower_v, float(cfg.get("mu_upper", 1.3)))
            vals.append(float(np.sqrt(np.mean((b.yv - pred) ** 2))))
        score = float(np.mean(vals))
        trial.set_user_attr("seed_rmse", vals)
        return score

    db = Path(out) / "optuna.sqlite3"
    study = optuna.create_study(
        direction="minimize",
        study_name="safegrip_pfr",
        storage=f"sqlite:///{db}",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=int(cfg.get("seed", 0))),
    )
    remaining = max(0, trials - len(study.trials))
    if remaining:
        study.optimize(objective, n_trials=remaining)
    best = dict(study.best_trial.params)
    best["epochs"] = run_epochs
    best["patience"] = int(cfg.get("training", {}).get("patience", 10))
    (Path(out) / "best_hparams.yaml").write_text(yaml.safe_dump(best, sort_keys=False), encoding="utf-8")
    pd.DataFrame(study.trials_dataframe()).to_csv(Path(out) / "trials.csv", index=False)
    best_bundle = make_bundle(
        csv_path, cfg, sequence_length=int(best["sequence_length"]),
        scaler_kind=str(best["scaler"]), eval_start=start, feature_mode="raw"
    )
    ids = best_bundle.idv.astype(str).tolist()
    endpoint_hash = __import__("hashlib").sha256("\n".join(ids).encode()).hexdigest()
    (Path(out) / "tuning_endpoint_manifest.json").write_text(
        json.dumps({"eval_start": int(start), "n": len(ids), "sha256": endpoint_hash, "endpoint_ids": ids}, indent=2),
        encoding="utf-8",
    )
    summary = {
        "method": "safegrip_pfr",
        "objective": "mean projected validation RMSE across fixed tuning seeds",
        "best_value": float(study.best_value),
        "best_trial": int(study.best_trial.number),
        "best_params": best,
        "n_trials_total": len(study.trials),
        "tuning_seeds": seeds,
        "test_used_during_search": False,
        "mathematical_parameters_tuned": False,
        "common_eval_start": int(start),
    }
    (Path(out) / "tuning_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
