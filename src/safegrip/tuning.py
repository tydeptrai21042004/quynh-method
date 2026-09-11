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
    space=cfg.get("tuning",{}).get("space",{})
    return {
        "sequence_length":trial.suggest_categorical("sequence_length",space.get("sequence_length",[8,16,24,32,64])),
        "hidden":trial.suggest_categorical("hidden",space.get("hidden",[32,64,96])),
        "gru_hidden":trial.suggest_categorical("gru_hidden",space.get("gru_hidden",[16,32,64])),
        "dropout":trial.suggest_float("dropout",*space.get("dropout",[0.0,0.3])),
        "lr":trial.suggest_float("lr",*space.get("lr",[1e-4,3e-3]),log=True),
        "weight_decay":trial.suggest_float("weight_decay",*space.get("weight_decay",[1e-6,1e-3]),log=True),
        "batch_size":trial.suggest_categorical("batch_size",space.get("batch_size",[128,256,512])),
        "huber_beta":trial.suggest_categorical("huber_beta",space.get("huber_beta",[0.03,0.05,0.10])),
        "evidence_window":trial.suggest_categorical("evidence_window",space.get("evidence_window",[4,8,12,16])),
        "delta_scale":trial.suggest_categorical("delta_scale",space.get("delta_scale",[0.2,0.3,0.5,0.8])),
        "state_persistence":trial.suggest_categorical("state_persistence",space.get("state_persistence",[0.5,0.7,0.8,0.9,0.97])),
        "counterfactual_delta":trial.suggest_categorical("counterfactual_delta",space.get("counterfactual_delta",[0.03,0.05,0.08,0.12])),
        "identifiability_lambda":trial.suggest_categorical("identifiability_lambda",space.get("identifiability_lambda",[1e-4,1e-3,1e-2])),
        "acceptance_temperature":trial.suggest_categorical("acceptance_temperature",space.get("acceptance_temperature",[6.0,12.0,20.0])),
        "acceptance_margin":trial.suggest_categorical("acceptance_margin",space.get("acceptance_margin",[-0.02,0.0,0.02])),
        "acceptance_tolerance":trial.suggest_categorical("acceptance_tolerance",space.get("acceptance_tolerance",[0.05,0.10,0.20])),
        "acceptance_strength":trial.suggest_categorical("acceptance_strength",space.get("acceptance_strength",[0.2,0.35,0.5])),
        "inverse_dynamics_ridge":trial.suggest_categorical("inverse_dynamics_ridge",space.get("inverse_dynamics_ridge",[1e-4,1e-3,1e-2])),
        "inverse_dynamics_max_step":trial.suggest_categorical("inverse_dynamics_max_step",space.get("inverse_dynamics_max_step",[0.06,0.12,0.20])),
        "inverse_expert_scale":trial.suggest_categorical("inverse_expert_scale",space.get("inverse_expert_scale",[0.25,0.5,0.75,1.0])),
        "arbitration_loss_weight":trial.suggest_categorical("arbitration_loss_weight",space.get("arbitration_loss_weight",[0.10,0.20,0.30,0.50])),
        "prior_loss_weight":trial.suggest_categorical("prior_loss_weight",space.get("prior_loss_weight",[0.0,0.03,0.05,0.10])),
        "teacher_forcing_start":trial.suggest_categorical("teacher_forcing_start",space.get("teacher_forcing_start",[0.5,0.8,1.0])),
        "innovation_loss_weight":trial.suggest_categorical("innovation_loss_weight",space.get("innovation_loss_weight",[0.10,0.20,0.35])),
        "state_update_loss_weight":trial.suggest_categorical("state_update_loss_weight",space.get("state_update_loss_weight",[0.20,0.35,0.50])),
        "candidate_loss_weight":trial.suggest_categorical("candidate_loss_weight",space.get("candidate_loss_weight",[0.10,0.25,0.40])),
        "direction_loss_weight":trial.suggest_categorical("direction_loss_weight",space.get("direction_loss_weight",[0.0,0.03,0.05,0.10])),
        "dynamics_loss_weight":trial.suggest_categorical("dynamics_loss_weight",space.get("dynamics_loss_weight",[0.05,0.10,0.20])),
        "counterfactual_loss_weight":trial.suggest_categorical("counterfactual_loss_weight",space.get("counterfactual_loss_weight",[0.02,0.05,0.10])),
        "counterfactual_margin":trial.suggest_categorical("counterfactual_margin",space.get("counterfactual_margin",[0.0,0.01,0.02,0.05])),
        "dynamics_pretrain_epochs":trial.suggest_categorical("dynamics_pretrain_epochs",space.get("dynamics_pretrain_epochs",[3,5,8,12])),
        "do_no_harm_weight":trial.suggest_categorical("do_no_harm_weight",space.get("do_no_harm_weight",[0.05,0.10,0.20])),
        # UQ-only: carried explicitly but not suggested against point-RMSE.
        "information_beta":float(cfg.get("proposal",{}).get("information_beta",1.0)),
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

    def bundle(seq):
        if seq not in cache: cache[seq]=make_bundle(csv_path,cfg,sequence_length=seq,scaler_kind="standard",eval_start=start,feature_mode="raw")
        return cache[seq]

    def objective(trial):
        hp=suggest_safegrip(trial,cfg); b=bundle(hp["sequence_length"])
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
    study=optuna.create_study(direction="minimize",study_name="safegrip_ci_v110_rmse",storage=f"sqlite:///{db}",load_if_exists=True,
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
    ref_bundle=bundle(int(best["sequence_length"]))
    endpoint_ids=ref_bundle.idv.astype(str).tolist()
    endpoint_hash=__import__("hashlib").sha256("\n".join(endpoint_ids).encode()).hexdigest()
    (out/"tuning_endpoint_manifest.json").write_text(json.dumps({"eval_start":start,"n":len(endpoint_ids),"sha256":endpoint_hash,"endpoint_ids":endpoint_ids},indent=2),encoding="utf-8")
    summary={"objective":"validation RMSE","best_value":study.best_value,"best_trial":study.best_trial.number,"best_params":best,
             "test_used_during_search":False,"n_trials_total":len(study.trials),"param_importance":importance,
             "common_eval_start":start,"validation_endpoint_sha256":endpoint_hash,
             "fixed_not_tuned":{"mu_upper":cfg["mu_upper"],"alpha":cfg["alpha"],
                                "information_beta":cfg.get("proposal",{}).get("information_beta",1.0),
                                "uq_reason":"point-model search uses validation RMSE; UQ-only parameters are not searched against a point metric"}}
    if evaluate_test:
        b=bundle(int(best["sequence_length"])); seed_everything(cfg["seed"]); model,_=fit_proposal("safegrip",b,cfg,epochs,best)
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
    "state_persistence",
    "counterfactual_delta",
    "identifiability_lambda",
    "inverse_dynamics_max_step",
    "inverse_expert_scale",
    "arbitration_loss_weight",
    "prior_loss_weight",
    "teacher_forcing_start",
    "innovation_loss_weight",
    "direction_loss_weight",
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
    b=make_bundle(csv_path,cfg,sequence_length=seq,scaler_kind="standard",eval_start=start,feature_mode="raw")
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


def tune_literature_baselines(csv_path,out_dir,cfg,names=None,trials=None,epochs=None):
    """Give every literature comparator the same validation-trial budget.

    Core architecture/preprocessing constraints from the papers remain fixed.
    Only dataset-dependent settings (and unknown architecture details for the
    explicitly adapted Transformer/SV-DKL comparators) are selected on validation.
    """
    optuna=require_optuna(); out=ensure_dir(out_dir)
    names=list(names or PAPER_BASELINES); validate_paper_baselines(names)
    trials=int(trials or cfg.get("baseline_tuning",{}).get("trials",cfg.get("tuning",{}).get("trials",30)))
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
            seed_everything(int(cfg["seed"])+10000*model_index+trial.number)
            run_epochs=int(epochs or cfg.get("baseline_tuning",{}).get("epochs",full["epochs"]))
            model=fit_literature(name,b,cfg,run_epochs,"paper",full)
            p,_=predict_literature(model,name,b.Xv)
            rmse=float(np.sqrt(np.mean((b.yv-p)**2)))
            trial.set_user_attr("val_rmse",rmse); return rmse

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
        summaries[name]={"objective":"validation RMSE","best_value":study.best_value,"best_trial":study.best_trial.number,
                         "best_params":best,"n_trials_total":len(study.trials),"test_used_during_search":False,
                         "common_eval_start":start,"validation_endpoint_sha256":eh}
        (model_out/"tuning_summary.json").write_text(json.dumps(summaries[name],indent=2),encoding="utf-8")

    (out/"best_hparams.yaml").write_text(yaml.safe_dump(all_best,sort_keys=False),encoding="utf-8")
    (out/"tuning_summary.json").write_text(json.dumps({"same_trial_budget":trials,"common_eval_start":start,
                                                         "primary_selection_metric":"validation RMSE","models":summaries},indent=2),encoding="utf-8")
    return {"same_trial_budget":trials,"common_eval_start":start,"models":summaries,"best_params":all_best}


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
        def bundle(seq):
            key=int(seq)
            if key not in cache:
                cache[key]=make_bundle(csv_path,cfg,sequence_length=key,scaler_kind="standard",eval_start=start,feature_mode=mode)
            return cache[key]
        def objective(trial):
            hp=suggest_safegrip(trial,cfg); b=bundle(hp["sequence_length"])
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
        rb=bundle(best["sequence_length"]); ids=rb.idv.astype(str).tolist()
        eh=__import__("hashlib").sha256("\n".join(ids).encode()).hexdigest()
        (vo/"best_hparams.yaml").write_text(yaml.safe_dump(best,sort_keys=False),encoding="utf-8")
        pd.DataFrame(study.trials_dataframe()).to_csv(vo/"trials.csv",index=False)
        (vo/"tuning_endpoint_manifest.json").write_text(json.dumps({"eval_start":start,"n":len(ids),"sha256":eh,"endpoint_ids":ids},indent=2),encoding="utf-8")
        summaries[variant]={"best_validation_rmse":float(study.best_value),"best_trial":int(study.best_trial.number),
                            "best_params":best,"n_trials_total":len(study.trials),"validation_endpoint_sha256":eh}
    (out/"best_hparams.yaml").write_text(yaml.safe_dump(best_all,sort_keys=False),encoding="utf-8")
    (out/"retuned_ablation_summary.json").write_text(json.dumps({"protocol":"retuned supplementary ablation","same_validation_endpoints":True,"common_eval_start":start,"models":summaries},indent=2),encoding="utf-8")
    return summaries
