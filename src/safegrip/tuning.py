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
    seq_choices=space.get("sequence_length",[8,16,24,32,64])
    hp={
        "sequence_length":trial.suggest_categorical("sequence_length",seq_choices),
        "hidden":trial.suggest_categorical("hidden",space.get("hidden",[32,64,96])),
        "gru_hidden":trial.suggest_categorical("gru_hidden",space.get("gru_hidden",[16,32,64])),
        "dropout":trial.suggest_float("dropout",*space.get("dropout",[0.0,0.3])),
        "lr":trial.suggest_float("lr",*space.get("lr",[1e-4,3e-3]),log=True),
        "weight_decay":trial.suggest_float("weight_decay",*space.get("weight_decay",[1e-6,1e-3]),log=True),
        "batch_size":trial.suggest_categorical("batch_size",space.get("batch_size",[128,256,512])),
        "huber_beta":trial.suggest_categorical("huber_beta",space.get("huber_beta",[0.03,0.05,0.10])),
        "evidence_window":trial.suggest_categorical("evidence_window",space.get("evidence_window",[4,8,12])),
        "delta_scale":trial.suggest_categorical("delta_scale",space.get("delta_scale",[1.0,2.0,3.0])),
        "delta_loss_weight":trial.suggest_categorical("delta_loss_weight",space.get("delta_loss_weight",[0.05,0.15,0.35])),
        "rank_loss_weight":trial.suggest_categorical("rank_loss_weight",space.get("rank_loss_weight",[0.0,0.03,0.05])),
        "excitation_beta":trial.suggest_categorical("excitation_beta",space.get("excitation_beta",[0.5,1.0,2.0])),
    }
    return hp


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
        if seq not in cache: cache[seq]=make_bundle(csv_path,cfg,sequence_length=seq,scaler_kind="standard",eval_start=start,proposal_features=True)
        return cache[seq]

    def objective(trial):
        hp=suggest_safegrip(trial,cfg); b=bundle(hp["sequence_length"])
        seed_everything(int(cfg["seed"])+trial.number)
        model,_=fit_proposal("safegrip_no_uq",b,cfg,epochs,hp)
        d=predict_proposal_details(model,"safegrip_no_uq",b.Xv,b.lov,b.raw_lov,cfg["mu_upper"],excitation=b.ev)
        p=d["prediction"]
        rmse=float(np.sqrt(np.mean((b.yv-p)**2)))
        lower_violation=float(np.mean(b.lov>b.yv))
        trial.set_user_attr("val_rmse",rmse); trial.set_user_attr("lower_violation_rate",lower_violation)
        # Main benchmark ranks point estimators by RMSE; use the same validation
        # selection metric for the proposal and every literature comparator.
        return rmse

    db=out/"optuna.sqlite3"
    study=optuna.create_study(direction="minimize",study_name="safegrip_v070_rmse",storage=f"sqlite:///{db}",load_if_exists=True,
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
             "fixed_not_tuned":{"mu_upper":cfg["mu_upper"],"alpha":cfg["alpha"]}}
    if evaluate_test:
        b=bundle(int(best["sequence_length"])); seed_everything(cfg["seed"]); model,_=fit_proposal("safegrip",b,cfg,epochs,best)
        d=predict_proposal_details(model,"safegrip",b.Xt,b.lot,b.raw_lot,cfg["mu_upper"],excitation=b.et)
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
        raw=_proposal_flags(variant)["raw_features_only"]
        def bundle(seq):
            key=int(seq)
            if key not in cache:
                cache[key]=make_bundle(csv_path,cfg,sequence_length=key,scaler_kind="standard",eval_start=start,proposal_features=not raw)
            return cache[key]
        def objective(trial):
            hp=suggest_safegrip(trial,cfg); b=bundle(hp["sequence_length"])
            seed_everything(int(cfg["seed"])+100000*vi+trial.number)
            model,_=fit_proposal(variant,b,cfg,epochs,hp)
            d=predict_proposal_details(model,variant,b.Xv,b.lov,b.raw_lov,cfg["mu_upper"],excitation=b.ev)
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
