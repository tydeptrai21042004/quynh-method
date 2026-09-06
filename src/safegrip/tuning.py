from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

from .benchmark import make_bundle, fit_proposal, predict_proposal, regression_metrics
from .utils import ensure_dir, seed_everything


def require_optuna():
    try:
        import optuna
        return optuna
    except ImportError as e:
        raise RuntimeError("Hyperparameter search requires Optuna. Install with: pip install -e '.[paper]'") from e


def suggest_safegrip(trial, cfg):
    space=cfg.get("tuning",{}).get("space",{})
    seq_choices=space.get("sequence_length",[16,32,64,128])
    hp={
        "sequence_length":trial.suggest_categorical("sequence_length",seq_choices),
        "hidden":trial.suggest_categorical("hidden",space.get("hidden",[32,64,96,128])),
        "tcn_blocks":trial.suggest_int("tcn_blocks",*space.get("tcn_blocks",[2,5])),
        "kernel_size":trial.suggest_categorical("kernel_size",space.get("kernel_size",[2,3,5])),
        "dropout":trial.suggest_float("dropout",*space.get("dropout",[0.0,0.3])),
        "lr":trial.suggest_float("lr",*space.get("lr",[1e-4,3e-3]),log=True),
        "weight_decay":trial.suggest_float("weight_decay",*space.get("weight_decay",[1e-6,1e-3]),log=True),
        "batch_size":trial.suggest_categorical("batch_size",space.get("batch_size",[128,256,512])),
        "lambda_mse":trial.suggest_float("lambda_mse",*space.get("lambda_mse",[0.05,0.5]),log=True),
        "lambda_physics":trial.suggest_float("lambda_physics",*space.get("lambda_physics",[1e-3,0.3]),log=True),
    }
    return hp


def _gaussian_nll(y,p,s):
    s=np.maximum(np.asarray(s),1e-5)
    return float(np.mean(.5*((np.asarray(y)-np.asarray(p))/s)**2+np.log(s)))


def tune_safegrip(csv_path, out_dir, cfg, trials=None, epochs=None, evaluate_test=True):
    """Tune ONLY the proposed method using train/calibration/validation.

    The held-out test partition is never consulted by the Optuna objective. After
    the best validation trial is selected, evaluate_test optionally reports one
    final test score for that selected configuration.
    """
    optuna=require_optuna(); out=ensure_dir(out_dir); seed_everything(cfg["seed"])
    trials=int(trials or cfg.get("tuning",{}).get("trials",30))
    epochs=int(epochs or cfg.get("tuning",{}).get("epochs",cfg["training"]["epochs_paper"]))
    cache={}

    def bundle(seq):
        if seq not in cache: cache[seq]=make_bundle(csv_path,cfg,sequence_length=seq)
        return cache[seq]

    def objective(trial):
        hp=suggest_safegrip(trial,cfg); b=bundle(hp["sequence_length"])
        model,_=fit_proposal("safegrip",b,cfg,epochs,hp)
        p,s,_=predict_proposal(model,"safegrip",b.Xv,b.lov,b.raw_lov,cfg["mu_upper"])
        rmse=float(np.sqrt(np.mean((b.yv-p)**2))); nll=_gaussian_nll(b.yv,p,s)
        lower_violation=float(np.mean(b.lov>b.yv))
        trial.set_user_attr("val_rmse",rmse); trial.set_user_attr("val_nll",nll); trial.set_user_attr("lower_violation_rate",lower_violation)
        # NLL is a proper scoring rule and evaluates the proposal's point + scale
        # output jointly. Safety calibration itself is fixed by alpha, not tuned.
        return nll

    db=out/"optuna.sqlite3"
    study=optuna.create_study(direction="minimize",study_name="safegrip",storage=f"sqlite:///{db}",load_if_exists=True,
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
    summary={"objective":"validation Gaussian NLL","best_value":study.best_value,"best_trial":study.best_trial.number,"best_params":best,
             "test_used_during_search":False,"n_trials_total":len(study.trials),"param_importance":importance,
             "fixed_not_tuned":{"mu_upper":cfg["mu_upper"],"alpha":cfg["alpha"]}}
    if evaluate_test:
        b=bundle(int(best["sequence_length"])); model,_=fit_proposal("safegrip",b,cfg,epochs,best)
        p,s,bound=predict_proposal(model,"safegrip",b.Xt,b.lot,b.raw_lot,cfg["mu_upper"])
        summary["final_test_metrics"]=regression_metrics(b.yt,p,bound,cfg["mu_upper"],s)
        pd.DataFrame({"y_true":b.yt,"prediction":p,"sigma":s,"physics_lower":bound}).to_csv(out/"best_test_predictions.csv",index=False)
    (out/"tuning_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    return summary
