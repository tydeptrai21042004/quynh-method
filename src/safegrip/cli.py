from __future__ import annotations
import argparse, json
import yaml
from pathlib import Path

from .config import load_config
from .datasets import DATASET_REGISTRY, PRIMARY_FRICTION_DATASETS
from .download import download_dataset
from .data import prepare_dataset
from .pfr_benchmark import run_pfr_benchmark
from .tuning import tune_pfr, tune_literature_baselines
from .plots import make_plots
from .experiments import run_force_validation, run_statistical_comparison, run_external_friction_reference_validation


def _primary_csv(name: str) -> str:
    return {
        "lira": "lira_aligned.csv",
        "mssp2023_friction": "mssp2023_friction_aligned.csv",
    }[name]


def _ensure_primary(name,cfg):
    proc=Path("data/processed")/name; csv=proc/_primary_csv(name)
    if not csv.exists():
        prepare_dataset(name,Path("data/raw")/name,proc,cfg)
    return csv


def _load_mapping(path,label):
    if not path: return None
    with open(path,"r",encoding="utf-8") as f:
        data=yaml.safe_load(f) or {}
    if not isinstance(data,dict):
        raise ValueError(f"{label} file must contain a YAML mapping")
    return data


def main():
    ap=argparse.ArgumentParser(
        prog="safegrip",
        description="SafeGrip-PFR-ECR open-data road-friction benchmark",
    )
    ap.add_argument("--config",default=None)
    sub=ap.add_subparsers(dest="cmd",required=True)

    sub.add_parser("datasets",help="list supported open datasets and scientific roles")
    d=sub.add_parser("download"); d.add_argument("--datasets",nargs="+",default=["lira"]); d.add_argument("--force",action="store_true"); d.add_argument("--full",action="store_true")
    p=sub.add_parser("prepare"); p.add_argument("--dataset",choices=list(DATASET_REGISTRY),required=True)

    b=sub.add_parser("benchmark",help="benchmark the only active proposal: SafeGrip-PFR-ECR")
    b.add_argument("--dataset",choices=list(PRIMARY_FRICTION_DATASETS),required=True)
    b.add_argument("--preset",choices=["quick","trust","paper"],default="quick")
    b.add_argument("--models",default=None,help="comma-separated subset of the four registered paper-supported friction baselines")
    b.add_argument("--proposal",choices=["pfr"],default="pfr",help="backward-compatible flag; SafeGrip-PFR-ECR is the only active proposal")
    b.add_argument("--protocol",choices=["controlled","source-faithful"],default="controlled")
    b.add_argument("--proposal-hparams",default=None)
    b.add_argument("--baseline-hparams",default=None)

    t=sub.add_parser("tune",help="tune SafeGrip-PFR-ECR")
    t.add_argument("--dataset",choices=list(PRIMARY_FRICTION_DATASETS),required=True)
    t.add_argument("--method",choices=["pfr"],default="pfr",help="backward-compatible flag; PFR-ECR is the only active proposal")
    t.add_argument("--trials",type=int,default=None)
    t.add_argument("--epochs",type=int,default=None)
    t.add_argument("--no-test",action="store_true",help="retained for command compatibility; PFR tuning never selects on test labels")

    tb=sub.add_parser("tune-baselines")
    tb.add_argument("--dataset",choices=list(PRIMARY_FRICTION_DATASETS),required=True)
    tb.add_argument("--models",default=None)
    tb.add_argument("--protocol",choices=["controlled","source-faithful"],default="controlled")
    tb.add_argument("--trials",type=int,default=None)
    tb.add_argument("--epochs",type=int,default=None)

    st=sub.add_parser("statistics")
    st.add_argument("--results",required=True)
    st.add_argument("--proposal",choices=["safegrip_pfr"],default="safegrip_pfr")
    st.add_argument("--bootstrap",type=int,default=2000)

    fv=sub.add_parser("force-validate")
    fv.add_argument("--dataset",choices=["kit","kuleuven"],required=True)
    fv.add_argument("--eps-t",type=float,default=250.0)
    fv.add_argument("--eps-z",type=float,default=250.0)

    rv=sub.add_parser("friction-reference-validate", help="validate a real external friction-reference dataset without forcing incompatible sensor inputs")
    rv.add_argument("--dataset", choices=["mendeley_friction"], required=True)

    pl=sub.add_parser("plots"); pl.add_argument("--results",required=True)
    sub.add_parser("universal-smoke", help="run the dataset-independent Universal SafeGrip architecture smoke test")

    args=ap.parse_args(); cfg=load_config(args.config)
    if args.cmd=="universal-smoke":
        from .universal.smoke import run_universal_smoke
        print(json.dumps(run_universal_smoke(), indent=2)); return
    if args.cmd=="datasets":
        print(json.dumps(DATASET_REGISTRY,indent=2)); return
    if args.cmd=="download":
        names=list(DATASET_REGISTRY) if "all" in args.datasets else args.datasets
        for n in names: download_dataset(n,force=args.force,full=args.full)
        return
    if args.cmd=="prepare":
        prepare_dataset(args.dataset,Path("data/raw")/args.dataset,Path("data/processed")/args.dataset,cfg)
        return
    if args.cmd=="benchmark":
        csv=_ensure_primary(args.dataset,cfg)
        models=args.models.split(",") if args.models else None
        proposal_hp=_load_mapping(args.proposal_hparams,"proposal hyperparameter")
        baseline_hp=_load_mapping(args.baseline_hparams,"baseline hyperparameter")
        suffix="" if args.protocol=="controlled" else "_source_faithful"
        out=Path("results")/f"{args.dataset}_{args.preset}{suffix}"
        run_pfr_benchmark(csv,out,cfg,args.preset,models,proposal_hp,baseline_hp,args.protocol)
        return
    if args.cmd=="tune":
        csv=_ensure_primary(args.dataset,cfg); out=Path("results")/f"{args.dataset}_pfr_tuning"
        print(json.dumps(tune_pfr(csv,out,cfg,args.trials,args.epochs),indent=2)); return
    if args.cmd=="tune-baselines":
        csv=_ensure_primary(args.dataset,cfg); out=Path("results")/f"{args.dataset}_baseline_tuning"
        models=args.models.split(",") if args.models else None
        print(json.dumps(tune_literature_baselines(csv,out,cfg,models,args.trials,args.epochs,args.protocol),indent=2)); return
    if args.cmd=="statistics":
        print(run_statistical_comparison(Path(args.results),Path(args.results)/"statistics",proposal=args.proposal,bootstrap=args.bootstrap).to_string(index=False)); return
    if args.cmd=="force-validate":
        proc=Path("data/processed")/args.dataset
        prepared=proc/("kit_force.csv" if args.dataset=="kit" else "kuleuven_wft.csv")
        if not prepared.exists(): prepare_dataset(args.dataset,Path("data/raw")/args.dataset,proc,cfg)
        print(run_force_validation(prepared,Path("results")/f"{args.dataset}_force_validation",cfg,eps_t=args.eps_t,eps_z=args.eps_z).to_string(index=False)); return
    if args.cmd=="friction-reference-validate":
        proc=Path("data/processed")/args.dataset
        prepared=proc/"mendeley_friction.csv"
        if not prepared.exists(): prepare_dataset(args.dataset,Path("data/raw")/args.dataset,proc,cfg)
        print(run_external_friction_reference_validation(prepared,Path("results")/f"{args.dataset}_reference_validation").to_string(index=False)); return
    if args.cmd=="plots":
        r=Path(args.results)
        if (r/"metrics.csv").exists(): make_plots(r)


if __name__=="__main__": main()
