from __future__ import annotations
import argparse, json
import yaml
from pathlib import Path

from .config import load_config
from .datasets import DATASET_REGISTRY
from .download import download_dataset
from .data import prepare_dataset, make_synthetic
from .benchmark import run_benchmark, run_ablation, PROPOSAL_VARIANTS
from .tuning import tune_safegrip
from .plots import make_plots, make_ablation_plots


def _primary_csv(name: str) -> str:
    return {"lira":"lira_aligned.csv","synthetic":"synthetic.csv"}[name]


def _ensure_primary(name,cfg):
    proc=Path("data/processed")/name; csv=proc/_primary_csv(name)
    if not csv.exists():
        if name=="synthetic": make_synthetic(proc,seed=cfg["seed"])
        else: prepare_dataset(name,Path("data/raw")/name,proc,cfg)
    return csv


def _load_hparams(path):
    if not path:
        return None
    with open(path,"r",encoding="utf-8") as f:
        data=yaml.safe_load(f) or {}
    if not isinstance(data,dict):
        raise ValueError("proposal hyperparameter file must contain a YAML mapping")
    return data


def main():
    ap=argparse.ArgumentParser(prog="safegrip",description="Open-data SafeGrip research benchmark")
    ap.add_argument("--config",default=None)
    sub=ap.add_subparsers(dest="cmd",required=True)

    sub.add_parser("datasets",help="list supported open datasets and scientific roles")
    d=sub.add_parser("download"); d.add_argument("--datasets",nargs="+",default=["lira"]); d.add_argument("--force",action="store_true"); d.add_argument("--full",action="store_true",help="download full optional archives where the safe default is a subset")
    p=sub.add_parser("prepare"); p.add_argument("--dataset",choices=list(DATASET_REGISTRY)+["synthetic"],required=True)
    b=sub.add_parser("benchmark"); b.add_argument("--dataset",choices=["lira","synthetic"],required=True); b.add_argument("--preset",choices=["quick","paper"],default="quick"); b.add_argument("--models",default=None,help="comma-separated cited baselines; uncited names are rejected"); b.add_argument("--proposal-hparams",default=None,help="YAML from safegrip tune; applied only to the proposal while all methods share the selected input window")
    a=sub.add_parser("ablation"); a.add_argument("--dataset",choices=["lira","synthetic"],required=True); a.add_argument("--preset",choices=["quick","paper"],default="paper"); a.add_argument("--variants",default=None,help="comma-separated proposal ablations"); a.add_argument("--proposal-hparams",default=None,help="YAML from safegrip tune; same hyperparameters are reused across ablations")
    t=sub.add_parser("tune"); t.add_argument("--dataset",choices=["lira","synthetic"],required=True); t.add_argument("--trials",type=int,default=None); t.add_argument("--epochs",type=int,default=None); t.add_argument("--no-test",action="store_true")
    pl=sub.add_parser("plots"); pl.add_argument("--results",required=True)

    args=ap.parse_args(); cfg=load_config(args.config)
    if args.cmd=="datasets":
        print(json.dumps(DATASET_REGISTRY,indent=2)); return
    if args.cmd=="download":
        names=list(DATASET_REGISTRY) if "all" in args.datasets else args.datasets
        for n in names: download_dataset(n,force=args.force,full=args.full)
        return
    if args.cmd=="prepare":
        if args.dataset=="synthetic": prepare_dataset("synthetic","data/raw/synthetic","data/processed/synthetic",cfg)
        else: prepare_dataset(args.dataset,Path("data/raw")/args.dataset,Path("data/processed")/args.dataset,cfg)
        return
    if args.cmd=="benchmark":
        csv=_ensure_primary(args.dataset,cfg); out=Path("results")/f"{args.dataset}_{args.preset}"
        models=args.models.split(",") if args.models else None
        run_benchmark(csv,out,cfg,args.preset,models,_load_hparams(args.proposal_hparams)); make_plots(out); return
    if args.cmd=="ablation":
        csv=_ensure_primary(args.dataset,cfg); out=Path("results")/f"{args.dataset}_ablation_{args.preset}"
        variants=args.variants.split(",") if args.variants else None
        if variants:
            bad=[v for v in variants if v not in PROPOSAL_VARIANTS]
            if bad: raise ValueError("Unknown ablations: "+", ".join(bad))
        run_ablation(csv,out,cfg,args.preset,variants,_load_hparams(args.proposal_hparams)); make_ablation_plots(out); return
    if args.cmd=="tune":
        csv=_ensure_primary(args.dataset,cfg); out=Path("results")/f"{args.dataset}_tuning"
        print(json.dumps(tune_safegrip(csv,out,cfg,args.trials,args.epochs,not args.no_test),indent=2)); return
    if args.cmd=="plots":
        r=Path(args.results)
        if (r/"metrics.csv").exists(): make_plots(r)
        if (r/"ablation_metrics.csv").exists(): make_ablation_plots(r)

if __name__=="__main__": main()
