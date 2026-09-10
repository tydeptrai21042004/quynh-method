from __future__ import annotations
import argparse, json
import yaml
from pathlib import Path

from .config import load_config
from .datasets import DATASET_REGISTRY
from .download import download_dataset
from .data import prepare_dataset, make_synthetic
from .benchmark import run_benchmark, run_ablation, PROPOSAL_VARIANTS
from .tuning import tune_safegrip, tune_literature_baselines, tune_ablation_variants, run_hyperparameter_sensitivity
from .plots import make_plots, make_ablation_plots
from .experiments import (
    run_excitation_analysis, run_robustness_analysis, run_scarcity_analysis,
    run_cross_route_analysis, run_force_validation, run_statistical_comparison,
)


def _primary_csv(name: str) -> str:
    return {"lira":"lira_aligned.csv","synthetic":"synthetic.csv"}[name]


def _ensure_primary(name,cfg):
    proc=Path("data/processed")/name; csv=proc/_primary_csv(name)
    if not csv.exists():
        if name=="synthetic": make_synthetic(proc,seed=cfg["seed"])
        else: prepare_dataset(name,Path("data/raw")/name,proc,cfg)
    return csv


def _load_mapping(path,label):
    if not path: return None
    with open(path,"r",encoding="utf-8") as f:
        data=yaml.safe_load(f) or {}
    if not isinstance(data,dict):
        raise ValueError(f"{label} file must contain a YAML mapping")
    return data


def main():
    ap=argparse.ArgumentParser(prog="safegrip",description="Open-data SafeGrip research benchmark")
    ap.add_argument("--config",default=None)
    sub=ap.add_subparsers(dest="cmd",required=True)

    sub.add_parser("datasets",help="list supported open datasets and scientific roles")
    d=sub.add_parser("download"); d.add_argument("--datasets",nargs="+",default=["lira"]); d.add_argument("--force",action="store_true"); d.add_argument("--full",action="store_true",help="download full optional archives where the safe default is a subset")
    p=sub.add_parser("prepare"); p.add_argument("--dataset",choices=list(DATASET_REGISTRY)+["synthetic"],required=True)
    b=sub.add_parser("benchmark"); b.add_argument("--dataset",choices=["lira","synthetic"],required=True); b.add_argument("--preset",choices=["quick","trust","paper"],default="quick"); b.add_argument("--models",default=None,help="comma-separated cited baselines; uncited names are rejected"); b.add_argument("--proposal-hparams",default=None,help="YAML from safegrip tune; applied only to SafeGrip"); b.add_argument("--baseline-hparams",default=None,help="aggregate YAML from safegrip tune-baselines; each comparator uses its own selected settings")
    a=sub.add_parser("ablation"); a.add_argument("--dataset",choices=["lira","synthetic"],required=True); a.add_argument("--preset",choices=["quick","trust","paper"],default="paper"); a.add_argument("--variants",default=None,help="comma-separated proposal ablations"); a.add_argument("--proposal-hparams",default=None,help="YAML from safegrip tune; same hyperparameters are reused across ablations")
    t=sub.add_parser("tune"); t.add_argument("--dataset",choices=["lira","synthetic"],required=True); t.add_argument("--trials",type=int,default=None); t.add_argument("--epochs",type=int,default=None); t.add_argument("--no-test",action="store_true")
    tb=sub.add_parser("tune-baselines"); tb.add_argument("--dataset",choices=["lira","synthetic"],required=True); tb.add_argument("--models",default=None,help="comma-separated literature baselines; default is the full paper set"); tb.add_argument("--trials",type=int,default=None); tb.add_argument("--epochs",type=int,default=None,help="optional development cap; omit for each method's paper-mode epoch budget")
    ta=sub.add_parser("tune-ablation"); ta.add_argument("--dataset",choices=["lira","synthetic"],required=True); ta.add_argument("--variants",default=None); ta.add_argument("--trials",type=int,default=15); ta.add_argument("--epochs",type=int,default=None)
    hs=sub.add_parser("sensitivity",help="one-factor proposal hyperparameter sensitivity on locked validation endpoints"); hs.add_argument("--dataset",choices=["lira","synthetic"],required=True); hs.add_argument("--proposal-hparams",required=True,help="selected best_hparams.yaml from safegrip tune"); hs.add_argument("--parameters",default=None,help="optional comma-separated proposal parameters"); hs.add_argument("--epochs",type=int,default=None)
    st=sub.add_parser("statistics"); st.add_argument("--results",required=True); st.add_argument("--proposal",default="safegrip"); st.add_argument("--bootstrap",type=int,default=2000)
    e=sub.add_parser("experiment",help="run reviewer-oriented SafeGrip analyses")
    e.add_argument("--dataset",choices=["lira","synthetic"],required=True)
    e.add_argument("--study",choices=["excitation","robustness","scarcity","cross-route"],required=True)
    e.add_argument("--results",default=None,help="benchmark directory; defaults to results/<dataset>_paper")
    e.add_argument("--preset",choices=["quick","trust","paper"],default="paper")
    e.add_argument("--proposal-hparams",default=None)
    e.add_argument("--bins",type=int,default=4)
    fv=sub.add_parser("force-validate",help="validate mechanics lower bound on prepared force datasets")
    fv.add_argument("--dataset",choices=["kit","kuleuven"],required=True)
    fv.add_argument("--eps-t",type=float,default=250.0)
    fv.add_argument("--eps-z",type=float,default=250.0)
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
        run_benchmark(csv,out,cfg,args.preset,models,
                      _load_mapping(args.proposal_hparams,"proposal hyperparameter"),
                      _load_mapping(args.baseline_hparams,"baseline hyperparameter")); make_plots(out); return
    if args.cmd=="ablation":
        csv=_ensure_primary(args.dataset,cfg); out=Path("results")/f"{args.dataset}_ablation_{args.preset}"
        variants=args.variants.split(",") if args.variants else None
        if variants:
            bad=[v for v in variants if v not in PROPOSAL_VARIANTS]
            if bad: raise ValueError("Unknown ablations: "+", ".join(bad))
        run_ablation(csv,out,cfg,args.preset,variants,_load_mapping(args.proposal_hparams,"proposal hyperparameter")); make_ablation_plots(out); return
    if args.cmd=="tune":
        csv=_ensure_primary(args.dataset,cfg); out=Path("results")/f"{args.dataset}_tuning"
        print(json.dumps(tune_safegrip(csv,out,cfg,args.trials,args.epochs,not args.no_test),indent=2)); return
    if args.cmd=="tune-baselines":
        csv=_ensure_primary(args.dataset,cfg); out=Path("results")/f"{args.dataset}_baseline_tuning"
        models=args.models.split(",") if args.models else None
        print(json.dumps(tune_literature_baselines(csv,out,cfg,models,args.trials,args.epochs),indent=2)); return
    if args.cmd=="tune-ablation":
        csv=_ensure_primary(args.dataset,cfg); out=Path("results")/f"{args.dataset}_ablation_retuned"
        variants=args.variants.split(",") if args.variants else None
        print(json.dumps(tune_ablation_variants(csv,out,cfg,variants,args.trials,args.epochs),indent=2)); return
    if args.cmd=="sensitivity":
        csv=_ensure_primary(args.dataset,cfg); out=Path("results")/f"{args.dataset}_hyperparameter_sensitivity"
        hp=_load_mapping(args.proposal_hparams,"proposal hyperparameter")
        params=args.parameters.split(",") if args.parameters else None
        table=run_hyperparameter_sensitivity(csv,out,cfg,hp,params,args.epochs)
        print(table.to_string(index=False)); return
    if args.cmd=="statistics":
        print(run_statistical_comparison(Path(args.results),Path(args.results)/"statistics",proposal=args.proposal,bootstrap=args.bootstrap).to_string(index=False)); return
    if args.cmd=="experiment":
        csv=_ensure_primary(args.dataset,cfg)
        results=Path(args.results) if args.results else Path("results")/f"{args.dataset}_paper"
        out=Path("results")/f"{args.dataset}_{args.study.replace('-', '_')}"
        hp=_load_mapping(args.proposal_hparams,"proposal hyperparameter")
        if args.study=="excitation":
            print(run_excitation_analysis(csv,results,out,cfg,args.bins).to_string(index=False)); return
        if args.study=="robustness":
            print(run_robustness_analysis(csv,results,out,cfg).to_string(index=False)); return
        if args.study=="scarcity":
            print(run_scarcity_analysis(csv,out,cfg,preset=args.preset,hp_overrides=hp).to_string(index=False)); return
        if args.study=="cross-route":
            print(run_cross_route_analysis(csv,out,cfg,preset=args.preset,hp_overrides=hp).to_string(index=False)); return
    if args.cmd=="force-validate":
        proc=Path("data/processed")/args.dataset
        prepared=proc/("kit_force.csv" if args.dataset=="kit" else "kuleuven_wft.csv")
        if not prepared.exists():
            prepare_dataset(args.dataset,Path("data/raw")/args.dataset,proc,cfg)
        print(run_force_validation(prepared,Path("results")/f"{args.dataset}_force_validation",cfg,eps_t=args.eps_t,eps_z=args.eps_z).to_string(index=False)); return
    if args.cmd=="plots":
        r=Path(args.results)
        if (r/"metrics.csv").exists(): make_plots(r)
        if (r/"ablation_metrics.csv").exists(): make_ablation_plots(r)

if __name__=="__main__": main()
