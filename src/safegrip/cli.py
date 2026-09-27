from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import yaml

from .ablations import ABLATIONS, get_ablation
from .config import load_config
from .datasets import DATASET_REGISTRY, PAPER_DATASETS
from .literature import LITERATURE_BASELINES, baselines_for_dataset, validate_paper_baselines
from .download import download_dataset
from .data import prepare_dataset
from .pfr_benchmark import run_pfr_benchmark
from .plots import make_plots
from .experiments import run_statistical_comparison


def _primary_csv(name: str) -> str:
    if name != "lira_cd":
        raise ValueError("legacy PFR benchmark is retained only for D1/lira_cd")
    return "lira_aligned.csv"


def _ensure_primary(name, cfg):
    proc = Path("data/processed") / name
    csv = proc / _primary_csv(name)
    if not csv.exists():
        prepare_dataset(name, Path("data/raw") / name, proc, cfg)
    return csv


def _load_mapping(path, label):
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{label} file must contain a YAML mapping")
    return data


def _universal_check(dataset: str, ablation_name: str) -> dict:
    import torch
    from .universal.schema import SensorMeta, SensorChannel, SensorRecord
    from .universal.tokenizer import UniversalSensorTokenizer, TokenizerConfig
    from .universal.dataset import collate_sensor_records
    from .universal.queries import queries_for_dataset
    from .model.safegrip_universal import UniversalSafeGrip

    ab = get_ablation(ablation_name)
    tok = UniversalSensorTokenizer(TokenizerConfig(use_spectrum=ab.spectrum))
    t = np.linspace(0.0, 1.0, 21)
    record = SensorRecord(
        [SensorChannel(np.sin(2 * np.pi * t), t, SensorMeta("acceleration", "m/s^2", "lateral", "vehicle_body", 20.0))],
        sequence_id="cli-check",
        source_domain=dataset,
    )
    queries = queries_for_dataset(dataset)
    batch = collate_sensor_records([record], tok, queries)
    model = UniversalSafeGrip(
        tok.feature_dim, token_dim=24, latent_dim=48, latent_tokens=4,
        cross_attention_heads=4, latent_heads=4, latent_layers=1,
        dropout=0.0, ablation=ab,
    )
    model.eval()
    with torch.no_grad():
        out = model(batch.sensors, queries, domains=batch.domains)
    return {
        "dataset": dataset,
        "ablation": ablation_name,
        "queries": [q.name or q.quantity for q in queries],
        "point_shape": list(out.point.shape),
        "scale_shape": list(out.scale.shape),
        "aggregation": ab.aggregation,
        "learned_scale": ab.learned_scale,
        "passed": bool(torch.isfinite(out.point).all() and torch.isfinite(out.scale).all()),
    }


def main():
    ap = argparse.ArgumentParser(prog="safegrip", description="Universal SafeGrip D1--D4 paper benchmark")
    ap.add_argument("--config", default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)

    ds = sub.add_parser("datasets", help="list the closed D1--D4 paper datasets")
    ds.add_argument("action", nargs="?", choices=["list"], default="list")

    bl = sub.add_parser("baselines", help="show the only paper comparators allowed for a dataset")
    bl.add_argument("--dataset", choices=PAPER_DATASETS, required=True)

    d = sub.add_parser("download")
    d.add_argument("--datasets", nargs="+", default=["lira_cd"])
    d.add_argument("--force", action="store_true")
    d.add_argument("--full", action="store_true")

    p = sub.add_parser("prepare")
    p.add_argument("--dataset", choices=PAPER_DATASETS, required=True)

    b = sub.add_parser("benchmark", help="run locally implemented paper benchmark paths")
    b.add_argument("--dataset", choices=PAPER_DATASETS, required=True)
    b.add_argument("--preset", choices=["quick", "trust", "paper"], default="quick")
    b.add_argument("--models", default=None)
    b.add_argument("--protocol", choices=["controlled", "source-faithful"], default="controlled")
    b.add_argument("--proposal-hparams", default=None)
    b.add_argument("--baseline-hparams", default=None)

    uc = sub.add_parser("universal-check", help="architecture/query/ablation sanity check without dataset labels")
    uc.add_argument("--dataset", choices=PAPER_DATASETS, required=True)
    uc.add_argument("--ablation", choices=tuple(ABLATIONS), default="full")

    st = sub.add_parser("statistics")
    st.add_argument("--results", required=True)
    st.add_argument("--proposal", default="safegrip_pfr")
    st.add_argument("--bootstrap", type=int, default=2000)

    pl = sub.add_parser("plots")
    pl.add_argument("--results", required=True)
    sub.add_parser("universal-smoke")

    args = ap.parse_args()
    cfg = load_config(args.config)

    if args.cmd == "datasets":
        print(json.dumps(DATASET_REGISTRY, indent=2)); return
    if args.cmd == "baselines":
        payload = {name: LITERATURE_BASELINES[name] for name in baselines_for_dataset(args.dataset)}
        print(json.dumps(payload, indent=2)); return
    if args.cmd == "download":
        names = list(PAPER_DATASETS) if "all" in args.datasets else args.datasets
        for n in names:
            download_dataset(n, force=args.force, full=args.full)
        return
    if args.cmd == "prepare":
        prepare_dataset(args.dataset, Path("data/raw") / args.dataset, Path("data/processed") / args.dataset, cfg)
        return
    if args.cmd == "universal-check":
        print(json.dumps(_universal_check(args.dataset, args.ablation), indent=2)); return
    if args.cmd == "universal-smoke":
        from .universal.smoke import run_universal_smoke
        print(json.dumps(run_universal_smoke(), indent=2)); return
    if args.cmd == "benchmark":
        names = args.models.split(",") if args.models else list(baselines_for_dataset(args.dataset))
        validate_paper_baselines(names, dataset=args.dataset)
        if args.dataset != "lira_cd":
            missing = [n for n in names if not LITERATURE_BASELINES[n]["runnable"]]
            raise RuntimeError(
                "Dataset and paper-baseline provenance are registered, but exact local baseline wrappers are not yet implemented for "
                f"{args.dataset}: {', '.join(missing)}. Do not substitute generic baselines."
            )
        validate_paper_baselines(names, dataset="lira_cd", require_runnable=True)
        csv = _ensure_primary(args.dataset, cfg)
        proposal_hp = _load_mapping(args.proposal_hparams, "proposal hyperparameter")
        baseline_hp = _load_mapping(args.baseline_hparams, "baseline hyperparameter")
        suffix = "" if args.protocol == "controlled" else "_source_faithful"
        out = Path("results") / f"{args.dataset}_{args.preset}{suffix}"
        run_pfr_benchmark(csv, out, cfg, args.preset, names, proposal_hp, baseline_hp, args.protocol)
        return
    if args.cmd == "statistics":
        print(run_statistical_comparison(Path(args.results), Path(args.results) / "statistics", proposal=args.proposal, bootstrap=args.bootstrap).to_string(index=False)); return
    if args.cmd == "plots":
        r = Path(args.results)
        if (r / "metrics.csv").exists():
            make_plots(r)


if __name__ == "__main__":
    main()
