from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

from .ablations import ABLATIONS, get_ablation
from .config import load_config
from .datasets import DATASET_REGISTRY, PAPER_DATASETS
from .literature import LITERATURE_BASELINES, baselines_for_dataset, validate_paper_baselines
from .download import download_dataset
from .data import prepare_dataset
from .plots import make_plots
from .experiments import run_statistical_comparison


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
        "proposal": "Universal SafeGrip-NPI",
        "proposal_revision": "normalized_physical_innovation_v3",
        "dataset": dataset,
        "ablation": ablation_name,
        "queries": [q.name or q.quantity for q in queries],
        "point_shape": list(out.point.shape),
        "scale_shape": list(out.scale.shape),
        "aggregation": ab.aggregation,
        "learned_scale": ab.learned_scale,
        "passed": bool(torch.isfinite(out.point).all() and torch.isfinite(out.scale).all()),
    }


def _baseline_check(dataset: str, *, train_step: bool = False) -> dict:
    import torch
    from .paper_baselines import make_paper_baseline
    from .baseline_training import train_baseline

    input_dims = {"uc3m_tire": 3, "deep_dynamics_iac": 5, "io_vnbd": 8}
    if dataset == "lira_cd":
        return {
            "dataset": dataset,
            "note": "D1 publication-backed comparators are exercised by the all-real-data paper runner.",
            "passed": True,
        }
    d = input_dims[dataset]
    torch.manual_seed(7)
    x = 0.1 * torch.randn(2, 12, d)
    if dataset == "deep_dynamics_iac":
        x[..., 0] += 8.0
    rows = []
    for name in baselines_for_dataset(dataset):
        build = make_paper_baseline(name, d, debug_scale=True)
        build.model.eval()
        with torch.no_grad():
            y = build.model(x)
        trained = None
        if train_step:
            target = torch.zeros_like(y)
            result = train_baseline(build.model, x, target, epochs=1, lr=1e-4)
            trained = result.losses[-1]
        rows.append({
            "name": name,
            "targets": list(build.target_names),
            "output_shape": list(y.shape),
            "finite": bool(torch.isfinite(y).all()),
            "train_step_loss": trained,
            "fidelity": LITERATURE_BASELINES[name]["fidelity"],
        })
    return {"dataset": dataset, "baselines": rows, "passed": all(r["finite"] for r in rows)}


def main():
    ap = argparse.ArgumentParser(
        prog="safegrip",
        description="Universal SafeGrip-NPI real-data research utilities",
    )
    ap.add_argument("--config", default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)

    ds = sub.add_parser("datasets", help="list the closed D1--D4 paper datasets")
    ds.add_argument("action", nargs="?", choices=["list"], default="list")

    bl = sub.add_parser("baselines", help="show publication-backed comparators for a dataset")
    bl.add_argument("--dataset", choices=PAPER_DATASETS, required=True)

    bc = sub.add_parser("baseline-check", help="instantiate and smoke-test local D2--D4 comparator reproductions")
    bc.add_argument("--dataset", choices=PAPER_DATASETS, required=True)
    bc.add_argument("--train-step", action="store_true")

    d = sub.add_parser("download")
    d.add_argument("--datasets", nargs="+", default=["lira_cd"])
    d.add_argument("--force", action="store_true")
    d.add_argument("--full", action="store_true")

    p = sub.add_parser("prepare")
    p.add_argument("--dataset", choices=PAPER_DATASETS, required=True)

    uc = sub.add_parser("universal-check", help="NPI architecture/query/ablation sanity check")
    uc.add_argument("--dataset", choices=PAPER_DATASETS, required=True)
    uc.add_argument("--ablation", choices=tuple(ABLATIONS), default="full")

    st = sub.add_parser("statistics")
    st.add_argument("--results", required=True)
    st.add_argument("--proposal", default="universal_safegrip")
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
    if args.cmd == "baseline-check":
        validate_paper_baselines(baselines_for_dataset(args.dataset), dataset=args.dataset, require_runnable=True)
        print(json.dumps(_baseline_check(args.dataset, train_step=args.train_step), indent=2)); return
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
    if args.cmd == "statistics":
        print(run_statistical_comparison(
            Path(args.results), Path(args.results) / "statistics",
            proposal=args.proposal, bootstrap=args.bootstrap,
        ).to_string(index=False)); return
    if args.cmd == "plots":
        r = Path(args.results)
        if (r / "metrics.csv").exists():
            make_plots(r)


if __name__ == "__main__":
    main()
