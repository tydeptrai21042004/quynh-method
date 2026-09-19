from __future__ import annotations

"""Primary theorem-driven SafeGrip-FRC benchmark.

This path is intentionally separate from the legacy SafeGrip-CI v1.4 benchmark.
It reuses the repository's prepared data, literature baselines, metrics, and
endpoint identity checks, while making the proposal minimal and theorem-aligned.
"""

from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from sklearn.preprocessing import StandardScaler

from .benchmark import (
    META,
    common_eval_start,
    literature_hparams,
    fit_literature,
    predict_literature,
    make_bundle,
    regression_metrics,
    _aggregate_seed_metrics,
    _fit_deterministic,
)
from .friction_resolution import (
    ResolutionResult,
    make_mu_grid,
    nearest_grid_values,
    separation_margin,
    resolution_certificate,
    select_horizon,
    empirical_residual_radius,
)
from .literature import PAPER_BASELINES, QUICK_BASELINES, LITERATURE_BASELINES, validate_paper_baselines
from .models import FrictionResponseNet, DirectGRUControl
from .physics import project_numpy
from .utils import ensure_dir, seed_everything, device


@dataclass
class FRCBundle:
    features: list[str]
    response_channels: list[str]
    scaler: StandardScaler
    response_mean: np.ndarray
    response_std: np.ndarray
    context_length: int
    max_horizon: int
    eval_start: int
    Xtr: np.ndarray; ytr: np.ndarray; mutr: np.ndarray; Rtr: np.ndarray; idtr: np.ndarray
    Xc: np.ndarray; yc: np.ndarray; muc: np.ndarray; Rc: np.ndarray; idc: np.ndarray
    Xv: np.ndarray; yv: np.ndarray; muv: np.ndarray; Rv: np.ndarray; idv: np.ndarray
    Xt: np.ndarray; yt: np.ndarray; mut: np.ndarray; Rt: np.ndarray; idt: np.ndarray


class _TransitionDataset(Dataset):
    """Causal transition examples without materializing duplicated contexts."""

    def __init__(self, X: np.ndarray, mu_seq: np.ndarray, responses: np.ndarray, context_length: int):
        self.X = X
        self.mu = mu_seq
        self.responses = responses
        self.c = int(context_length)
        self.h = int(responses.shape[1])
        if X.shape[0] != responses.shape[0] or mu_seq.shape != responses.shape[:2]:
            raise ValueError("Inconsistent FRC transition arrays")

    def __len__(self):
        return int(len(self.X) * self.h)

    def __getitem__(self, idx):
        n = int(idx // self.h)
        j = int(idx % self.h)
        # Transition target is at sample c+j. Context stops at c+j-1.
        context = self.X[n, j:j + self.c]
        return (
            torch.from_numpy(context.astype(np.float32, copy=False)),
            torch.tensor(self.mu[n, j], dtype=torch.float32),
            torch.from_numpy(self.responses[n, j].astype(np.float32, copy=False)),
        )


def _numeric_features(df: pd.DataFrame) -> list[str]:
    return [
        c for c in df.columns
        if c not in META
        and c not in ("mu_ref", "physics_lower_raw")
        and pd.api.types.is_numeric_dtype(df[c])
    ]


def _frc_windows_for_split(
    df: pd.DataFrame,
    features: list[str],
    response_channels: list[str],
    split: str,
    context_length: int,
    max_horizon: int,
    stride: int,
    eval_start: int,
):
    z = df[df["split"] == split].copy()
    w = int(context_length + max_horizon)
    xs: list[np.ndarray] = []
    ys: list[float] = []
    mus: list[np.ndarray] = []
    responses: list[np.ndarray] = []
    ids: list[str] = []
    if z.empty:
        return (
            np.empty((0, w, len(features)), np.float32),
            np.empty(0, np.float32),
            np.empty((0, max_horizon), np.float32),
            np.empty((0, max_horizon, len(response_channels)), np.float32),
            np.empty(0, dtype=str),
        )
    group_cols = [c for c in ("segment_id",) if c in z.columns]
    if not group_cols:
        group_cols = [c for c in ("trip_id",) if c in z.columns]
    groups = z.groupby(group_cols, sort=False, dropna=False) if group_cols else [("all", z)]
    first = max(w - 1, int(eval_start))
    ridx = [features.index(c) for c in response_channels]
    for gkey, g in groups:
        g = g.copy()
        sort_cols = [c for c in ("time", "route_s_m", "distance") if c in g.columns and g[c].notna().any()]
        g = g.sort_values(sort_cols[0], kind="stable") if sort_cols else g.sort_index(kind="stable")
        X = g[features].to_numpy(np.float32)
        mu = g["mu_ref"].to_numpy(np.float32)
        uid = g["sample_uid"].astype(str).to_numpy() if "sample_uid" in g.columns else np.asarray(
            [f"{gkey}:{split}:{i}" for i in range(len(g))], dtype=str
        )
        for end in range(first, len(g), int(stride)):
            start = end - w + 1
            window = X[start:end + 1]
            if not np.isfinite(window).all() or not np.isfinite(mu[start:end + 1]).all():
                continue
            # H responses at positions C..C+H-1, each relative to its preceding sample.
            response_block = window[context_length:, ridx] - window[context_length - 1:-1, ridx]
            mu_block = mu[start + context_length:end + 1]
            if response_block.shape != (max_horizon, len(response_channels)) or len(mu_block) != max_horizon:
                continue
            xs.append(window)
            ys.append(float(mu[end]))
            mus.append(mu_block.astype(np.float32))
            responses.append(response_block.astype(np.float32))
            ids.append(str(uid[end]))
    return (
        np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32),
        np.asarray(mus, dtype=np.float32), np.asarray(responses, dtype=np.float32),
        np.asarray(ids, dtype=str),
    )


def make_frc_bundle(csv_path, cfg: dict, *, eval_start: int | None = None) -> FRCBundle:
    fc = cfg.get("frc", {})
    c = int(fc.get("context_length", 16))
    horizons = sorted({int(x) for x in fc.get("horizons", [4, 8, 16, 32])})
    if not horizons or horizons[0] <= 0:
        raise ValueError("frc.horizons must contain positive integers")
    hmax = int(max(horizons))
    df = pd.read_csv(csv_path)
    features = _numeric_features(df)
    requested = list(fc.get("response_channels", ["ax", "ay", "wheel_fl", "wheel_fr", "wheel_rl", "wheel_rr"]))
    response_channels = [x for x in requested if x in features]
    min_channels = int(fc.get("min_response_channels", 2))
    if len(response_channels) < min_channels:
        raise RuntimeError(
            f"SafeGrip-FRC requires at least {min_channels} configured response channels; "
            f"found {response_channels}. Available numeric features: {features}"
        )
    start = int(common_eval_start(cfg) if eval_start is None else eval_start)
    stride = int(cfg.get("stride", 1))
    raw = {
        sp: _frc_windows_for_split(df, features, response_channels, sp, c, hmax, stride, start)
        for sp in ("train", "calibration", "validation", "test")
    }
    if min(len(raw["train"][0]), len(raw["calibration"][0]), len(raw["validation"][0]), len(raw["test"][0])) == 0:
        raise RuntimeError("One FRC split has no common-evaluation windows")

    Xtr = raw["train"][0]
    scaler = StandardScaler().fit(Xtr.reshape(-1, len(features)))
    def scale_x(x):
        return scaler.transform(x.reshape(-1, len(features))).reshape(x.shape).astype(np.float32)
    # Response normalization is fit on train responses only and then frozen.
    rtrain = raw["train"][3].reshape(-1, len(response_channels)).astype(np.float64)
    rmean = np.mean(rtrain, axis=0)
    rstd = np.std(rtrain, axis=0)
    rstd = np.where(rstd < 1e-6, 1.0, rstd)
    def scale_r(r):
        return ((r - rmean[None, None, :]) / rstd[None, None, :]).astype(np.float32)

    vals = {}
    for sp in raw:
        x, y, mu, r, ids = raw[sp]
        vals[sp] = (scale_x(x), y, mu, scale_r(r), ids)
    return FRCBundle(
        features=features, response_channels=response_channels, scaler=scaler,
        response_mean=rmean.astype(np.float32), response_std=rstd.astype(np.float32),
        context_length=c, max_horizon=hmax, eval_start=start,
        Xtr=vals["train"][0], ytr=vals["train"][1], mutr=vals["train"][2], Rtr=vals["train"][3], idtr=vals["train"][4],
        Xc=vals["calibration"][0], yc=vals["calibration"][1], muc=vals["calibration"][2], Rc=vals["calibration"][3], idc=vals["calibration"][4],
        Xv=vals["validation"][0], yv=vals["validation"][1], muv=vals["validation"][2], Rv=vals["validation"][3], idv=vals["validation"][4],
        Xt=vals["test"][0], yt=vals["test"][1], mut=vals["test"][2], Rt=vals["test"][3], idt=vals["test"][4],
    )


def _frc_hparams(cfg: dict, overrides: dict | None = None) -> dict:
    p = dict(cfg.get("frc", {}))
    if overrides:
        p.update(overrides)
    p.setdefault("hidden", 128)
    p.setdefault("gru_layers", 1)
    p.setdefault("dropout", 0.1)
    p.setdefault("lr", 1e-3)
    p.setdefault("weight_decay", 1e-4)
    p.setdefault("batch_size", 128)
    p.setdefault("epochs", cfg.get("training", {}).get("epochs_paper", 60))
    p.setdefault("patience", cfg.get("training", {}).get("patience", 10))
    return p


def fit_frc_response_model(bundle: FRCBundle, cfg: dict, hp_overrides: dict | None = None):
    hp = _frc_hparams(cfg, hp_overrides)
    dev = device()
    model = FrictionResponseNet(
        len(bundle.features), len(bundle.response_channels), hidden=int(hp["hidden"]),
        gru_layers=int(hp["gru_layers"]), dropout=float(hp["dropout"]),
        mu_upper=float(cfg.get("mu_upper", 1.3)),
    ).to(dev)
    train_ds = _TransitionDataset(bundle.Xtr, bundle.mutr, bundle.Rtr, bundle.context_length)
    val_ds = _TransitionDataset(bundle.Xv, bundle.muv, bundle.Rv, bundle.context_length)
    gen = torch.Generator().manual_seed(int(cfg.get("seed", 0)))
    dl = DataLoader(train_ds, batch_size=int(hp["batch_size"]), shuffle=True, generator=gen)
    vl = DataLoader(val_ds, batch_size=max(256, int(hp["batch_size"])), shuffle=False)
    opt = torch.optim.AdamW(model.parameters(), lr=float(hp["lr"]), weight_decay=float(hp["weight_decay"]))
    best = None; bestloss = float("inf"); bad = 0
    for _ in range(int(hp["epochs"])):
        model.train()
        for xb, mub, rb in dl:
            xb = xb.to(dev); mub = mub.to(dev); rb = rb.to(dev)
            opt.zero_grad(set_to_none=True)
            pred = model(xb, mub)
            loss = nn.functional.mse_loss(pred, rb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        model.eval(); total = 0.0; count = 0
        with torch.no_grad():
            for xb, mub, rb in vl:
                xb = xb.to(dev); mub = mub.to(dev); rb = rb.to(dev)
                loss = nn.functional.mse_loss(model(xb, mub), rb, reduction="sum")
                total += float(loss.item()); count += int(rb.numel())
        score = total / max(1, count)
        if score < bestloss - 1e-8:
            bestloss = score; bad = 0
            best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if bad >= int(hp["patience"]):
            break
    if best is not None:
        model.load_state_dict(best)
    return model, hp, bestloss


def _contexts_from_windows(X: np.ndarray, context_length: int, max_horizon: int) -> np.ndarray:
    # [N,H,C,D], with response j predicted from X[j:j+C].
    return np.stack([X[:, j:j + context_length, :] for j in range(max_horizon)], axis=1)


def predict_signatures(model: FrictionResponseNet, X: np.ndarray, mu_grid: np.ndarray,
                       context_length: int, max_horizon: int, *, context_batch: int = 96) -> np.ndarray:
    """Return predicted response signatures [N,M,H,R]."""
    dev = device(); model.eval()
    ctx = _contexts_from_windows(X, context_length, max_horizon)
    n, h, c, d = ctx.shape
    flat = ctx.reshape(n * h, c, d)
    m = len(mu_grid)
    out_chunks = []
    grid = torch.as_tensor(mu_grid, dtype=torch.float32, device=dev)
    with torch.no_grad():
        for s in range(0, len(flat), int(context_batch)):
            xb = torch.from_numpy(flat[s:s + context_batch]).to(dev)
            q = len(xb)
            repeated = xb[:, None, :, :].expand(q, m, c, d).reshape(q * m, c, d)
            mus = grid[None, :].expand(q, m).reshape(q * m)
            pred = model(repeated, mus).reshape(q, m, -1).cpu().numpy()
            out_chunks.append(pred)
    pred = np.concatenate(out_chunks, axis=0).reshape(n, h, m, -1)
    return np.transpose(pred, (0, 2, 1, 3)).astype(np.float32)


def predict_signature_at_mu(model: FrictionResponseNet, X: np.ndarray, mu: np.ndarray,
                            context_length: int, max_horizon: int, *, batch: int = 1024) -> np.ndarray:
    """Response signature for one constant friction hypothesis per window."""
    dev = device(); model.eval()
    ctx = _contexts_from_windows(X, context_length, max_horizon)
    n, h, c, d = ctx.shape
    flat = ctx.reshape(n * h, c, d)
    muf = np.repeat(np.asarray(mu, dtype=np.float32), h)
    ps = []
    with torch.no_grad():
        for s in range(0, len(flat), int(batch)):
            xb = torch.from_numpy(flat[s:s + batch]).to(dev)
            mb = torch.from_numpy(muf[s:s + batch]).to(dev)
            ps.append(model(xb, mb).cpu().numpy())
    return np.concatenate(ps, axis=0).reshape(n, h, -1).astype(np.float32)


def calibrate_residual_radii(model: FrictionResponseNet, bundle: FRCBundle, cfg: dict,
                              mu_grid: np.ndarray) -> dict[int, float]:
    fc = cfg.get("frc", {})
    horizons = sorted({int(x) for x in fc.get("horizons", [4, 8, 16, 32])})
    quantile = float(fc.get("residual_quantile", 0.95))
    nearest = nearest_grid_values(bundle.yc, mu_grid)
    pred = predict_signature_at_mu(model, bundle.Xc, nearest, bundle.context_length, bundle.max_horizon)
    radii = {}
    for h in horizons:
        residual = bundle.Rc[:, -h:, :] - pred[:, -h:, :]
        norms = np.sqrt(np.sum(residual ** 2, axis=(1, 2)))
        radii[h] = empirical_residual_radius(norms, quantile)
    return radii


def evaluate_frc_split(model: FrictionResponseNet, X: np.ndarray, R: np.ndarray, y: np.ndarray,
                       bundle: FRCBundle, cfg: dict, mu_grid: np.ndarray,
                       radii: dict[int, float], *, endpoint_batch: int = 64):
    fc = cfg.get("frc", {})
    horizons = sorted({int(x) for x in fc.get("horizons", [4, 8, 16, 32])})
    deltas = sorted({float(x) for x in fc.get("certificate_deltas", [0.02,0.04,0.06,0.08,0.10,0.15,0.20])})
    fallback = int(fc.get("fallback_horizon", max(horizons)))
    grid_step = float(np.max(np.diff(mu_grid)))
    by_h = {h: {"mu": [], "obj": [], "sep": [], "dc": [], "ec": [], "cert": [], "true_resid": []} for h in horizons}
    for s in range(0, len(X), int(endpoint_batch)):
        xb = X[s:s + endpoint_batch]; rb = R[s:s + endpoint_batch]; yb = y[s:s + endpoint_batch]
        sig = predict_signatures(model, xb, mu_grid, bundle.context_length, bundle.max_horizon)
        nearest = nearest_grid_values(yb, mu_grid)
        nearest_idx = np.abs(yb[:, None] - mu_grid[None, :]).argmin(axis=1)
        for h in horizons:
            sh = sig[:, :, -h:, :]
            rh = rb[:, -h:, :]
            obj = np.sum((sh - rh[:, None, :, :]) ** 2, axis=(2, 3))
            idx = np.argmin(obj, axis=1)
            mu_hat = mu_grid[idx]
            sep = separation_margin(sh, mu_grid, deltas, batch_size=max(16, endpoint_batch))
            dc, ec, cert = resolution_certificate(sep, deltas, radii[h], grid_step)
            true_resid = np.sqrt(obj[np.arange(len(obj)), nearest_idx])
            d = by_h[h]
            d["mu"].append(mu_hat); d["obj"].append(obj[np.arange(len(obj)), idx]); d["sep"].append(sep)
            d["dc"].append(dc); d["ec"].append(ec); d["cert"].append(cert); d["true_resid"].append(true_resid)
    results = []
    audits = {}
    for h in horizons:
        d = by_h[h]
        mu = np.concatenate(d["mu"]); obj = np.concatenate(d["obj"]); sep = np.concatenate(d["sep"])
        dc = np.concatenate(d["dc"]); ec = np.concatenate(d["ec"]); cert = np.concatenate(d["cert"])
        tr = np.concatenate(d["true_resid"])
        results.append(ResolutionResult(h, mu, obj, sep, dc, ec, cert))
        audits[h] = {"true_residual": tr, "separation": sep}
    mu_hat, selected_h, selected_ec, selected_cert = select_horizon(results, fallback_horizon=fallback)
    return mu_hat, selected_h, selected_ec, selected_cert, results, audits, deltas


def _fit_direct_control(bundle: FRCBundle, cfg: dict, hp: dict):
    model = DirectGRUControl(
        len(bundle.features), hidden=int(hp["hidden"]), gru_layers=int(hp["gru_layers"]), dropout=float(hp["dropout"])
    )
    train_hp = {
        "optimizer": "adamw", "lr": float(hp["lr"]), "weight_decay": float(hp["weight_decay"]),
        "batch_size": int(hp["batch_size"]), "patience": int(hp["patience"]),
    }
    return _fit_deterministic(model, bundle.Xtr, bundle.ytr, bundle.Xv, bundle.yv, train_hp, int(hp["epochs"]))


def _predict_direct(model, X, batch=1024):
    dev = device(); model.eval(); out=[]
    with torch.no_grad():
        for s in range(0, len(X), batch):
            out.append(model(torch.from_numpy(X[s:s+batch]).to(dev)).cpu().numpy())
    return np.concatenate(out)


def _controlled_baseline_hparams(name: str, cfg: dict, selected: dict | None = None) -> dict:
    hp = literature_hparams(name, cfg, (selected or {}).get(name), "paper")
    cc = cfg.get("comparison", {}).get("controlled", {})
    hp.update({
        "epochs": int(cc.get("epochs", cfg.get("training", {}).get("epochs_paper", 60))),
        "patience": int(cc.get("patience", cfg.get("training", {}).get("patience", 10))),
        "batch_size": int(cc.get("batch_size", 128)),
    })
    # Equal basic optimizer budget; architecture/preprocessing remain method-specific.
    if bool(cc.get("common_optimizer_hparams", True)):
        hp["lr"] = float(cc.get("lr", 1e-3))
        hp["weight_decay"] = float(cc.get("weight_decay", 1e-4))
    return hp


def _source_baseline_hparams(name: str, cfg: dict, selected: dict | None = None) -> dict:
    return literature_hparams(name, cfg, (selected or {}).get(name), "paper")


def _common_eval_start_frc(cfg: dict, models: list[str]) -> int:
    fc = cfg.get("frc", {})
    w = int(fc.get("context_length", 16)) + max(int(x) for x in fc.get("horizons", [4,8,16,32]))
    seqs = [w, int(common_eval_start(cfg)) + 1]
    for name in models:
        seqs.append(int(cfg.get("baseline", {}).get(name, {}).get("sequence_length", cfg.get("sequence_length", 100))))
    return max(seqs) - 1


def _endpoint_hash(ids: np.ndarray) -> str:
    return hashlib.sha256("\n".join(np.asarray(ids, dtype=str).tolist()).encode()).hexdigest()


def run_frc_benchmark(csv_path, out_dir, cfg: dict, preset: str = "paper", models=None,
                      frc_hparams: dict | None = None, baseline_hparams: dict | None = None,
                      protocol: str = "controlled"):
    """Run SafeGrip-FRC plus independent literature comparators.

    ``controlled`` gives the literature models a common optimization budget.
    ``source_faithful`` preserves each comparator's configured literature-style
    training budget.  Both protocols use identical locked endpoint IDs.
    """
    protocol = str(protocol).replace("-", "_")
    if protocol not in {"controlled", "source_faithful"}:
        raise ValueError("protocol must be controlled or source_faithful")
    out = ensure_dir(out_dir)
    names = list(models) if models is not None else list(QUICK_BASELINES if preset == "quick" else PAPER_BASELINES)
    validate_paper_baselines(names)
    eval_start = _common_eval_start_frc(cfg, names)
    if baseline_hparams:
        selected_lengths=[int(v.get("sequence_length",0)) for v in baseline_hparams.values() if isinstance(v,dict)]
        if selected_lengths: eval_start=max(eval_start,max(selected_lengths)-1)
    bundle = make_frc_bundle(csv_path, cfg, eval_start=eval_start)
    fc_hp = _frc_hparams(cfg, frc_hparams)
    if not frc_hparams or "epochs" not in frc_hparams:
        if preset=="quick": fc_hp["epochs"]=int(cfg.get("training",{}).get("epochs_quick",3))
        elif preset=="trust": fc_hp["epochs"]=int(cfg.get("training",{}).get("epochs_trust",40))
    if preset=="paper":
        seeds=list(cfg.get("comparison",{}).get("controlled",{}).get("evaluation_seeds",cfg.get("evaluation",{}).get("seeds",[0,1,2,3,4])))
    else:
        seeds = list(cfg.get("evaluation", {}).get("seeds", [0,1,2,3,4]))
    if preset == "quick": seeds = seeds[:1]
    elif preset == "trust": seeds = list(cfg.get("evaluation", {}).get("trust_seeds", seeds[:3]))
    mu_grid = make_mu_grid(
        float(cfg.get("frc", {}).get("mu_min", 0.0)),
        float(cfg.get("frc", {}).get("mu_max", cfg.get("mu_upper", 1.3))),
        float(cfg.get("frc", {}).get("mu_grid_step", 0.02)),
    )

    # Reuse the existing bundle builder only for equal physical-projection control.
    ref_bundle = make_bundle(csv_path, cfg, sequence_length=bundle.context_length + bundle.max_horizon,
                             scaler_kind="standard", eval_start=eval_start, feature_mode="raw")
    if not np.array_equal(ref_bundle.idt, bundle.idt) or not np.allclose(ref_bundle.yt, bundle.yt):
        raise RuntimeError("FRC and repository benchmark builders disagree on locked test endpoints")

    rows=[]; pred_rows=[]; fixed_rows=[]; cert_rows=[]; sep_rows=[]; theorem_rows=[]; projection_rows=[]
    frc_models=[]; baseline_selected=baseline_hparams or {}

    for seed in seeds:
        seed_everything(int(seed)); t0=time.time()
        model, hp, val_loss = fit_frc_response_model(bundle, cfg, fc_hp)
        radii = calibrate_residual_radii(model, bundle, cfg, mu_grid)
        mu_hat, selected_h, selected_ec, selected_cert, hresults, audits, deltas = evaluate_frc_split(
            model, bundle.Xt, bundle.Rt, bundle.yt, bundle, cfg, mu_grid, radii
        )
        rows.append({"model":"safegrip_frc","kind":"proposal","doi":"","seed":int(seed),
                     **regression_metrics(bundle.yt, mu_hat), "seconds":time.time()-t0,
                     "mean_error_certificate":float(np.mean(selected_ec[np.isfinite(selected_ec)])) if np.isfinite(selected_ec).any() else float("nan"),
                     "certificate_rate":float(np.mean(selected_cert))})
        pred_rows.append(pd.DataFrame({"endpoint_id":bundle.idt,"y_true":bundle.yt,"model":"safegrip_frc","seed":int(seed),
                                       "prediction":mu_hat,"selected_horizon":selected_h,"error_certificate":selected_ec,
                                       "certified":selected_cert.astype(int)}))
        projection_rows.append({"model":"safegrip_frc__projection_control","source_model":"safegrip_frc","seed":int(seed),
                                **regression_metrics(bundle.yt, project_numpy(mu_hat, ref_bundle.lot, float(cfg["mu_upper"])))})
        nearest = nearest_grid_values(bundle.yt, mu_grid)
        for res in hresults:
            h=int(res.horizon); true_resid=audits[h]["true_residual"]
            fixed_rows.append({"model":f"frc_fixed_h{h}","kind":"fixed_horizon_ablation","doi":"","seed":int(seed),
                               **regression_metrics(bundle.yt,res.mu_hat),
                               "certificate_rate":float(np.mean(res.certified)),
                               "mean_error_certificate":float(np.mean(res.error_certificate[np.isfinite(res.error_certificate)])) if np.isfinite(res.error_certificate).any() else float("nan")})
            for di,delta in enumerate(deltas):
                condition = res.separation[:,di] > 2.0 * float(radii[h])
                premise = condition & (true_resid <= float(radii[h]) + 1e-7)
                conclusion_grid = np.abs(res.mu_hat-nearest) < float(delta) + 1e-6
                conclusion_cont = np.abs(res.mu_hat-bundle.yt) <= float(delta) + float(np.max(np.diff(mu_grid)))/2.0 + 1e-6
                theorem_rows.append({
                    "seed":int(seed),"horizon":h,"delta":float(delta),"residual_radius":float(radii[h]),
                    "separation_condition_rate":float(np.mean(condition)),"residual_premise_rate":float(np.mean(true_resid<=float(radii[h])+1e-7)),
                    "full_premise_count":int(np.sum(premise)),"grid_theorem_violations":int(np.sum(premise & ~conclusion_grid)),
                    "continuous_bound_violations":int(np.sum(premise & ~conclusion_cont)),
                })
                sep_rows.append({"seed":int(seed),"horizon":h,"delta":float(delta),
                                 "mean_separation":float(np.mean(res.separation[:,di])),
                                 "median_separation":float(np.median(res.separation[:,di])),
                                 "q25_separation":float(np.quantile(res.separation[:,di],.25)),
                                 "q75_separation":float(np.quantile(res.separation[:,di],.75)),
                                 "condition_rate":float(np.mean(condition))})
            cert_rows.append(pd.DataFrame({
                "endpoint_id":bundle.idt,"seed":int(seed),"horizon":h,"mu_true":bundle.yt,
                "mu_hat":res.mu_hat,"nearest_grid_mu":nearest,"absolute_error":np.abs(res.mu_hat-bundle.yt),
                "true_residual_norm":true_resid,"residual_radius":float(radii[h]),
                "delta_certificate":res.delta_certificate,"error_certificate":res.error_certificate,
                "certified":res.certified.astype(int),
            }))
        frc_models.append(model)

        # Same-family direct-regression control: same seed and FRC capacity/budget.
        seed_everything(int(seed))
        direct = _fit_direct_control(bundle, cfg, fc_hp)
        dp = _predict_direct(direct, bundle.Xt)
        rows.append({"model":"direct_gru_control","kind":"same_encoder_control","doi":"","seed":int(seed),
                     **regression_metrics(bundle.yt,dp)})
        pred_rows.append(pd.DataFrame({"endpoint_id":bundle.idt,"y_true":bundle.yt,"model":"direct_gru_control","seed":int(seed),"prediction":dp}))
        projection_rows.append({"model":"direct_gru_control__projection_control","source_model":"direct_gru_control","seed":int(seed),
                                **regression_metrics(bundle.yt,project_numpy(dp,ref_bundle.lot,float(cfg["mu_upper"])))})

    # Independent literature baselines.
    for mi,name in enumerate(names):
        hp = (_controlled_baseline_hparams(name,cfg,baseline_selected) if protocol=="controlled"
              else _source_baseline_hparams(name,cfg,baseline_selected))
        if preset=="quick":
            hp["epochs"]=int(cfg.get("training",{}).get("epochs_quick",3)); hp["patience"]=max(1,min(int(hp.get("patience",3)),3))
        elif preset=="trust":
            hp["epochs"]=int(cfg.get("training",{}).get("epochs_trust",40)); hp["patience"]=int(cfg.get("training",{}).get("patience_trust",8))
        b = make_bundle(csv_path,cfg,sequence_length=int(hp["sequence_length"]),scaler_kind=str(hp["scaler"]),
                        eval_start=eval_start,feature_mode="raw")
        if not np.array_equal(b.idt,bundle.idt) or not np.array_equal(b.idv,bundle.idv):
            raise RuntimeError(f"{name} endpoint IDs differ from FRC endpoints")
        if not np.allclose(b.yt,bundle.yt) or not np.allclose(b.yv,bundle.yv):
            raise RuntimeError(f"{name} target values differ on locked endpoints")
        for seed in seeds:
            seed_everything(int(seed)); t0=time.time()
            bm=fit_literature(name,b,cfg,epochs=int(hp["epochs"]),preset="paper",hp_overrides=hp)
            bp,bs=predict_literature(bm,name,b.Xt)
            rows.append({"model":name,"kind":"literature_adapted_baseline","doi":LITERATURE_BASELINES[name]["doi"],"seed":int(seed),
                         **regression_metrics(b.yt,bp,sigma=bs),"seconds":time.time()-t0})
            pred_rows.append(pd.DataFrame({"endpoint_id":b.idt,"y_true":b.yt,"model":name,"seed":int(seed),"prediction":bp}))
            projection_rows.append({"model":name+"__projection_control","source_model":name,"seed":int(seed),
                                    **regression_metrics(b.yt,project_numpy(bp,ref_bundle.lot,float(cfg["mu_upper"])),sigma=bs)})

    per_seed=pd.DataFrame(rows); metrics=_aggregate_seed_metrics(rows)
    metrics.to_csv(out/"metrics.csv",index=False); per_seed.to_csv(out/"metrics_by_seed.csv",index=False)
    pd.concat(pred_rows,ignore_index=True).to_csv(out/"predictions_by_seed.csv",index=False)
    _aggregate_seed_metrics(fixed_rows).to_csv(out/"fixed_horizon_ablation.csv",index=False)
    pd.DataFrame(fixed_rows).to_csv(out/"fixed_horizon_ablation_by_seed.csv",index=False)
    pd.concat(cert_rows,ignore_index=True).to_csv(out/"frc_resolution_certificates.csv",index=False)
    pd.DataFrame(sep_rows).to_csv(out/"frc_separation_curves.csv",index=False)
    theorem_df=pd.DataFrame(theorem_rows); theorem_df.to_csv(out/"frc_certificate_validity.csv",index=False)
    _aggregate_seed_metrics(projection_rows).to_csv(out/"projection_control.csv",index=False)

    violations=int(theorem_df["grid_theorem_violations"].sum()) if len(theorem_df) else 0
    cont_violations=int(theorem_df["continuous_bound_violations"].sum()) if len(theorem_df) else 0
    theorem_audit={
        "status":"PASS" if violations==0 and cont_violations==0 else "FAIL",
        "grid_theorem_violations":violations,"continuous_bound_violations":cont_violations,
        "claim":"conditional finite-grid recovery; empirical calibration radius is not itself a universal probabilistic guarantee",
        "grid_step":float(np.max(np.diff(mu_grid))),"candidate_count":int(len(mu_grid)),
        "test_labels_used_for_prediction_or_horizon_selection":False,
    }
    (out/"theorem_audit.json").write_text(json.dumps(theorem_audit,indent=2),encoding="utf-8")
    fairness={
        "status":"PASS",
        "protocol":protocol,"seeds":seeds,"common_eval_start":int(eval_start),
        "validation_endpoint_hash":_endpoint_hash(bundle.idv),"test_endpoint_hash":_endpoint_hash(bundle.idt),
        "same_locked_endpoints":True,"test_labels_used_for_tuning":False,
        "frc_response_scaler_fit_on_train_only":True,"frc_response_normalization_fit_on_train_only":True,
        "frc_context_strictly_precedes_predicted_response":True,
        "primary_metrics_use_physical_projection":False,
        "projection_control_reported_separately":True,
        "controlled_budget":({
            **cfg.get("comparison",{}).get("controlled",{}),
            "effective_preset":preset,
            "effective_frc_epochs":int(fc_hp["epochs"]),
            "effective_literature_epochs":int(cfg.get("training",{}).get("epochs_quick",3) if preset=="quick" else (cfg.get("training",{}).get("epochs_trust",40) if preset=="trust" else cfg.get("comparison",{}).get("controlled",{}).get("epochs",cfg.get("training",{}).get("epochs_paper",60)))),
        } if protocol=="controlled" else None),
        "baseline_fidelity":{n:LITERATURE_BASELINES[n]["fidelity"] for n in names},
    }
    (out/"fairness_audit.json").write_text(json.dumps(fairness,indent=2),encoding="utf-8")
    manifest={
        "proposal":"SafeGrip-FRC: finite-window friction resolution certification",
        "mathematical_core":["J_H(mu)","S_H(delta)","delta_cert","certificate-guided H*"],
        "frc_hparams":fc_hp,"mu_grid":mu_grid.tolist(),"response_channels":bundle.response_channels,
        "features":bundle.features,"response_mean":bundle.response_mean.tolist(),"response_std":bundle.response_std.tolist(),
        "legacy_safegrip_ci_v14_retained":True,"primary_protocol":protocol,
    }
    (out/"reproducibility_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    if theorem_audit["status"]!="PASS":
        raise RuntimeError("FRC theorem audit failed; numerical implementation contradicts the stated grid theorem")
    return metrics
