from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .physics import vehicle_level_lower_bound
from .utils import ensure_dir, normalize_name


def read_table(path: Path) -> pd.DataFrame:
    last = None
    for enc in ("utf-8-sig", "utf-8", "latin1"):
        try:
            return pd.read_csv(path, sep=None, engine="python", encoding=enc, on_bad_lines="skip")
        except Exception as exc:  # pragma: no cover - only reached for malformed external files
            last = exc
    raise last


def _normmap(df):
    return {c: normalize_name(c) for c in df.columns}


def find_col(df: pd.DataFrame, groups, exclude=()):
    nm = _normmap(df)
    excludes = [normalize_name(x) for x in exclude]
    for alt in groups:
        toks = alt if isinstance(alt, (tuple, list)) else (alt,)
        toks = [normalize_name(t) for t in toks]
        for c, n in nm.items():
            if all(t in n for t in toks) and not any(x in n for x in excludes):
                return c
    return None


CANONICAL = {
    "time": [("timestamp",), ("time",), ("tid",)],
    "distance": [("totaldist",), ("distance",), ("dist",)],
    "lat": [("latitude",), ("gps", "lat"), ("lat",)],
    "lon": [("longitude",), ("gps", "lon"), ("lon",)],
    "speed": [("vehicle", "speed"), ("carspeed",), ("vehspd",), ("speed",)],
    "ax": [("longitudinal", "acc"), ("acceleration", "x"), ("accel", "x"), ("ax",)],
    "ay": [("lateral", "acc"), ("acceleration", "y"), ("accel", "y"), ("ay",)],
    "yaw_rate": [("yaw", "rate"), ("yawrate",)],
    "steer": [("steer", "angle"), ("steering",), ("steer",)],
    "wheel_fl": [("wheel", "speed", "front", "left"), ("wheelspeedfl",), ("fl", "wheel", "speed")],
    "wheel_fr": [("wheel", "speed", "front", "right"), ("wheelspeedfr",), ("fr", "wheel", "speed")],
    "wheel_rl": [("wheel", "speed", "rear", "left"), ("wheelspeedrl",), ("rl", "wheel", "speed")],
    "wheel_rr": [("wheel", "speed", "rear", "right"), ("wheelspeedrr",), ("rr", "wheel", "speed")],
    "torque": [("estimated", "torque"), ("requested", "torque"), ("drive", "torque"), ("motortorque",), ("torque",)],
    "brake_torque": [("brake", "wheel", "torque"), ("brake", "torque")],
    "pressure_fl": [("pressure", "front", "left"), ("tirepressurefl",)],
    "pressure_fr": [("pressure", "front", "right"), ("tirepressurefr",)],
    "pressure_rl": [("pressure", "rear", "left"), ("tirepressurerl",)],
    "pressure_rr": [("pressure", "rear", "right"), ("tirepressurerr",)],
}


def _to_numeric(s):
    return pd.to_numeric(s, errors="coerce")


def _time_seconds(s: pd.Series) -> pd.Series:
    """Convert common numeric/datetime timestamp formats to relative seconds.

    The conversion deliberately uses only within-file information. If a numeric
    stream cannot be interpreted reliably it is still returned as a monotone
    relative coordinate; resampling later checks whether the implied rate is
    plausible before using it.
    """
    num = pd.to_numeric(s, errors="coerce")
    if num.notna().sum() >= max(3, len(s) // 3):
        x = num.astype(float).to_numpy()
        finite = x[np.isfinite(x)]
        if finite.size:
            mag = float(np.nanmedian(np.abs(finite)))
            dif = np.diff(finite)
            dif = dif[np.isfinite(dif) & (dif > 0)]
            md = float(np.nanmedian(dif)) if dif.size else 1.0
            scale = 1.0
            if mag > 1e17:
                scale = 1e9
            elif mag > 1e14:
                scale = 1e6
            elif mag > 1e11:
                scale = 1e3
            elif md > 5.0 and md <= 5000.0:
                # Common vehicle logs store elapsed milliseconds (20--100 ms).
                scale = 1e3
            x = x / scale
            first = x[np.isfinite(x)][0]
            return pd.Series(x - first, index=s.index, dtype=float)
    dt = pd.to_datetime(s, errors="coerce", utc=True)
    if dt.notna().sum() >= 3:
        x = dt.astype("int64").astype(float) / 1e9
        x[dt.isna().to_numpy()] = np.nan
        finite = x[np.isfinite(x)]
        first = finite[0] if finite.size else 0.0
        return pd.Series(x - first, index=s.index, dtype=float)
    return num.astype(float)


def canonical_vehicle(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for key, pats in CANONICAL.items():
        c = find_col(df, pats)
        if c is None:
            continue
        if key == "time":
            out[key] = _time_seconds(df[c])
            continue
        values = _to_numeric(df[c])
        unit = str(c).lower().replace(" ", "")
        if key == "speed":
            if any(tok in unit for tok in ("km/h", "kmh", "kph", "kmph")):
                values = values / 3.6
            elif "mph" in unit:
                values = values * 0.44704
        elif key in ("ax", "ay"):
            raw_name = str(c).lower()
            if re.search(r"(?:\[|\()\s*g\s*(?:\]|\))", raw_name):
                values = values * 9.80665
        out[key] = values
    return out


def canonical_friction(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    aliases = {
        "time": [("tid",), ("time",)],
        "distance": [("totaldist",), ("distance",)],
        "lat": [("lat",)],
        "lon": [("lon",)],
        "mu_l": [("muv",), ("frictioncoefficient", "left")],
        "mu_r": [("muh",), ("frictioncoefficient", "right")],
        "fz_l": [("fvertikalv",), ("vertical", "left")],
        "fz_r": [("fvertikalh",), ("vertical", "right")],
        "fx_l": [("ffriksjonv",), ("frictional", "left")],
        "fx_r": [("ffriksjonh",), ("frictional", "right")],
        "slip_l": [("slipv",), ("sliprate", "left")],
        "slip_r": [("sliph",), ("sliprate", "right")],
    }
    for k, pats in aliases.items():
        c = find_col(df, pats)
        if c is not None:
            out[k] = _time_seconds(df[c]) if k == "time" else _to_numeric(df[c])
    mus = [c for c in ("mu_l", "mu_r") if c in out]
    if mus:
        out["mu_ref"] = out[mus].mean(axis=1)
    return out


def parse_lira_context(path: Path, root: Path | None = None) -> dict[str, str]:
    """Extract route/direction identifiers only when they are explicit in paths.

    Unknown metadata remains ``unknown``; the function never invents a route
    from GPS coordinates or file ordering.
    """
    try:
        rel = path.relative_to(root) if root is not None else path
    except ValueError:
        rel = path
    text = rel.as_posix().upper()
    route = "unknown"
    m = re.search(r"(?:^|[^A-Z0-9])(CPH\s*[-_]?\s*\d+|M\s*[-_]?\s*\d+)(?:[^A-Z0-9]|$)", text)
    if m:
        route = re.sub(r"[^A-Z0-9]", "", m.group(1))
    direction = "unknown"
    direction_patterns = [
        (r"(?:^|[^A-Z0-9])(HH)(?:[^A-Z0-9]|$)", "HH"),
        (r"(?:^|[^A-Z0-9])(VH)(?:[^A-Z0-9]|$)", "VH"),
        (r"(?:^|[^A-Z0-9])(NORTHBOUND|NB)(?:[^A-Z0-9]|$)", "NB"),
        (r"(?:^|[^A-Z0-9])(SOUTHBOUND|SB)(?:[^A-Z0-9]|$)", "SB"),
        (r"(?:^|[^A-Z0-9])(EASTBOUND|EB)(?:[^A-Z0-9]|$)", "EB"),
        (r"(?:^|[^A-Z0-9])(WESTBOUND|WB)(?:[^A-Z0-9]|$)", "WB"),
    ]
    for pat, value in direction_patterns:
        if re.search(pat, text):
            direction = value
            break
    trip = normalize_name(rel.with_suffix("").as_posix()).strip("_") or normalize_name(path.stem)
    return {"route_id": route, "direction": direction, "trip_id": trip}


def haversine_project(lat, lon, lat0=None):
    lat = np.asarray(lat, float)
    lon = np.asarray(lon, float)
    lat0 = np.nanmedian(lat) if lat0 is None else lat0
    radius = 6371000.0
    x = np.deg2rad(lon) * radius * np.cos(np.deg2rad(lat0))
    y = np.deg2rad(lat) * radius
    return np.c_[x, y]


def cumulative_route_distance(lat, lon) -> np.ndarray:
    xy = haversine_project(lat, lon)
    if len(xy) == 0:
        return np.empty(0, dtype=float)
    step = np.hypot(np.diff(xy[:, 0]), np.diff(xy[:, 1]))
    step = np.nan_to_num(step, nan=0.0, posinf=0.0, neginf=0.0)
    return np.r_[0.0, np.cumsum(step)]


def geometric_heading_deg(lat, lon) -> np.ndarray:
    xy = haversine_project(lat, lon)
    n = len(xy)
    if n == 0:
        return np.empty(0, dtype=float)
    if n == 1:
        return np.full(1, np.nan)
    dx = np.gradient(xy[:, 0])
    dy = np.gradient(xy[:, 1])
    heading = (np.degrees(np.arctan2(dx, dy)) + 360.0) % 360.0
    bad = ~np.isfinite(dx) | ~np.isfinite(dy) | (np.hypot(dx, dy) < 1e-6)
    heading[bad] = np.nan
    return heading


def _heading_diff_deg(a, b):
    return np.abs((a - b + 180.0) % 360.0 - 180.0)


def _copy_reference_columns(cc: pd.DataFrame, rr: pd.DataFrame) -> pd.DataFrame:
    preferred = {"mu_ref", "mu_l", "mu_r", "fz_l", "fz_r", "fx_l", "fx_r", "slip_l", "slip_r"}
    for col in rr.columns:
        if col.startswith("_"):
            continue
        if col not in cc or col in preferred:
            dest = f"ref_{col}" if col in cc else col
            cc[dest] = rr[col].to_numpy()
    return cc


def spatial_align(
    car: pd.DataFrame,
    ref: pd.DataFrame,
    max_m: float = 10.0,
    *,
    heading_tolerance_deg: float | None = 45.0,
    enforce_monotonic: bool = True,
    k_candidates: int = 8,
) -> pd.DataFrame:
    """Route-local spatial alignment with optional heading/monotonic checks.

    The function is deliberately independent of route naming. Callers should
    first restrict ``ref`` to the same known route/direction when metadata are
    available. When GPS is unavailable, distance-asof matching is used with a
    metric tolerance rather than an unconstrained global join.
    """
    if all(c in car for c in ("lat", "lon")) and all(c in ref for c in ("lat", "lon")):
        cvalid = car[["lat", "lon"]].notna().all(axis=1)
        rvalid = ref[["lat", "lon"]].notna().all(axis=1)
        c = car.loc[cvalid].copy().reset_index(drop=True)
        r = ref.loc[rvalid].copy().reset_index(drop=True)
        if c.empty or r.empty:
            return pd.DataFrame()
        lat0 = np.nanmedian(np.r_[c.lat.to_numpy(), r.lat.to_numpy()])
        cxy = haversine_project(c.lat, c.lon, lat0)
        rxy = haversine_project(r.lat, r.lon, lat0)
        if "route_s_m" not in c:
            c["route_s_m"] = cumulative_route_distance(c.lat, c.lon)
        if "ref_route_s_m" not in r:
            r["ref_route_s_m"] = cumulative_route_distance(r.lat, r.lon)
        chead = geometric_heading_deg(c.lat, c.lon)
        rhead = geometric_heading_deg(r.lat, r.lon)
        c["gps_heading_deg"] = chead
        r["ref_gps_heading_deg"] = rhead
        tree = cKDTree(rxy)
        k = max(1, min(int(k_candidates), len(r)))
        dists, idxs = tree.query(cxy, k=k)
        if k == 1:
            dists = dists[:, None]
            idxs = idxs[:, None]
        chosen_car = []
        chosen_ref = []
        chosen_dist = []
        last_ref = -1
        for i in range(len(c)):
            order = np.argsort(dists[i])
            pick = None
            for j in order:
                dist = float(dists[i, j])
                ridx = int(idxs[i, j])
                if dist > float(max_m):
                    continue
                if enforce_monotonic and ridx < last_ref:
                    continue
                if heading_tolerance_deg is not None and np.isfinite(chead[i]) and np.isfinite(rhead[ridx]):
                    if _heading_diff_deg(chead[i], rhead[ridx]) > float(heading_tolerance_deg):
                        continue
                pick = ridx
                break
            if pick is None:
                continue
            chosen_car.append(i)
            chosen_ref.append(pick)
            chosen_dist.append(float(np.linalg.norm(cxy[i] - rxy[pick])))
            if enforce_monotonic:
                last_ref = max(last_ref, pick)
        if not chosen_car:
            return pd.DataFrame()
        cc = c.iloc[chosen_car].reset_index(drop=True)
        rr = r.iloc[chosen_ref].reset_index(drop=True)
        cc = _copy_reference_columns(cc, rr)
        cc["match_distance_m"] = np.asarray(chosen_dist, dtype=float)
        cc["match_ref_index"] = np.asarray(chosen_ref, dtype=int)
        return cc
    if "distance" in car and "distance" in ref:
        c = car.dropna(subset=["distance"]).sort_values("distance").copy()
        r = ref.dropna(subset=["distance"]).sort_values("distance").copy()
        z = pd.merge_asof(
            c,
            r,
            on="distance",
            direction="nearest",
            tolerance=float(max_m),
            suffixes=("", "_ref"),
        ).reset_index(drop=True)
        if "mu_ref" in z:
            z = z.dropna(subset=["mu_ref"]).reset_index(drop=True)
        z["route_s_m"] = z["distance"] - z["distance"].min()
        return z
    raise RuntimeError("LiRA alignment requires GPS or distance columns; inspect raw schema and extend aliases if upstream names changed.")


def _select_reference_pool(ref: pd.DataFrame, route_id: str, direction: str) -> pd.DataFrame:
    pool = ref
    if route_id != "unknown" and "route_id" in pool and (pool.route_id == route_id).any():
        pool = pool[pool.route_id == route_id]
    if direction != "unknown" and "direction" in pool and (pool.direction == direction).any():
        pool = pool[pool.direction == direction]
    return pool.reset_index(drop=True)


def _select_reference_trace_by_geometry(car: pd.DataFrame, pool: pd.DataFrame) -> pd.DataFrame:
    """Choose one reference trace when several files share route/direction metadata."""
    if "ref_source_file" not in pool or pool.ref_source_file.nunique(dropna=True) <= 1:
        return pool.reset_index(drop=True)
    if all(c in car for c in ("lat", "lon")) and car[["lat", "lon"]].notna().any().all():
        c = car.dropna(subset=["lat", "lon"])
        if len(c):
            sample = c.iloc[np.linspace(0, len(c)-1, min(100, len(c))).astype(int)]
            lat0 = float(np.nanmedian(sample.lat))
            cxy = haversine_project(sample.lat, sample.lon, lat0)
            scores = []
            for source, g in pool.groupby("ref_source_file", sort=False):
                r = g.dropna(subset=["lat", "lon"])
                if r.empty:
                    continue
                tree = cKDTree(haversine_project(r.lat, r.lon, lat0))
                dist, _ = tree.query(cxy, k=1)
                scores.append((float(np.nanmedian(dist)), source))
            if scores:
                _, best = min(scores, key=lambda x: x[0])
                return pool[pool.ref_source_file == best].reset_index(drop=True)
    if "distance" in car and "distance" in pool and car.distance.notna().any():
        cmin, cmax = float(car.distance.min()), float(car.distance.max())
        overlaps = []
        for source, g in pool.groupby("ref_source_file", sort=False):
            if not g.distance.notna().any():
                continue
            rmin, rmax = float(g.distance.min()), float(g.distance.max())
            overlap = max(0.0, min(cmax, rmax) - max(cmin, rmin))
            overlaps.append((overlap, source))
        if overlaps:
            _, best = max(overlaps, key=lambda x: x[0])
            return pool[pool.ref_source_file == best].reset_index(drop=True)
    # If no defensible discriminator exists, retain the pool but disable no
    # safeguards here; the caller still applies distance/heading constraints.
    return pool.reset_index(drop=True)


def assign_spatial_splits(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Assign contiguous train/calibration/validation/test blocks per trip.

    Optional purge rows are marked ``purged`` and are never used by the model.
    Splitting happens before feature interpolation/resampling.
    """
    z = df.copy()
    split_cfg = cfg.get("split", {})
    lira_cfg = cfg.get("lira", {})
    train = float(split_cfg.get("train", 0.60))
    cal = train + float(split_cfg.get("calibration", 0.10))
    val = cal + float(split_cfg.get("validation", 0.10))
    guard = max(0, int(lira_cfg.get("split_guard_samples", 0)))
    z["split"] = "purged"
    z["split_position"] = np.nan
    group_col = "trip_id" if "trip_id" in z else None
    groups = z.groupby(group_col, sort=False, dropna=False) if group_col else [("all", z)]
    for _, g in groups:
        idx = g.index.to_numpy()
        if "route_s_m" in g and g.route_s_m.notna().sum() >= 2:
            coord = g.route_s_m.to_numpy(float)
            order = np.argsort(np.nan_to_num(coord, nan=np.inf), kind="stable")
        elif "distance" in g and g.distance.notna().sum() >= 2:
            coord = g.distance.to_numpy(float)
            order = np.argsort(np.nan_to_num(coord, nan=np.inf), kind="stable")
        else:
            order = np.arange(len(g))
        ordered_idx = idx[order]
        n = len(ordered_idx)
        if n == 0:
            continue
        pos = np.arange(n, dtype=float) / max(n, 1)
        labels = np.where(pos < train, "train", np.where(pos < cal, "calibration", np.where(pos < val, "validation", "test"))).astype(object)
        if guard > 0 and n > 4 * guard:
            for boundary in (int(round(train * n)), int(round(cal * n)), int(round(val * n))):
                lo = max(0, boundary - guard)
                hi = min(n, boundary + guard)
                labels[lo:hi] = "purged"
        z.loc[ordered_idx, "split"] = labels
        z.loc[ordered_idx, "split_position"] = pos
    return z


def impute_features_by_partition(df: pd.DataFrame, features: list[str], limit: int | None = 5) -> pd.DataFrame:
    """Impute features strictly within (trip, split) groups; labels are untouched."""
    pieces = []
    group_cols = [c for c in ("trip_id", "split") if c in df]
    groups = df.groupby(group_cols, sort=False, dropna=False) if group_cols else [("all", df)]
    for _, g in groups:
        g = g.copy()
        if str(g["split"].iloc[0]) == "purged" if "split" in g else False:
            pieces.append(g)
            continue
        vals = g[features].replace([np.inf, -np.inf], np.nan)
        vals = vals.interpolate(limit=limit, limit_direction="both").ffill().bfill()
        g[features] = vals
        pieces.append(g)
    return pd.concat(pieces, axis=0).sort_index() if pieces else df.copy()


def _resample_partition(g: pd.DataFrame, features: list[str], hz: float, max_gap_s: float) -> pd.DataFrame:
    if hz <= 0 or "time" not in g or g.time.notna().sum() < 4:
        return g.copy()
    x = g.sort_values("time").drop_duplicates(subset=["time"]).copy()
    t = x.time.to_numpy(float)
    finite = np.isfinite(t)
    x = x.loc[finite].copy()
    t = t[finite]
    if len(t) < 4 or t[-1] <= t[0]:
        return g.copy()
    dt = np.diff(t)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if dt.size == 0:
        return g.copy()
    implied_hz = 1.0 / float(np.median(dt))
    if not (0.5 <= implied_hz <= 1000.0):
        return g.copy()
    step = 1.0 / float(hz)
    grid = np.arange(t[0], t[-1] + step * 0.25, step)
    if len(grid) < 2:
        return g.copy()
    # Nearest source sample determines labels/metadata and gap validity.
    right = np.searchsorted(t, grid, side="left")
    right = np.clip(right, 0, len(t) - 1)
    left = np.clip(right - 1, 0, len(t) - 1)
    choose_right = np.abs(t[right] - grid) < np.abs(t[left] - grid)
    nearest = np.where(choose_right, right, left)
    gap = np.abs(t[nearest] - grid)
    keep = gap <= float(max_gap_s)
    grid = grid[keep]
    nearest = nearest[keep]
    if len(grid) < 2:
        return g.copy()
    out = x.iloc[nearest].reset_index(drop=True).copy()
    out["time"] = grid
    for col in features:
        vals = x[col].to_numpy(float)
        good = np.isfinite(vals) & np.isfinite(t)
        if good.sum() >= 2:
            out[col] = np.interp(grid, t[good], vals[good])
        elif good.sum() == 1:
            out[col] = vals[good][0]
    out["resampled"] = True
    return out


def resample_by_partition(df: pd.DataFrame, features: list[str], hz: float, max_gap_s: float) -> pd.DataFrame:
    pieces = []
    group_cols = [c for c in ("trip_id", "split") if c in df]
    groups = df.groupby(group_cols, sort=False, dropna=False) if group_cols else [("all", df)]
    for _, g in groups:
        if "split" in g and str(g["split"].iloc[0]) == "purged":
            pieces.append(g.copy())
        else:
            pieces.append(_resample_partition(g, features, hz, max_gap_s))
    return pd.concat(pieces, ignore_index=True) if pieces else df.copy()


def prepare_lira(raw: str | Path, out: str | Path, cfg: dict) -> Path:
    raw = Path(raw)
    out = Path(out)
    ensure_dir(out)
    lira_cfg = cfg.get("lira", {})
    max_m = float(lira_cfg.get("match_max_m", 10.0))
    heading_tol = lira_cfg.get("heading_tolerance_deg", 45.0)
    heading_tol = None if heading_tol is None else float(heading_tol)
    k_candidates = int(lira_cfg.get("k_candidates", 8))
    enforce_monotonic = bool(lira_cfg.get("enforce_monotonic", True))
    resample_hz = float(lira_cfg.get("resample_hz", 20.0))
    resample_gap = float(lira_cfg.get("resample_max_gap_s", 0.25))
    interp_limit = lira_cfg.get("interpolation_limit", 5)
    interp_limit = None if interp_limit is None else int(interp_limit)

    fric = list(raw.rglob("*fric_custom*.csv")) or [p for p in raw.rglob("*.csv") if "fric" in p.name.lower()]
    car = list(raw.rglob("task_7505*.txt")) or list(raw.rglob("*.txt"))
    if not fric or not car:
        raise FileNotFoundError(
            f"Expected LiRA *fric_custom*.csv and task_7505*.txt under {raw}; "
            f"found {len(fric)} friction, {len(car)} car files"
        )

    refs = []
    for p in fric:
        r = canonical_friction(read_table(p))
        ctx = parse_lira_context(p, raw)
        for k, v in ctx.items():
            r[k] = v if k != "trip_id" else f"ref_{v}"
        r["ref_source_file"] = str(p.relative_to(raw))
        if all(c in r for c in ("lat", "lon")):
            r["ref_route_s_m"] = cumulative_route_distance(r.lat, r.lon)
        refs.append(r)
    ref = pd.concat(refs, ignore_index=True)

    aligned = []
    alignment_rows = []
    for p in car:
        x = canonical_vehicle(read_table(p))
        ctx = parse_lira_context(p, raw)
        for k, v in ctx.items():
            x[k] = v
        x["source_file"] = str(p.relative_to(raw))
        if all(c in x for c in ("lat", "lon")):
            x["route_s_m"] = cumulative_route_distance(x.lat, x.lon)
        pool = _select_reference_pool(ref, ctx["route_id"], ctx["direction"])
        pool = _select_reference_trace_by_geometry(x, pool)
        use_monotonic = enforce_monotonic and len(pool) > 1
        z = spatial_align(
            x,
            pool,
            max_m=max_m,
            heading_tolerance_deg=heading_tol,
            enforce_monotonic=use_monotonic,
            k_candidates=k_candidates,
        )
        alignment_rows.append(
            {
                "trip_id": ctx["trip_id"],
                "route_id": ctx["route_id"],
                "direction": ctx["direction"],
                "vehicle_rows": int(len(x)),
                "candidate_reference_rows": int(len(pool)),
                "reference_trace": str(pool.ref_source_file.iloc[0]) if len(pool) and "ref_source_file" in pool else "unknown",
                "matched_rows": int(len(z)),
                "retention": float(len(z) / max(len(x), 1)),
            }
        )
        if not z.empty:
            # Keep the vehicle trip metadata authoritative after reference copy.
            for k, v in ctx.items():
                z[k] = v
            z["source_file"] = str(p.relative_to(raw))
            aligned.append(z)
    pd.DataFrame(alignment_rows).to_csv(out / "lira_alignment_report.csv", index=False)
    if not aligned:
        raise RuntimeError(
            "No LiRA rows survived route-aware GPS/distance alignment. Inspect lira_alignment_report.csv "
            "and relax only documented matching tolerances if necessary."
        )
    df = pd.concat(aligned, ignore_index=True)
    if "mu_ref" not in df:
        raise RuntimeError("Could not identify LiRA VIAFRIK mu columns. Inspect raw schema and canonical aliases.")

    # Speed/acceleration unit conversion is source-column-name based in
    # canonical_vehicle; no magnitude-only conversion is applied here.
    for needed in ("speed", "ax", "ay"):
        if needed not in df:
            df[needed] = 0.0

    feats = [c for c in CANONICAL if c in df and c not in ("time", "distance", "lat", "lon")]
    if len(feats) < 3:
        raise RuntimeError(f"Only {feats} production features resolved. Extend CANONICAL aliases for this LiRA revision.")

    # Split first. All interpolation/resampling is performed only inside a single
    # (trip_id, split) block, preventing adjacent validation/test values from
    # filling training samples and preventing any operation across trips.
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["mu_ref"]).reset_index(drop=True)
    df = assign_spatial_splits(df, cfg)
    df = impute_features_by_partition(df, feats, limit=interp_limit)
    df = df.dropna(subset=feats).reset_index(drop=True)
    if resample_hz > 0:
        df = resample_by_partition(df, feats, resample_hz, resample_gap)
        df = impute_features_by_partition(df, feats, limit=interp_limit)
        df = df.dropna(subset=feats).reset_index(drop=True)

    # Recompute the physics lower bound after all signal preprocessing.
    v = cfg["vehicle"]
    df["physics_lower_raw"] = vehicle_level_lower_bound(
        df.ax,
        df.ay,
        df.speed,
        mass=v["mass_kg"],
        g=v["gravity"],
        crr=v["crr"],
        rho_air=v["rho_air"],
        cdA=v["cdA_m2"],
        accel_error=v["accel_error_ms2"],
        external_force_margin=v["external_force_margin_n"],
        vertical_force_margin=v["vertical_force_margin_n"],
    )

    # Stable sample IDs let the benchmark prove that methods with different
    # history lengths are evaluated on exactly the same endpoints.
    df["sample_uid"] = [f"{trip}:{split}:{i}" for i, (trip, split) in enumerate(zip(df.trip_id, df.split))]
    keep_meta = [
        "time", "distance", "lat", "lon", "route_s_m", "gps_heading_deg", "match_distance_m",
        "match_ref_index", "mu_ref", "physics_lower_raw", "source_file", "ref_source_file",
        "route_id", "direction", "trip_id", "split", "split_position", "sample_uid", "resampled",
    ]
    keep = feats + [c for c in keep_meta if c in df]
    df = df[keep].copy()
    split_counts = df["split"].value_counts(dropna=False).to_dict()
    report = {
        "alignment": {
            "match_max_m": max_m,
            "heading_tolerance_deg": heading_tol,
            "enforce_monotonic": enforce_monotonic,
            "k_candidates": k_candidates,
        },
        "preprocessing": {
            "split_before_imputation": True,
            "partition_local_imputation": True,
            "resample_hz": resample_hz,
            "resample_max_gap_s": resample_gap,
            "interpolation_limit": interp_limit,
        },
        "split_counts": {str(k): int(v) for k, v in split_counts.items()},
        "features": feats,
        "gps_used_as_model_feature": False,
    }
    (out / "lira_preprocessing_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    path = out / "lira_aligned.csv"
    df.to_csv(path, index=False)
    (out / "lira_features.json").write_text(json.dumps(feats, indent=2), encoding="utf-8")
    return path
def prepare_kuleuven(raw: str|Path, out: str|Path) -> Path:
    """Canonicalize KU Leuven files for optional wheel-force virtual sensing.

    Because the source exposes many maneuvers, we concatenate CSVs and resolve WFT
    columns by names. The generated file keeps only runs where at least one WFT force is found.
    """
    raw=Path(raw); out=Path(out); ensure_dir(out)
    rows=[]
    for p in raw.rglob("*.csv"):
        try: df=read_table(p)
        except Exception: continue
        n=_normmap(df)
        def find(tokens):
            for col,nm in n.items():
                if all(t in nm for t in tokens): return col
        # Generic Kistler/WFT aliases.
        targets={}
        for side in ("fl","fr"):
            for comp in ("fx","fy","fz"):
                pats=[(side,comp),("wft",side,comp),("roadyn",side,comp)]
                col=None
                for pat in pats:
                    col=find(pat)
                    if col: break
                if col: targets[f"{comp}_{side}"]=pd.to_numeric(df[col],errors="coerce")
        if not targets: continue
        x=canonical_vehicle(df)
        for k,v in targets.items(): x[k]=v
        x["run"]=p.stem
        rows.append(x)
    if not rows:
        raise RuntimeError("No KU Leuven wheel-force columns resolved. Raw files are present; update alias resolver using the included source readme/column names.")
    allx=pd.concat(rows,ignore_index=True).dropna(how="all")
    path=out/"kuleuven_wft.csv"; allx.to_csv(path,index=False); return path


def prepare_kit(raw: str|Path, out: str|Path) -> Path:
    raw=Path(raw); out=Path(out); ensure_dir(out)
    rows=[]
    for p in list(raw.rglob("*.csv"))+list(raw.rglob("*.txt")):
        try: df=read_table(p)
        except Exception: continue
        nm=_normmap(df)
        def choose(*alts):
            for alt in alts:
                toks=[normalize_name(t) for t in alt]
                for col,name in nm.items():
                    if all(t in name for t in toks): return col
        fx=choose(("fx",),("longitudinal","force")); fy=choose(("fy",),("lateral","force")); fz=choose(("fz",),("vertical","force"),("normal","force"))
        if not fz or (not fx and not fy): continue
        x=pd.DataFrame()
        x["fx"]=pd.to_numeric(df[fx],errors="coerce") if fx else 0.0
        x["fy"]=pd.to_numeric(df[fy],errors="coerce") if fy else 0.0
        x["fz"]=pd.to_numeric(df[fz],errors="coerce")
        for key, alts in {
            "slip_ratio":(("slip","ratio"),("sr",)), "slip_angle":(("slip","angle"),("sa",)),
            "speed":(("speed",),("velocity",)), "pressure":(("pressure",),), "camber":(("camber",),("inclination",))}.items():
            cc=choose(*alts)
            if cc: x[key]=pd.to_numeric(df[cc],errors="coerce")
        x["source_file"]=p.name
        rows.append(x)
    if not rows: raise RuntimeError("No KIT force tables resolved; inspect extracted archive and extend aliases.")
    z=pd.concat(rows,ignore_index=True).dropna(subset=["fz"])
    z=z[np.abs(z.fz)>10].copy(); z["mu_util"]=np.hypot(z.fx,z.fy)/np.abs(z.fz)
    path=out/"kit_force.csv"; z.to_csv(path,index=False); return path


def make_synthetic(out: str|Path, n=12000, seed=20260905) -> Path:
    rng=np.random.default_rng(seed); out=Path(out); ensure_dir(out)
    t=np.arange(n)/20.0
    block=max(50,n//12); mu=np.repeat(rng.uniform(.25,1.1,size=max(1,n//block+1)),block)[:n]
    speed=np.clip(18+5*np.sin(t/37)+rng.normal(0,.5,n),2,35)
    excitation=np.clip(rng.beta(1.2,4,size=n),0,1)
    theta=rng.uniform(-np.pi,np.pi,n)
    rho=mu*excitation
    ax=9.81*rho*np.cos(theta)+rng.normal(0,.08,n)
    ay=9.81*rho*np.sin(theta)+rng.normal(0,.08,n)
    steer=np.clip(ay/np.maximum(speed,2)**2*2.7,-.5,.5)+rng.normal(0,.01,n)
    yaw=ay/np.maximum(speed,2)+rng.normal(0,.01,n)
    base=speed/.31*60/(2*np.pi)
    wheels=np.column_stack([base+rng.normal(0,2,n) for _ in range(4)])
    torque=np.maximum(0,150*ax)+rng.normal(0,20,n)
    df=pd.DataFrame({"time":t,"speed":speed,"ax":ax,"ay":ay,"yaw_rate":yaw,"steer":steer,
      "wheel_fl":wheels[:,0],"wheel_fr":wheels[:,1],"wheel_rl":wheels[:,2],"wheel_rr":wheels[:,3],
      "torque":torque,"mu_ref":mu,"distance":np.cumsum(speed/20)})
    df["physics_lower_raw"]=np.clip(rho-rng.uniform(0,.03,n),0,1.3)
    q=np.linspace(0,1,n,endpoint=False); df["split"]=np.where(q<.6,"train",np.where(q<.7,"calibration",np.where(q<.8,"validation","test")))
    df["route_id"]="SYNTH"; df["direction"]="FWD"; df["trip_id"]="synthetic_trip_0"
    df["sample_uid"]=[f"synthetic_trip_0:{sp}:{i}" for i,sp in enumerate(df.split)]
    path=out/"synthetic.csv"; df.to_csv(path,index=False); return path


def _write_manifest(paths, out_path: Path, root: Path):
    rows=[]
    for p in paths:
        try: size=p.stat().st_size
        except OSError: size=None
        rows.append({"path":str(p.relative_to(root)),"suffix":p.suffix.lower(),"size_bytes":size})
    pd.DataFrame(rows).to_csv(out_path,index=False)
    return out_path


def prepare_deep_dynamics(raw: str|Path, out: str|Path) -> Path:
    """Collect open Deep Dynamics/IAC/BayesRace tables without inventing friction labels."""
    raw=Path(raw); out=Path(out); ensure_dir(out)
    csvs=list(raw.rglob("*.csv"))
    if not csvs: return _write_manifest(list(raw.rglob("*")),out/"deep_dynamics_manifest.csv",raw)
    rows=[]
    for p in csvs:
        try: df=read_table(p)
        except Exception: continue
        veh=canonical_vehicle(df)
        if len(veh.columns)<2: continue
        veh["source_file"]=str(p.relative_to(raw)); rows.append(veh)
    if not rows: return _write_manifest(csvs,out/"deep_dynamics_manifest.csv",raw)
    z=pd.concat(rows,ignore_index=True); path=out/"deep_dynamics_vehicle.csv"; z.to_csv(path,index=False); return path


def _load_numpy_pair(folder: Path):
    """Load comma2k19's t/value arrays; handles .npy and extension-less numpy files."""
    if not folder.exists(): return None
    def load_candidate(names):
        for n in names:
            p=folder/n
            if p.exists():
                try: return np.load(p,allow_pickle=False)
                except Exception:
                    try: return np.fromfile(p,dtype=np.float64)
                    except Exception: pass
        return None
    t=load_candidate(["t","t.npy","time","time.npy"])
    v=load_candidate(["value","value.npy","values","values.npy"])
    if t is None or v is None: return None
    return np.asarray(t),np.asarray(v)


def prepare_comma2k19(raw: str|Path, out: str|Path) -> Path:
    """Prepare the bundled 1-minute comma2k19 example when present.

    This auxiliary dataset has no friction ground truth and is therefore never
    included in the direct friction benchmark. It is suitable for SSL/domain tests.
    """
    raw=Path(raw); out=Path(out); ensure_dir(out)
    plogs=[p for p in raw.rglob("processed_log") if p.is_dir()]
    records=[]
    for pl in plogs:
        specs={
            "speed":["CAN/car_speed","car_speed"],
            "steer":["CAN/steering_angle","steering_angle"],
            "wheels":["CAN/wheel_speeds","wheel_speeds"],
            "acc":["IMU/acceleration","acceleration"],
            "gyro":["IMU/gyro","gyro"],
        }
        loaded={}
        for key,rels in specs.items():
            for rel in rels:
                pair=_load_numpy_pair(pl/rel)
                if pair is not None: loaded[key]=pair; break
        if "speed" not in loaded: continue
        t0,v0=loaded["speed"]; t0=np.asarray(t0).reshape(-1); v0=np.asarray(v0).reshape(-1)
        n=min(len(t0),len(v0)); t0=t0[:n]; v0=v0[:n]
        frame=pd.DataFrame({"time":t0,"speed":v0})
        for key,(tt,vv) in loaded.items():
            if key=="speed": continue
            tt=np.asarray(tt).reshape(-1); vv=np.asarray(vv)
            if len(tt)==0 or len(vv)==0: continue
            if vv.ndim==1:
                frame[key]=np.interp(t0,tt[:len(vv)],vv[:len(tt)])
            else:
                m=min(len(tt),len(vv)); tt=tt[:m]; vv=vv[:m]
                for j in range(vv.shape[1]): frame[f"{key}_{j}"]=np.interp(t0,tt,vv[:,j])
        frame["source_segment"]=str(pl.parent.relative_to(raw)); records.append(frame)
    if not records: return _write_manifest(list(raw.rglob("*")),out/"comma2k19_manifest.csv",raw)
    z=pd.concat(records,ignore_index=True); path=out/"comma2k19_example.csv"; z.to_csv(path,index=False); return path


def prepare_extreme_road(raw: str|Path, out: str|Path) -> Path:
    raw=Path(raw); out=Path(out); ensure_dir(out)
    exts={".jpg",".jpeg",".png",".bmp",".webp"}; imgs=[p for p in raw.rglob("*") if p.suffix.lower() in exts]
    rows=[]
    for p in imgs:
        rel=p.relative_to(raw); parts=[x.lower() for x in rel.parts]
        label=rel.parent.name
        rows.append({"path":str(rel),"label":label,"filename":p.name})
    path=out/"extreme_road_images.csv"; pd.DataFrame(rows).to_csv(path,index=False); return path


def prepare_bicycle_tire(raw: str|Path, out: str|Path) -> Path:
    """Create a transparent file manifest; YAML/MAT data remain in source form.

    We intentionally do not coerce bicycle test-rig measurements into passenger-car
    mu labels. This dataset is auxiliary mechanics evidence only.
    """
    raw=Path(raw); out=Path(out); ensure_dir(out)
    return _write_manifest([p for p in raw.rglob("*") if p.is_file()],out/"bicycle_tire_manifest.csv",raw)


def prepare_mendeley_friction(raw: str|Path, out: str|Path) -> Path:
    raw=Path(raw); out=Path(out); ensure_dir(out); tables=[]
    files=list(raw.rglob("*.xlsx"))+list(raw.rglob("*.xls"))+list(raw.rglob("*.csv"))
    for p in files:
        try:
            if p.suffix.lower() in (".xlsx",".xls"):
                book=pd.read_excel(p,sheet_name=None)
                for sheet,df in book.items():
                    df=df.copy(); df["source_sheet"]=sheet; df["source_file"]=p.name; tables.append(df)
            else:
                df=read_table(p); df["source_file"]=p.name; tables.append(df)
        except Exception: continue
    if not tables: raise RuntimeError("No readable Mendeley friction workbook/CSV found")
    z=pd.concat(tables,ignore_index=True,sort=False)
    # Preserve source columns but add canonical friction/speed/surface when discoverable.
    mu=find_col(z,[("friction","coefficient"),("coefficient","friction"),("mu",)])
    speed=find_col(z,[("vehicle","speed"),("speed",),("velocity",)])
    surface=find_col(z,[("road","surface"),("surface",),("pavement",)])
    if mu: z["mu_ref"]=pd.to_numeric(z[mu],errors="coerce")
    if speed: z["speed_canonical"]=pd.to_numeric(z[speed],errors="coerce")
    if surface: z["surface_canonical"]=z[surface].astype(str)
    path=out/"mendeley_friction.csv"; z.to_csv(path,index=False); return path


def prepare_dataset(name: str, raw: str|Path, out: str|Path, cfg: dict | None = None) -> Path:
    name=name.lower(); cfg=cfg or {}
    if name=="lira": return prepare_lira(raw,out,cfg)
    if name=="kit": return prepare_kit(raw,out)
    if name=="kuleuven": return prepare_kuleuven(raw,out)
    if name=="deep_dynamics": return prepare_deep_dynamics(raw,out)
    if name=="comma2k19": return prepare_comma2k19(raw,out)
    if name=="extreme_road": return prepare_extreme_road(raw,out)
    if name=="bicycle_tire": return prepare_bicycle_tire(raw,out)
    if name=="mendeley_friction": return prepare_mendeley_friction(raw,out)
    if name=="synthetic": return make_synthetic(out,seed=(cfg or {}).get("seed",20260905))
    raise ValueError(name)
