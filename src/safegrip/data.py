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


def _absolute_time_seconds(s: pd.Series) -> pd.Series:
    """Convert a LiRA timestamp to seconds without resetting each file to zero.

    ``task_7505_*.txt`` is distributed as separate asynchronous sensor files.
    Synchronising those streams requires preserving a common timestamp origin;
    using :func:`_time_seconds` on every file independently would erase the
    offsets between streams.  This helper mirrors the numeric/datetime unit
    detection used above but deliberately keeps the absolute/elapsed origin.
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
                scale = 1e3
            return pd.Series(x / scale, index=s.index, dtype=float)
    dt = pd.to_datetime(s, errors="coerce", utc=True)
    if dt.notna().sum() >= 3:
        x = dt.astype("int64").astype(float) / 1e9
        x[dt.isna().to_numpy()] = np.nan
        return pd.Series(x, index=s.index, dtype=float)
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


# LiRA platoon-test vehicle signals are published as separate
# ``task_<id>_<sensor>.txt`` streams.  The keys below map the official file
# suffixes to the canonical model features used by SafeGrip.  Longest/specific
# suffixes are matched before generic ones so ``rpm_fl`` is never mistaken for
# ``rpm`` and ``brk_trq_req_*`` is not mistaken for ``brk_trq``.
LIRA_TASK_FILE_FEATURES = {
    "acc_lon": "ax",
    "acc_trans": "ay",
    "acc_yaw": "yaw_rate",
    "strg_ang": "steer",
    "strg_pos": "steer",
    "rpm_fl": "wheel_fl",
    "rpm_fr": "wheel_fr",
    "rpm_rl": "wheel_rl",
    "rpm_rr": "wheel_rr",
    "whl_trq_est": "torque",
    "brk_trq": "brake_torque",
    "whl_prs_fl": "pressure_fl",
    "whl_prs_fr": "pressure_fr",
    "whl_prs_rl": "pressure_rl",
    "whl_prs_rr": "pressure_rr",
    "speed": "speed",
    "odo": "distance",
}


def _task_id_from_name(path: Path) -> str:
    m = re.search(r"task[_-]?(\d+)", path.stem, flags=re.IGNORECASE)
    return m.group(1) if m else normalize_name(path.stem)


def _lira_file_suffix(path: Path) -> str:
    stem = path.stem.lower()
    m = re.match(r"task[_-]?\d+[_-]?(.*)", stem)
    return (m.group(1) if m else stem).strip("_-")


def _find_time_column(df: pd.DataFrame) -> str | None:
    c = find_col(df, CANONICAL["time"])
    if c is not None:
        return c
    # Some exported task files use a terse first column without an informative
    # header.  Accept it only when it is predominantly numeric/datetime-like and
    # monotone, which avoids silently treating a sensor value as time.
    for c in df.columns[:2]:
        s = _absolute_time_seconds(df[c])
        v = s.to_numpy(float)
        finite = v[np.isfinite(v)]
        if finite.size >= 3 and np.mean(np.diff(finite) >= 0) > 0.98 and np.nanmax(finite) > np.nanmin(finite):
            return c
    return None


def _numeric_candidates(df: pd.DataFrame, exclude=()) -> list[str]:
    out = []
    excluded = set(exclude)
    for c in df.columns:
        if c in excluded:
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().sum() >= max(3, len(df) // 5):
            out.append(c)
    return out


def _extract_lira_gps_stream(path: Path) -> pd.DataFrame:
    """Parse ``task_*_gps_raw.txt`` into ``time, lat, lon``.

    Named latitude/longitude columns are preferred.  A guarded numeric fallback
    supports the published flat-file export when generic column names are used.
    """
    df = read_table(path)
    time_col = _find_time_column(df)
    if time_col is None:
        raise RuntimeError(f"Could not identify timestamp column in LiRA GPS file: {path.name}")
    lat_col = find_col(df, CANONICAL["lat"])
    lon_col = find_col(df, CANONICAL["lon"])
    candidates = _numeric_candidates(df, exclude=(time_col,))

    if lat_col is None or lon_col is None:
        # Prefer columns whose values fall in valid latitude/longitude ranges.
        stats = []
        for c in candidates:
            s = pd.to_numeric(df[c], errors="coerce")
            med = float(np.nanmedian(s)) if s.notna().any() else np.nan
            stats.append((c, med))
        if lat_col is None:
            lat_like = [c for c, med in stats if np.isfinite(med) and -90 <= med <= 90]
            # Denmark is around 55--57 N; prefer that range but keep the parser
            # geographically generic for mirrored copies of the dataset.
            preferred = [c for c in lat_like if 45 <= abs(float(np.nanmedian(pd.to_numeric(df[c], errors="coerce")))) <= 70]
            lat_col = (preferred or lat_like or [None])[0]
        if lon_col is None:
            lon_like = [c for c, med in stats if c != lat_col and np.isfinite(med) and -180 <= med <= 180]
            lon_col = (lon_like or [None])[0]

    if lat_col is not None and lon_col is not None:
        lat = pd.to_numeric(df[lat_col], errors="coerce")
        lon = pd.to_numeric(df[lon_col], errors="coerce")
    else:
        # Last-resort support for flat exports that serialize GPS as a pair in
        # one field (e.g. "[12.34, 55.67]").  We still validate geographic
        # ranges and identify which component is latitude from the data rather
        # than assuming an undocumented order.
        pair = None
        for c in df.columns:
            if c == time_col:
                continue
            vals = []
            for v in df[c].astype(str):
                nums = re.findall(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", v)
                vals.append((float(nums[0]), float(nums[1])) if len(nums) >= 2 else (np.nan, np.nan))
            arr = np.asarray(vals, dtype=float)
            if np.isfinite(arr).all(axis=1).sum() >= 3:
                pair = arr
                break
        if pair is None:
            raise RuntimeError(
                f"Could not identify latitude/longitude columns in LiRA GPS file {path.name}; columns={list(df.columns)}"
            )
        a, b = pair[:, 0], pair[:, 1]
        med_a, med_b = float(np.nanmedian(a)), float(np.nanmedian(b))
        # Prefer the component in a plausible latitude range around Denmark;
        # otherwise use the component with the larger absolute median when both
        # are valid latitude candidates (12E vs 55N in this dataset).
        if 45 <= abs(med_a) <= 70 and abs(med_b) <= 180:
            lat, lon = pd.Series(a), pd.Series(b)
        elif 45 <= abs(med_b) <= 70 and abs(med_a) <= 180:
            lat, lon = pd.Series(b), pd.Series(a)
        elif abs(med_a) > abs(med_b) and abs(med_a) <= 90 and abs(med_b) <= 180:
            lat, lon = pd.Series(a), pd.Series(b)
        else:
            lat, lon = pd.Series(b), pd.Series(a)

    out = pd.DataFrame({
        "time": _absolute_time_seconds(df[time_col]),
        "lat": lat,
        "lon": lon,
    }).dropna(subset=["time", "lat", "lon"])
    out = out[(out.lat.between(-90, 90)) & (out.lon.between(-180, 180))]
    return out.sort_values("time").drop_duplicates("time").reset_index(drop=True)


def _extract_lira_scalar_stream(path: Path, feature: str) -> pd.DataFrame:
    """Parse one LiRA task sensor file into a timestamped canonical signal."""
    df = read_table(path)
    time_col = _find_time_column(df)
    if time_col is None:
        raise RuntimeError(f"Could not identify timestamp column in LiRA sensor file: {path.name}")

    # First prefer the normal semantic alias resolver.  The task file name is a
    # second source of truth because many official exports name the value column
    # generically (e.g. ``value``).
    canonical = canonical_vehicle(df)
    used_filename_fallback = not (feature in canonical and len(canonical[feature]) == len(df))
    if not used_filename_fallback:
        value = canonical[feature]
    else:
        candidates = _numeric_candidates(df, exclude=(time_col,))
        if not candidates:
            raise RuntimeError(f"No numeric value column found in LiRA sensor file: {path.name}")
        # Prefer non-index/id columns and the column with the largest finite count.
        candidates = sorted(
            candidates,
            key=lambda c: (
                any(tok in normalize_name(c) for tok in ("index", "id", "count")),
                -pd.to_numeric(df[c], errors="coerce").notna().sum(),
            ),
        )
        value = pd.to_numeric(df[candidates[0]], errors="coerce")

    # Units documented for the public task_7505 export.  ``canonical_vehicle``
    # already converts explicitly unit-labelled km/h columns, so only convert
    # again when the header did not itself advertise km/h.  The official odometer
    # is kilometres while canonical ``distance`` is metres.
    cols_text = " ".join(str(c).lower().replace(" ", "") for c in df.columns)
    if feature == "speed" and not any(tok in cols_text for tok in ("km/h", "kmh", "kph", "kmph")):
        value = value / 3.6
    elif feature == "distance":
        value = value * 1000.0

    out = pd.DataFrame({"time": _absolute_time_seconds(df[time_col]), feature: value})
    out = out.dropna(subset=["time", feature]).sort_values("time").drop_duplicates("time")
    return out.reset_index(drop=True)


def _relative_if_needed(
    streams: dict[str, pd.DataFrame],
    gps: pd.DataFrame,
    required: list[str] | None = None,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, bool]:
    """Fall back to per-stream elapsed time only when timestamp origins differ.

    The official files normally share a common timestamp domain.  Some mirrors
    have been observed to rewrite individual files to elapsed time.  If the GPS
    and essential streams have no temporal overlap at all, normalising every
    stream to its first valid sample is safer than producing an empty merge and
    is explicitly recorded in the preprocessing report.
    """
    ranges = []
    checked = [streams[k] for k in (required or list(streams)) if k in streams]
    for z in [gps, *checked]:
        if len(z) and z.time.notna().any():
            ranges.append((float(z.time.min()), float(z.time.max())))
    if len(ranges) < 2:
        return streams, gps, False
    overlap_lo = max(a for a, _ in ranges)
    overlap_hi = min(b for _, b in ranges)
    if overlap_hi > overlap_lo:
        return streams, gps, False

    def rel(z):
        z = z.copy()
        if len(z):
            z["time"] = z.time - float(z.time.min())
        return z

    return {k: rel(v) for k, v in streams.items()}, rel(gps), True


def _interp_stream_to_grid(stream: pd.DataFrame, col: str, grid: np.ndarray, max_gap_s: float | None = None) -> np.ndarray:
    t = stream.time.to_numpy(float)
    v = stream[col].to_numpy(float)
    good = np.isfinite(t) & np.isfinite(v)
    t, v = t[good], v[good]
    if len(t) < 2:
        return np.full(len(grid), np.nan)
    order = np.argsort(t, kind="stable")
    t, v = t[order], v[order]
    unique = np.r_[True, np.diff(t) > 0]
    t, v = t[unique], v[unique]
    y = np.interp(grid, t, v, left=np.nan, right=np.nan)
    if max_gap_s is not None and np.isfinite(max_gap_s):
        right = np.searchsorted(t, grid, side="left")
        right = np.clip(right, 0, len(t) - 1)
        left = np.clip(right - 1, 0, len(t) - 1)
        gap = np.minimum(np.abs(grid - t[left]), np.abs(grid - t[right]))
        y[gap > float(max_gap_s)] = np.nan
    return y


def assemble_lira_task_streams(files: list[Path], cfg: dict) -> tuple[pd.DataFrame, dict]:
    """Synchronise the separate official ``task_7505_*`` files into one trip.

    GPS is interpolated onto a common vehicle timeline *before* VIAFRIK
    alignment, matching the preprocessing recommended by the LiRA-CD authors.
    Model features remain production-sensor signals; GPS is used only for
    reference alignment and is excluded later from the learned input set.
    """
    files = sorted(Path(p) for p in files)
    gps_files = [p for p in files if "gps" in _lira_file_suffix(p)]
    if not gps_files:
        raise RuntimeError(f"LiRA task group has no GPS stream: {[p.name for p in files]}")
    gps = _extract_lira_gps_stream(gps_files[0])
    if len(gps) < 2:
        raise RuntimeError(f"LiRA GPS stream contains too few valid rows: {gps_files[0].name}")

    streams: dict[str, pd.DataFrame] = {}
    source_files: dict[str, str] = {"gps": gps_files[0].name}
    # Give steering-angle priority over steering-position if both are available.
    feature_priority = {"strg_ang": 0, "strg_pos": 1}
    selected_suffix: dict[str, tuple[int, str]] = {}
    for p in files:
        suffix = _lira_file_suffix(p)
        if "gps" in suffix:
            continue
        feature = LIRA_TASK_FILE_FEATURES.get(suffix)
        if feature is None:
            continue
        pri = feature_priority.get(suffix, 0)
        if feature in selected_suffix and selected_suffix[feature][0] <= pri:
            continue
        try:
            stream = _extract_lira_scalar_stream(p, feature)
        except RuntimeError:
            continue
        if len(stream) >= 2:
            streams[feature] = stream
            selected_suffix[feature] = (pri, suffix)
            source_files[feature] = p.name

    essential = [x for x in ("speed", "ax", "ay") if x in streams]
    if len(essential) < 3:
        raise RuntimeError(
            "LiRA task stream assembly could not resolve required speed/ax/ay files. "
            f"Resolved={sorted(streams)}; files={[p.name for p in files]}"
        )

    streams, gps, used_relative_fallback = _relative_if_needed(streams, gps, required=essential)
    lira_cfg = cfg.get("lira", {})
    hz = float(lira_cfg.get("resample_hz", 20.0))
    hz = hz if hz > 0 else 20.0
    sensor_gap = float(lira_cfg.get("sensor_merge_max_gap_s", max(0.5, 5.0 / hz)))
    gps_gap = float(lira_cfg.get("gps_interp_max_gap_s", 2.5))

    # Restrict the grid to GPS + required sensors. Optional streams may start or
    # stop later without discarding otherwise usable data.
    required_ranges = [(float(gps.time.min()), float(gps.time.max()))]
    for feature in essential:
        s = streams[feature]
        required_ranges.append((float(s.time.min()), float(s.time.max())))
    lo = max(a for a, _ in required_ranges)
    hi = min(b for _, b in required_ranges)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        raise RuntimeError(
            "LiRA task sensor/GPS timestamps do not overlap after parsing. "
            "Inspect the raw timestamp columns; the parser did not fabricate row-index alignment."
        )
    step = 1.0 / hz
    grid = np.arange(lo, hi + step * 0.25, step)
    if len(grid) < 4:
        raise RuntimeError("LiRA common task timeline contains fewer than four samples")

    out = pd.DataFrame({"time": grid})
    out["lat"] = _interp_stream_to_grid(gps, "lat", grid, max_gap_s=gps_gap)
    out["lon"] = _interp_stream_to_grid(gps, "lon", grid, max_gap_s=gps_gap)
    for feature, stream in streams.items():
        out[feature] = _interp_stream_to_grid(stream, feature, grid, max_gap_s=sensor_gap)

    out = out.dropna(subset=["lat", "lon", *essential]).reset_index(drop=True)
    task_id = _task_id_from_name(files[0])
    out["trip_id"] = f"task_{task_id}"
    out["source_file"] = f"task_{task_id}_assembled"
    out["route_id"] = "unknown"
    out["direction"] = "unknown"
    out["route_s_m"] = cumulative_route_distance(out.lat, out.lon)
    report = {
        "task_id": task_id,
        "input_files": [p.name for p in files],
        "resolved_streams": source_files,
        "rows_after_sync": int(len(out)),
        "resample_hz": hz,
        "timestamp_relative_fallback": bool(used_relative_fallback),
        "sensor_merge_max_gap_s": sensor_gap,
        "gps_interp_max_gap_s": gps_gap,
    }
    return out, report


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


def _align_to_reference_traces(
    car: pd.DataFrame,
    pool: pd.DataFrame,
    *,
    max_m: float,
    heading_tolerance_deg: float | None,
    enforce_monotonic: bool,
    k_candidates: int,
) -> pd.DataFrame:
    """Align one assembled vehicle task against every candidate VIAFRIK trace.

    The public task-7505 campaign can contain samples from both motorway
    directions while the reference is supplied as separate ``*_hh``/``*_vh``
    files.  Selecting a single reference file for the whole task can therefore
    discard half the drive.  We align each trace independently, keep the nearest
    valid match per vehicle timestamp, and never combine reference rows before
    the route/heading/monotonic safeguards are applied.
    """
    if car.empty or pool.empty:
        return pd.DataFrame()
    c = car.copy().reset_index(drop=True)
    c["_vehicle_row_id"] = np.arange(len(c), dtype=int)
    if "ref_source_file" in pool:
        traces = list(pool.groupby("ref_source_file", sort=False))
    else:
        traces = [("reference", pool)]
    matches = []
    for source, trace in traces:
        trace = trace.reset_index(drop=True)
        orientations = [("forward", trace)]
        if enforce_monotonic and len(trace) > 1:
            orientations.append(("reverse", trace.iloc[::-1].reset_index(drop=True)))
        best = None
        best_score = None
        for orientation, rr in orientations:
            z = spatial_align(
                c,
                rr,
                max_m=max_m,
                heading_tolerance_deg=heading_tolerance_deg,
                enforce_monotonic=enforce_monotonic,
                k_candidates=k_candidates,
            )
            if z.empty:
                continue
            med = float(z.match_distance_m.median()) if "match_distance_m" in z else np.inf
            score = (len(z), -med)
            if best_score is None or score > best_score:
                best_score = score
                best = z
                best["reference_orientation"] = orientation
        if best is not None and not best.empty:
            best["reference_trace"] = str(source)
            matches.append(best)
    if not matches:
        return pd.DataFrame()
    z = pd.concat(matches, ignore_index=True)
    if "match_distance_m" in z:
        z = z.sort_values(["_vehicle_row_id", "match_distance_m"], kind="stable")
    z = z.drop_duplicates("_vehicle_row_id", keep="first")
    sort_cols = [c for c in ("time", "_vehicle_row_id") if c in z]
    if sort_cols:
        z = z.sort_values(sort_cols, kind="stable")
    return z.drop(columns=["_vehicle_row_id"], errors="ignore").reset_index(drop=True)


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

    # The official platoon-test download stores each vehicle signal in a
    # separate task_<id>_<sensor>.txt file. Assemble/synchronise all streams for
    # a task first; treating each file as a complete trip is invalid because
    # only the GPS stream contains coordinates.
    car_groups: dict[str, list[Path]] = {}
    for p in car:
        car_groups.setdefault(_task_id_from_name(p), []).append(p)

    aligned = []
    alignment_rows = []
    assembly_rows = []
    for task_id, task_files in sorted(car_groups.items()):
        x, assembly_report = assemble_lira_task_streams(task_files, cfg)
        contexts = [parse_lira_context(p, raw) for p in task_files]
        known_routes = [c["route_id"] for c in contexts if c["route_id"] != "unknown"]
        known_dirs = [c["direction"] for c in contexts if c["direction"] != "unknown"]
        ctx = {
            "route_id": known_routes[0] if known_routes and len(set(known_routes)) == 1 else "unknown",
            "direction": known_dirs[0] if known_dirs and len(set(known_dirs)) == 1 else "unknown",
            "trip_id": f"task_{task_id}",
        }
        for k, v in ctx.items():
            x[k] = v
        x["source_file"] = f"task_{task_id}_assembled"
        pool = _select_reference_pool(ref, ctx["route_id"], ctx["direction"])
        z = _align_to_reference_traces(
            x,
            pool,
            max_m=max_m,
            heading_tolerance_deg=heading_tol,
            enforce_monotonic=enforce_monotonic,
            k_candidates=k_candidates,
        )
        assembly_rows.append(assembly_report)
        traces = []
        if not z.empty and "reference_trace" in z:
            traces = sorted(str(v) for v in z.reference_trace.dropna().unique())
        alignment_rows.append(
            {
                "trip_id": ctx["trip_id"],
                "route_id": ctx["route_id"],
                "direction": ctx["direction"],
                "vehicle_rows": int(len(x)),
                "candidate_reference_rows": int(len(pool)),
                "reference_trace": ";".join(traces) if traces else "unknown",
                "matched_rows": int(len(z)),
                "retention": float(len(z) / max(len(x), 1)),
            }
        )
        if not z.empty:
            # Keep the vehicle trip metadata authoritative after reference copy.
            for k, v in ctx.items():
                z[k] = v
            z["source_file"] = f"task_{task_id}_assembled"
            aligned.append(z)
    (out / "lira_stream_assembly_report.json").write_text(json.dumps(assembly_rows, indent=2), encoding="utf-8")
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
