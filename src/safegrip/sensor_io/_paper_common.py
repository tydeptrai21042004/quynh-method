from __future__ import annotations

from pathlib import Path
import re
import pandas as pd


def norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


def find_column(frame: pd.DataFrame, aliases: tuple[str, ...] | list[str]) -> str | None:
    mapping = {norm(c): c for c in frame.columns}
    for alias in aliases:
        key = norm(alias)
        if key in mapping:
            return mapping[key]
    # Conservative token containment fallback; all alias tokens must be present.
    for alias in aliases:
        toks = [t for t in norm(alias).split("_") if t]
        if not toks:
            continue
        for key, col in mapping.items():
            if all(t in key.split("_") for t in toks):
                return col
    return None


def iter_tables(root: str | Path):
    root = Path(root)
    paths = [root] if root.is_file() else sorted(p for p in root.rglob("*") if p.is_file())
    for path in paths:
        suffix = path.suffix.lower()
        try:
            if suffix == ".csv":
                yield path, "csv", pd.read_csv(path)
            elif suffix in {".txt", ".tsv"}:
                try:
                    frame = pd.read_csv(path, sep=None, engine="python")
                except Exception:
                    continue
                yield path, "text", frame
            elif suffix in {".xlsx", ".xls"}:
                for sheet, frame in pd.read_excel(path, sheet_name=None).items():
                    yield path, str(sheet), frame
        except Exception:
            continue
