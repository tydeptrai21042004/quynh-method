from __future__ import annotations
import json, os, random, re, unicodedata
from pathlib import Path
import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def normalize_name(s: str) -> str:
    s = str(s).replace("µ", "mu").replace("μ", "mu")
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = s.lower()
    return re.sub(r"[^a-z0-9]+", "", s)


def json_dump(obj, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"
