from __future__ import annotations
from pathlib import Path
import yaml


def load_config(path: str | Path | None = None) -> dict:
    path = Path(path or Path(__file__).resolve().parents[2] / "configs" / "default.yaml")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
