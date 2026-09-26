from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch

from .schema import SensorRecord
from .queries import (
    PhysicalQuery, FRICTION_QUERY, FORCE_X_QUERY, FORCE_Y_QUERY,
    FORCE_Z_QUERY, UTILIZATION_QUERY, GRIP_MARGIN_QUERY,
)
from .tokenizer import UniversalSensorTokenizer
from .batching import UniversalSensorBatch, pad_tokenized_records


DEFAULT_RESEARCH_QUERIES: tuple[PhysicalQuery, ...] = (
    FRICTION_QUERY,
    FORCE_X_QUERY,
    FORCE_Y_QUERY,
    FORCE_Z_QUERY,
    UTILIZATION_QUERY,
    GRIP_MARGIN_QUERY,
)


@dataclass
class UniversalResearchBatch:
    sensors: UniversalSensorBatch
    target_values: torch.Tensor
    target_mask: torch.Tensor
    queries: tuple[PhysicalQuery, ...]
    domains: tuple[str, ...]

    def to(self, device):
        return UniversalResearchBatch(
            self.sensors.to(device),
            self.target_values.to(device),
            self.target_mask.to(device),
            self.queries,
            self.domains,
        )


def _target_for_query(record: SensorRecord, query: PhysicalQuery):
    candidates = [query.name, query.quantity]
    # Common aliases used by the research benchmark.
    aliases = {
        "force_x": ("fx", "force_x"),
        "force_y": ("fy", "force_y"),
        "force_z": ("fz", "force_z"),
        "friction": ("mu", "mu_ref", "friction"),
        "utilization": ("u", "utilization"),
        "grip_margin": ("grip", "grip_margin"),
    }
    candidates.extend(aliases.get(query.name or query.quantity, ()))
    for name in candidates:
        if name and name in record.targets:
            tv = record.targets[name]
            if not tv.mask:
                return np.nan, False
            arr = np.asarray(tv.value, dtype=float)
            finite = arr[np.isfinite(arr)]
            if len(finite):
                return float(np.mean(finite)), True
    return np.nan, False


def collate_sensor_records(
    records: list[SensorRecord] | tuple[SensorRecord, ...],
    tokenizer: UniversalSensorTokenizer,
    queries: tuple[PhysicalQuery, ...] = DEFAULT_RESEARCH_QUERIES,
) -> UniversalResearchBatch:
    if not records:
        raise ValueError("records cannot be empty")
    tokenized = [tokenizer.tokenize(r) for r in records]
    sensor_batch = pad_tokenized_records(tokenized)
    values = torch.zeros((len(records), len(queries)), dtype=torch.float32)
    mask = torch.zeros((len(records), len(queries)), dtype=torch.bool)
    for i, record in enumerate(records):
        for j, query in enumerate(queries):
            value, ok = _target_for_query(record, query)
            if ok:
                values[i, j] = value
                mask[i, j] = True
    domains = tuple(str(r.source_domain or "unknown") for r in records)
    return UniversalResearchBatch(sensor_batch, values, mask, tuple(queries), domains)
