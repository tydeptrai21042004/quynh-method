from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch

from .schema import SensorRecord
from .queries import PhysicalQuery, DATASET_QUERIES, queries_for_dataset
from .tokenizer import UniversalSensorTokenizer
from .batching import UniversalSensorBatch, pad_tokenized_records

# Backward-compatible union for low-level tests only.  Paper experiments should
# always use queries_for_dataset()/collate_paper_records() so supervision from
# unrelated tasks is never manufactured.
DEFAULT_RESEARCH_QUERIES: tuple[PhysicalQuery, ...] = tuple(
    dict.fromkeys(q for qs in DATASET_QUERIES.values() for q in qs)
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
    aliases = {
        "force_x": ("fx", "force_x"),
        "force_y": ("fy", "force_y"),
        "force_z": ("fz", "force_z"),
        "friction": ("mu", "mu_ref", "friction"),
        "slip_angle": ("alpha", "slip_angle"),
        "velocity_x": ("vx", "velocity_x"),
        "velocity_y": ("vy", "velocity_y"),
        "yaw_rate": ("omega", "yaw", "yaw_rate"),
        "displacement": ("displacement", "distance_error", "position_error"),
        "orientation": ("orientation", "orientation_error", "yaw_error"),
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


def collate_paper_records(
    records: list[SensorRecord] | tuple[SensorRecord, ...],
    tokenizer: UniversalSensorTokenizer,
    dataset: str,
) -> UniversalResearchBatch:
    """Collate one paper dataset using only targets that belong to its task."""
    key = str(dataset).strip().lower()
    for record in records:
        domain = str(record.source_domain or key).lower()
        if domain not in {key, "unknown"}:
            raise ValueError(f"record domain {domain!r} does not match requested dataset {key!r}")
    return collate_sensor_records(records, tokenizer, queries_for_dataset(key))
