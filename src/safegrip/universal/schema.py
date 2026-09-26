from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any
import numpy as np

from .ontology import canonical_name
from .units import canonicalize_values, resolve_transform


@dataclass(frozen=True)
class SensorMeta:
    quantity: str
    unit: str
    axis: str | None = None
    location: str | None = None
    sample_rate_hz: float | None = None
    sensor_type: str | None = None
    confidence: float | None = None
    canonical_unit: str | None = None
    unit_class: str | None = None

    def canonicalized(self) -> "SensorMeta":
        tr = resolve_transform(self.quantity, self.unit)
        return replace(
            self,
            quantity=canonical_name(self.quantity),
            axis=canonical_name(self.axis),
            location=canonical_name(self.location),
            canonical_unit=tr.canonical_unit,
            unit_class=tr.unit_class,
        )


@dataclass(frozen=True)
class SensorChannel:
    values: np.ndarray
    timestamps: np.ndarray
    meta: SensorMeta

    def __post_init__(self):
        values = np.asarray(self.values, dtype=float).reshape(-1)
        timestamps = np.asarray(self.timestamps, dtype=float).reshape(-1)
        if len(values) != len(timestamps):
            raise ValueError("values and timestamps must have the same length")
        if len(values) == 0:
            raise ValueError("SensorChannel cannot be empty")
        if np.any(~np.isfinite(timestamps)):
            raise ValueError("timestamps must be finite")
        if np.any(np.diff(timestamps) < 0):
            raise ValueError("timestamps must be non-decreasing")
        meta = self.meta.canonicalized()
        canonical_values, _ = canonicalize_values(values, meta.quantity, meta.unit)
        object.__setattr__(self, "values", canonical_values)
        object.__setattr__(self, "timestamps", timestamps)
        object.__setattr__(self, "meta", meta)


@dataclass(frozen=True)
class ContextValue:
    value: float | int | str
    meta: SensorMeta

    def canonicalized(self) -> "ContextValue":
        meta = self.meta.canonicalized()
        if isinstance(self.value, (int, float, np.number)):
            val, _ = canonicalize_values([self.value], meta.quantity, meta.unit)
            return ContextValue(float(val[0]), meta)
        return ContextValue(self.value, meta)


@dataclass(frozen=True)
class TargetValue:
    name: str
    value: float | np.ndarray
    mask: bool = True
    query: Any | None = None


@dataclass(frozen=True)
class SensorRecord:
    channels: tuple[SensorChannel, ...] | list[SensorChannel]
    targets: dict[str, TargetValue] = field(default_factory=dict)
    sequence_id: str = ""
    context: tuple[ContextValue, ...] | list[ContextValue] = field(default_factory=tuple)
    source_domain: str | None = None

    def __post_init__(self):
        channels = tuple(self.channels)
        contexts = tuple(c.canonicalized() for c in self.context)
        if len(channels) == 0 and not any(isinstance(c.value, (int, float, np.number)) for c in contexts):
            raise ValueError("SensorRecord requires at least one sensor channel or numeric physical context")
        object.__setattr__(self, "channels", channels)
        object.__setattr__(self, "context", contexts)

    @property
    def start_time(self) -> float:
        return min((float(c.timestamps[0]) for c in self.channels), default=0.0)

    @property
    def end_time(self) -> float:
        return max((float(c.timestamps[-1]) for c in self.channels), default=0.0)
