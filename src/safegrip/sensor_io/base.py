from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from safegrip.universal.schema import SensorRecord


class SensorRecordLoader(Protocol):
    def __len__(self) -> int: ...
    def __getitem__(self, index: int) -> SensorRecord: ...


@dataclass(frozen=True)
class ChannelSpec:
    column: str
    quantity: str
    unit: str
    axis: str = "scalar"
    location: str = "unknown"
    sample_rate_hz: float | None = None


@dataclass(frozen=True)
class TargetSpec:
    column: str
    name: str
