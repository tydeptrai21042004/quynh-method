from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np

from .ontology import (
    QUANTITY_TO_ID,
    AXIS_TO_ID,
    LOCATION_TO_ID,
    UNIT_CLASS_TO_ID,
    ontology_id,
)
from .schema import SensorRecord, SensorChannel


@dataclass(frozen=True)
class TokenizerConfig:
    patch_duration_sec: float = 0.5
    waveform_points: int = 32
    use_statistics: bool = True
    use_spectrum: bool = True
    relative_spectral_bands: int = 4
    absolute_spectral_edges_hz: tuple[float, ...] = (0.0, 1.0, 5.0, 20.0, 100.0, float("inf"))


@dataclass(frozen=True)
class TokenizedRecord:
    features: np.ndarray
    quantity_ids: np.ndarray
    axis_ids: np.ndarray
    location_ids: np.ndarray
    unit_class_ids: np.ndarray
    sample_rates_hz: np.ndarray
    times_sec: np.ndarray
    channel_ids: np.ndarray
    sequence_id: str

    @property
    def num_tokens(self) -> int:
        return int(self.features.shape[0])


class UniversalSensorTokenizer:
    """Deterministic physical-time patch tokenizer.

    The extractor keeps neural learning out of the dataset adapters.  It creates
    fixed-size patch descriptors from arbitrary 1-D channels while retaining
    physical metadata for the trainable token encoder.
    """

    def __init__(self, config: TokenizerConfig | None = None):
        self.config = config or TokenizerConfig()
        if self.config.patch_duration_sec <= 0:
            raise ValueError("patch_duration_sec must be positive")
        if self.config.waveform_points < 2:
            raise ValueError("waveform_points must be >= 2")
        if self.config.relative_spectral_bands < 1:
            raise ValueError("relative_spectral_bands must be >= 1")

    @property
    def feature_dim(self) -> int:
        d = self.config.waveform_points
        if self.config.use_statistics:
            d += 7
        if self.config.use_spectrum:
            d += self.config.relative_spectral_bands
            d += len(self.config.absolute_spectral_edges_hz) - 1
        return d

    @staticmethod
    def _finite(values: np.ndarray, timestamps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mask = np.isfinite(values) & np.isfinite(timestamps)
        return values[mask], timestamps[mask]

    def _waveform(self, values: np.ndarray, timestamps: np.ndarray, start: float, end: float) -> np.ndarray:
        n = self.config.waveform_points
        if len(values) == 1:
            return np.full(n, float(values[0]), dtype=np.float32)
        denom = max(end - start, 1e-9)
        u = np.clip((timestamps - start) / denom, 0.0, 1.0)
        # np.interp requires increasing x. Duplicate timestamps are reduced by
        # keeping the last value at each location.
        uu, idx = np.unique(u, return_index=True)
        vv = values[idx]
        if len(uu) == 1:
            return np.full(n, float(vv[0]), dtype=np.float32)
        grid = np.linspace(0.0, 1.0, n)
        return np.interp(grid, uu, vv).astype(np.float32)

    @staticmethod
    def _statistics(values: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
        mean = float(np.mean(values))
        std = float(np.std(values))
        rms = float(np.sqrt(np.mean(np.square(values))))
        minv = float(np.min(values))
        maxv = float(np.max(values))
        delta = float(values[-1] - values[0])
        dt = float(timestamps[-1] - timestamps[0]) if len(values) > 1 else 0.0
        slope = delta / dt if dt > 1e-9 else 0.0
        return np.asarray([mean, std, rms, minv, maxv, delta, slope], dtype=np.float32)

    @staticmethod
    def _observed_rate(channel: SensorChannel, timestamps: np.ndarray) -> float:
        if channel.meta.sample_rate_hz is not None and channel.meta.sample_rate_hz > 0:
            return float(channel.meta.sample_rate_hz)
        if len(timestamps) > 1:
            dt = np.diff(timestamps)
            dt = dt[np.isfinite(dt) & (dt > 1e-9)]
            if len(dt):
                return float(1.0 / np.median(dt))
        return 1.0

    def _spectrum(self, values: np.ndarray, sample_rate_hz: float) -> np.ndarray:
        rel_n = self.config.relative_spectral_bands
        abs_edges = self.config.absolute_spectral_edges_hz
        out = np.zeros(rel_n + len(abs_edges) - 1, dtype=np.float32)
        if len(values) < 3 or sample_rate_hz <= 0:
            return out
        x = np.asarray(values, dtype=float) - float(np.mean(values))
        spec = np.fft.rfft(x)
        power = np.square(np.abs(spec))
        freqs = np.fft.rfftfreq(len(x), d=1.0 / sample_rate_hz)
        if len(power) <= 1:
            return out
        power[0] = 0.0
        total = float(np.sum(power))
        if total <= 1e-15:
            return out
        nyq = sample_rate_hz / 2.0
        rel_edges = np.linspace(0.0, 1.0, rel_n + 1)
        cursor = 0
        norm_f = freqs / max(nyq, 1e-12)
        for i in range(rel_n):
            lo, hi = rel_edges[i], rel_edges[i + 1]
            mask = (norm_f >= lo) & ((norm_f < hi) if i < rel_n - 1 else (norm_f <= hi))
            out[cursor] = float(np.sum(power[mask]) / total)
            cursor += 1
        for i in range(len(abs_edges) - 1):
            lo, hi = abs_edges[i], abs_edges[i + 1]
            if not math.isfinite(hi):
                mask = freqs >= lo
            else:
                mask = (freqs >= lo) & (freqs < hi)
            out[cursor] = float(np.sum(power[mask]) / total)
            cursor += 1
        return out

    def _patch_features(self, channel: SensorChannel, values: np.ndarray, timestamps: np.ndarray, start: float, end: float) -> np.ndarray:
        parts = [self._waveform(values, timestamps, start, end)]
        if self.config.use_statistics:
            parts.append(self._statistics(values, timestamps))
        if self.config.use_spectrum:
            parts.append(self._spectrum(values, self._observed_rate(channel, timestamps)))
        return np.concatenate(parts).astype(np.float32)

    def tokenize(self, record: SensorRecord) -> TokenizedRecord:
        t0, t1 = record.start_time, record.end_time
        width = self.config.patch_duration_sec
        features: list[np.ndarray] = []
        quantity_ids: list[int] = []
        axis_ids: list[int] = []
        location_ids: list[int] = []
        unit_class_ids: list[int] = []
        sample_rates: list[float] = []
        token_times: list[float] = []
        channel_ids: list[int] = []

        # Include the endpoint by adding a small epsilon; zero-duration records
        # still receive one patch.
        n_patches = max(1, int(math.ceil(max(t1 - t0, 0.0) / width)))
        for channel_idx, channel in enumerate(record.channels):
            values, times = self._finite(channel.values, channel.timestamps)
            if len(values) == 0:
                continue
            for j in range(n_patches):
                start = t0 + j * width
                end = start + width
                if j == n_patches - 1:
                    mask = (times >= start) & (times <= end + 1e-12)
                else:
                    mask = (times >= start) & (times < end)
                if not np.any(mask):
                    continue
                pv, pt = values[mask], times[mask]
                features.append(self._patch_features(channel, pv, pt, start, end))
                quantity_ids.append(ontology_id(channel.meta.quantity, QUANTITY_TO_ID))
                axis_ids.append(ontology_id(channel.meta.axis, AXIS_TO_ID))
                location_ids.append(ontology_id(channel.meta.location, LOCATION_TO_ID))
                unit_class_ids.append(ontology_id(channel.meta.unit_class, UNIT_CLASS_TO_ID))
                sample_rates.append(self._observed_rate(channel, pt))
                token_times.append((start + 0.5 * width) - t0)
                channel_ids.append(channel_idx)

        # Numeric physical context becomes a non-temporal token. Categorical
        # context is intentionally left to a future explicit categorical
        # ontology rather than silently ordinal-encoding strings.
        for ctx in record.context:
            if not isinstance(ctx.value, (int, float, np.number)):
                continue
            value = float(ctx.value)
            feat = np.zeros(self.feature_dim, dtype=np.float32)
            # Constant waveform + matching basic statistics; spectrum stays 0.
            feat[: self.config.waveform_points] = value
            cursor = self.config.waveform_points
            if self.config.use_statistics:
                feat[cursor : cursor + 7] = np.asarray([value, 0.0, abs(value), value, value, 0.0, 0.0], dtype=np.float32)
            features.append(feat)
            quantity_ids.append(ontology_id(ctx.meta.quantity, QUANTITY_TO_ID))
            axis_ids.append(ontology_id(ctx.meta.axis, AXIS_TO_ID))
            location_ids.append(ontology_id(ctx.meta.location, LOCATION_TO_ID))
            unit_class_ids.append(ontology_id(ctx.meta.unit_class, UNIT_CLASS_TO_ID))
            sample_rates.append(0.0)
            token_times.append(0.0)
            channel_ids.append(-2)  # static context, never treated as a sensor channel

        if not features:
            raise ValueError("record contains no finite samples to tokenize")
        return TokenizedRecord(
            features=np.stack(features).astype(np.float32),
            quantity_ids=np.asarray(quantity_ids, dtype=np.int64),
            axis_ids=np.asarray(axis_ids, dtype=np.int64),
            location_ids=np.asarray(location_ids, dtype=np.int64),
            unit_class_ids=np.asarray(unit_class_ids, dtype=np.int64),
            sample_rates_hz=np.asarray(sample_rates, dtype=np.float32),
            times_sec=np.asarray(token_times, dtype=np.float32),
            channel_ids=np.asarray(channel_ids, dtype=np.int64),
            sequence_id=record.sequence_id,
        )
