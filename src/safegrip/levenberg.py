from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class Levenberg2023STFTRegressor:
    """Common-input adaptation of Levenberg et al. (2023).

    The source method relies on high-rate transverse acceleration and high-
    frequency vibration content.  The common LiRA benchmark is resampled to
    20 Hz, so this implementation preserves the STFT/relative-dB/positive-linear
    structure while selecting the usable frequency bin on training data only.
    """

    scaler: object
    lateral_index: int
    sample_rate_hz: float
    frequency_hz: float
    mean_amplitude: float
    intercept: float
    slope: float

    def _raw_lateral(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        n, t, d = X.shape
        raw = self.scaler.inverse_transform(X.reshape(-1, d)).reshape(n, t, d)
        return np.asarray(raw[:, :, self.lateral_index], dtype=float)

    def feature_db(self, X: np.ndarray) -> np.ndarray:
        a = self._raw_lateral(X)
        a = a - np.mean(a, axis=1, keepdims=True)
        spec = np.abs(np.fft.rfft(a, axis=1)) / max(1, a.shape[1])
        freqs = np.fft.rfftfreq(a.shape[1], d=1.0 / self.sample_rate_hz)
        idx = int(np.argmin(np.abs(freqs - self.frequency_hz)))
        amp = np.maximum(spec[:, idx], 1e-12)
        return 20.0 * np.log10(amp / max(self.mean_amplitude, 1e-12))

    def predict(self, X: np.ndarray) -> np.ndarray:
        z = self.feature_db(X)
        return (self.intercept + self.slope * z).astype(np.float32)


def _lateral_feature_index(features: list[str]) -> int:
    exact = {str(x).lower(): i for i, x in enumerate(features)}
    for key in ("ay", "acc_y", "lateral_acceleration", "acceleration_y"):
        if key in exact:
            return exact[key]
    for i, name in enumerate(features):
        low = str(name).lower()
        if "lateral" in low and ("acc" in low or "acceler" in low):
            return i
    raise ValueError("Levenberg2023 STFT baseline requires a transverse/lateral acceleration channel (e.g. ay)")


def fit_levenberg2023_stft(bundle, cfg: dict) -> Levenberg2023STFTRegressor:
    idx = _lateral_feature_index(bundle.features)
    fs = float(cfg.get("lira", {}).get("resample_hz", 20.0))
    X = np.asarray(bundle.Xtr, dtype=np.float32)
    n, t, d = X.shape
    raw = bundle.scaler.inverse_transform(X.reshape(-1, d)).reshape(n, t, d)
    a = np.asarray(raw[:, :, idx], dtype=float)
    a = a - np.mean(a, axis=1, keepdims=True)
    spec = np.abs(np.fft.rfft(a, axis=1)) / max(1, t)
    freqs = np.fft.rfftfreq(t, d=1.0 / fs)

    # Source paper uses much higher frequencies. Under the locked 20-Hz common
    # benchmark we select one non-DC usable bin using training labels only.
    valid = np.where((freqs > 0.0) & (freqs <= fs / 2.0))[0]
    if len(valid) == 0:
        raise ValueError("No non-DC STFT frequency is available for Levenberg2023 baseline")

    y = np.asarray(bundle.ytr, dtype=float)
    best_idx = int(valid[0])
    best_score = -np.inf
    best_mean = 1.0
    best_z = None
    for j in valid:
        amp = np.maximum(spec[:, j], 1e-12)
        mean_amp = float(np.mean(amp))
        z = 20.0 * np.log10(amp / max(mean_amp, 1e-12))
        if np.std(z) < 1e-12 or np.std(y) < 1e-12:
            score = 0.0
        else:
            # Source mapping has a positive coefficient; choose the frequency
            # with strongest positive training-only association.
            score = float(np.corrcoef(z, y)[0, 1])
            if not np.isfinite(score):
                score = -np.inf
        if score > best_score:
            best_score, best_idx, best_mean, best_z = score, int(j), mean_amp, z

    z = np.asarray(best_z, dtype=float)
    var = float(np.var(z))
    slope = 0.0 if var < 1e-12 else float(np.cov(z, y, ddof=0)[0, 1] / var)
    slope = max(0.0, slope)  # preserve the paper's positive linear mapping
    intercept = float(np.mean(y) - slope * np.mean(z))
    return Levenberg2023STFTRegressor(
        scaler=bundle.scaler,
        lateral_index=idx,
        sample_rate_hz=fs,
        frequency_hz=float(freqs[best_idx]),
        mean_amplitude=float(best_mean),
        intercept=intercept,
        slope=slope,
    )
