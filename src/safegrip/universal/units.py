from __future__ import annotations

"""Canonical physical-unit conversion.

Numerical conversion is performed before neural tokenization.  Metadata
embeddings are therefore semantic hints, not a substitute for dimensional
normalization.
"""

from dataclasses import dataclass
import numpy as np

from .ontology import canonical_name


@dataclass(frozen=True)
class UnitTransform:
    canonical_unit: str
    unit_class: str
    scale: float = 1.0
    offset: float = 0.0


# y_canonical = scale * x + offset
_TRANSFORMS: dict[tuple[str, str], UnitTransform] = {
    ("acceleration", "m/s^2"): UnitTransform("m/s^2", "acceleration"),
    ("acceleration", "m/s2"): UnitTransform("m/s^2", "acceleration"),
    ("acceleration", "g"): UnitTransform("m/s^2", "acceleration", 9.80665),
    ("velocity", "m/s"): UnitTransform("m/s", "velocity"),
    ("velocity", "km/h"): UnitTransform("m/s", "velocity", 1.0 / 3.6),
    ("velocity", "kmh"): UnitTransform("m/s", "velocity", 1.0 / 3.6),
    ("wheel_speed", "m/s"): UnitTransform("m/s", "velocity"),
    ("angular_rate", "rad/s"): UnitTransform("rad/s", "angular_rate"),
    ("angular_rate", "deg/s"): UnitTransform("rad/s", "angular_rate", np.pi / 180.0),
    ("force", "n"): UnitTransform("N", "force"),
    ("force", "kn"): UnitTransform("N", "force", 1000.0),
    ("torque", "n*m"): UnitTransform("N*m", "torque"),
    ("torque", "nm"): UnitTransform("N*m", "torque"),
    ("pressure", "pa"): UnitTransform("Pa", "pressure"),
    ("pressure", "kpa"): UnitTransform("Pa", "pressure", 1000.0),
    ("pressure", "bar"): UnitTransform("Pa", "pressure", 100000.0),
    ("strain", "strain"): UnitTransform("strain", "strain"),
    ("strain", "microstrain"): UnitTransform("strain", "strain", 1e-6),
    ("strain", "ue"): UnitTransform("strain", "strain", 1e-6),
    ("steering_angle", "rad"): UnitTransform("rad", "angle"),
    ("steering_angle", "deg"): UnitTransform("rad", "angle", np.pi / 180.0),
    ("slip_angle", "rad"): UnitTransform("rad", "angle"),
    ("slip_angle", "deg"): UnitTransform("rad", "angle", np.pi / 180.0),
    ("slip_ratio", "1"): UnitTransform("1", "dimensionless"),
    ("friction", "1"): UnitTransform("1", "dimensionless"),
    ("grip_margin", "1"): UnitTransform("1", "dimensionless"),
    ("utilization", "1"): UnitTransform("1", "dimensionless"),
    ("temperature", "c"): UnitTransform("degC", "temperature"),
    ("temperature", "degc"): UnitTransform("degC", "temperature"),
    ("temperature", "k"): UnitTransform("degC", "temperature", 1.0, -273.15),
}


def _norm_unit(unit: str | None) -> str:
    if unit is None:
        return ""
    return str(unit).strip().lower().replace("²", "^2").replace("·", "*").replace(" ", "")


def resolve_transform(quantity: str, unit: str | None) -> UnitTransform:
    q = canonical_name(quantity)
    u = _norm_unit(unit)
    key = (q, u)
    if key in _TRANSFORMS:
        return _TRANSFORMS[key]
    if u in ("", "1", "dimensionless"):
        return UnitTransform("1", "dimensionless")
    # Unknown units are deliberately not silently rescaled.
    return UnitTransform(str(unit), "unknown")


def canonicalize_values(values, quantity: str, unit: str | None):
    transform = resolve_transform(quantity, unit)
    arr = np.asarray(values, dtype=float)
    return transform.scale * arr + transform.offset, transform
