"""Measurement noise model and measurement containers.

Noise model (all noise is zero-mean Gaussian and independent between channels):

    V_meas   = V_true * (1 + sigma_v * s * e1)      relative (multiplicative) noise
    I_meas   = I_true * (1 + sigma_i * s * e2)      relative (multiplicative) noise
    phi_meas = phi_true + sigma_phi * s * e3        absolute (additive) noise, radians

``e1, e2, e3`` are independent N(0, 1) draws and ``s >= 0`` is an optional
per-sample noise scale (used by the "noisy measurements" scenario).

Relative noise on V and I mimics gain/calibration-type errors of a probe, and
additive noise on the phase mimics a fixed timing/phase uncertainty. This is a
modelling choice, not a model of a specific instrument.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]

MAX_RELATIVE_STD: float = 0.3
"""Upper bound for relative noise, so that 1 + sigma*e stays positive in practice."""

MAX_PHASE_STD_RAD: float = 0.5

NOISE_LEVELS: tuple[str, ...] = ("off", "low", "medium", "high")


@dataclass(frozen=True)
class NoiseConfig:
    """Standard deviations of the measurement noise."""

    v_rel_std: float = 0.01
    i_rel_std: float = 0.01
    phase_std_rad: float = float(np.deg2rad(0.5))

    def __post_init__(self) -> None:
        for name in ("v_rel_std", "i_rel_std"):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0 or value > MAX_RELATIVE_STD:
                raise ValueError(f"{name} must be in [0, {MAX_RELATIVE_STD}], got {value}")
        if (
            not np.isfinite(self.phase_std_rad)
            or self.phase_std_rad < 0
            or self.phase_std_rad > MAX_PHASE_STD_RAD
        ):
            raise ValueError(f"phase_std_rad must be in [0, {MAX_PHASE_STD_RAD}]")

    @classmethod
    def preset(cls, level: str) -> "NoiseConfig":
        """Return a named noise level: 'off', 'low', 'medium' or 'high'."""
        presets = {
            "off": (0.0, 0.0, 0.0),
            "low": (0.01, 0.01, 0.5),
            "medium": (0.02, 0.02, 1.0),
            "high": (0.05, 0.05, 2.5),
        }
        if level not in presets:
            raise ValueError(f"unknown noise level {level!r}; choose from {NOISE_LEVELS}")
        v, i, phase_deg = presets[level]
        return cls(v_rel_std=v, i_rel_std=i, phase_std_rad=float(np.deg2rad(phase_deg)))

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def add_measurement_noise(
    v_rms_v: ArrayLike,
    i_rms_a: ArrayLike,
    phase_rad: ArrayLike,
    noise: NoiseConfig,
    rng: np.random.Generator,
    scale: ArrayLike = 1.0,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Return noisy copies of V_rms, I_rms and phase.

    The three noise sequences are always drawn in the same order, so a given
    seed produces the same underlying N(0, 1) draws regardless of the noise
    level. That makes results at different noise levels directly comparable.
    """
    v = np.asarray(v_rms_v, dtype=float)
    i = np.asarray(i_rms_a, dtype=float)
    phi = np.asarray(phase_rad, dtype=float)
    shape = np.broadcast_shapes(v.shape, i.shape, phi.shape)
    s = np.asarray(scale, dtype=float)
    if not np.all(np.isfinite(s)) or np.any(s < 0):
        raise ValueError("noise scale must be finite and non-negative")
    s = np.broadcast_to(s, shape)

    eps = rng.standard_normal((3, *shape))
    v_meas = v * (1.0 + noise.v_rel_std * s * eps[0])
    i_meas = i * (1.0 + noise.i_rel_std * s * eps[1])
    phi_meas = phi + noise.phase_std_rad * s * eps[2]
    return v_meas, i_meas, phi_meas


@dataclass(frozen=True)
class MeasurementSeries:
    """Time series of measured quantities (all arrays share the same length)."""

    time_s: FloatArray
    v_rms_v: FloatArray
    i_rms_a: FloatArray
    phase_rad: FloatArray

    def __post_init__(self) -> None:
        n = np.shape(self.time_s)
        if not (np.shape(self.v_rms_v) == np.shape(self.i_rms_a) == np.shape(self.phase_rad) == n):
            raise ValueError("time, voltage, current and phase arrays must have equal shape")
