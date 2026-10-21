"""Physical model of an equivalent series R-L-C RF load.

Conventions
-----------
* Steady-state sinusoidal (phasor) model at a single frequency ``f``.
* Equivalent load impedance ``Z = R + jX`` with ``X = wL - 1/(wC)``, ``w = 2*pi*f``.
* Phase ``phi = atan2(X, R)`` is the phase of the voltage *relative to the current*.
  ``phi > 0`` means inductive (voltage leads current), ``phi < 0`` capacitive.
* ``V_rms`` is the voltage at the measurement plane, across the load. The model
  assumes it is given; it does not model a generator with its own source impedance.

All functions accept scalars or NumPy arrays (with broadcasting) and validate inputs.
SI units are used everywhere: Hz, ohm, henry, farad, volt, ampere, radian.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]

DEFAULT_FREQUENCY_HZ: float = 13.56e6
"""Industrial ISM frequency commonly used for RF plasma processing."""

MIN_IMPEDANCE_OHM: float = 1e-9
"""Below this |Z| the voltage/current relation is treated as degenerate."""


def _finite(value: ArrayLike, name: str) -> FloatArray:
    arr = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite (no NaN or infinity)")
    return arr


def _positive(value: ArrayLike, name: str, *, allow_zero: bool = False) -> FloatArray:
    arr = _finite(value, name)
    if allow_zero:
        if np.any(arr < 0):
            raise ValueError(f"{name} must be non-negative")
    elif np.any(arr <= 0):
        raise ValueError(f"{name} must be strictly positive")
    return arr


def angular_frequency(frequency_hz: ArrayLike) -> FloatArray:
    """Return w = 2*pi*f in rad/s."""
    return 2.0 * np.pi * _positive(frequency_hz, "frequency_hz")


def reactance(
    frequency_hz: ArrayLike, inductance_h: ArrayLike, capacitance_f: ArrayLike
) -> FloatArray:
    """Return the series reactance X = wL - 1/(wC) in ohm."""
    w = angular_frequency(frequency_hz)
    inductance = _positive(inductance_h, "inductance_h", allow_zero=True)
    capacitance = _positive(capacitance_f, "capacitance_f")
    return w * inductance - 1.0 / (w * capacitance)


def rlc_impedance(
    frequency_hz: ArrayLike,
    resistance_ohm: ArrayLike,
    inductance_h: ArrayLike,
    capacitance_f: ArrayLike,
) -> NDArray[np.complex128]:
    """Return the complex impedance Z = R + jX of a series R-L-C load."""
    resistance = _positive(resistance_ohm, "resistance_ohm", allow_zero=True)
    x = reactance(frequency_hz, inductance_h, capacitance_f)
    return resistance + 1j * x


def impedance_magnitude(resistance_ohm: ArrayLike, reactance_ohm: ArrayLike) -> FloatArray:
    """Return |Z| = sqrt(R^2 + X^2) in ohm."""
    return np.hypot(_finite(resistance_ohm, "resistance_ohm"), _finite(reactance_ohm, "reactance_ohm"))


def phase_angle(resistance_ohm: ArrayLike, reactance_ohm: ArrayLike) -> FloatArray:
    """Return phi = atan2(X, R) in radians, in (-pi, pi]."""
    return np.arctan2(_finite(reactance_ohm, "reactance_ohm"), _finite(resistance_ohm, "resistance_ohm"))


def resonant_frequency(inductance_h: ArrayLike, capacitance_f: ArrayLike) -> FloatArray:
    """Return the series resonance f0 = 1 / (2*pi*sqrt(LC)) in Hz (where X = 0)."""
    inductance = _positive(inductance_h, "inductance_h")
    capacitance = _positive(capacitance_f, "capacitance_f")
    return 1.0 / (2.0 * np.pi * np.sqrt(inductance * capacitance))


@dataclass(frozen=True)
class IdealMeasurements:
    """Noise-free steady-state measurements of the load."""

    v_rms_v: FloatArray
    i_rms_a: FloatArray
    phase_rad: FloatArray


def ideal_measurements(
    v_rms_v: ArrayLike, resistance_ohm: ArrayLike, reactance_ohm: ArrayLike
) -> IdealMeasurements:
    """Compute noise-free V_rms, I_rms = V_rms/|Z| and phase for a given equivalent Z.

    Raises
    ------
    ValueError
        If V_rms is not strictly positive, R is negative, any input is not finite,
        or |Z| is below ``MIN_IMPEDANCE_OHM`` (current would be undefined).
    """
    v = _positive(v_rms_v, "v_rms_v")
    r = _positive(resistance_ohm, "resistance_ohm", allow_zero=True)
    x = _finite(reactance_ohm, "reactance_ohm")
    z_mag = np.hypot(r, x)
    if np.any(z_mag < MIN_IMPEDANCE_OHM):
        raise ValueError(f"|Z| must be at least {MIN_IMPEDANCE_OHM} ohm")
    v_b, _ = np.broadcast_arrays(v, z_mag)
    return IdealMeasurements(
        v_rms_v=v_b.copy()[()],
        i_rms_a=(v / z_mag)[()],
        phase_rad=np.arctan2(x, r)[()],
    )
