"""Baseline (direct, memoryless) impedance estimator.

Given measured V_rms, I_rms and phase phi:

    |Z_est| = V / I
    R_est   = |Z_est| cos(phi)
    X_est   = |Z_est| sin(phi)

Each sample is processed independently, so measurement noise passes straight
through (and is transformed non-linearly). The Kalman filter (a later stage)
will be compared against this baseline.

Note: R_est can become negative under noise when |phi| approaches or exceeds
pi/2. The estimator deliberately does not clip it, because clipping would hide
the effect of noise and bias error metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]

DEFAULT_MIN_CURRENT_A: float = 1e-12


@dataclass(frozen=True)
class ImpedanceEstimate:
    """Result of an impedance estimate (scalars or arrays)."""

    z_mag_ohm: FloatArray
    r_ohm: FloatArray
    x_ohm: FloatArray
    phase_rad: FloatArray

    @property
    def z_complex(self) -> NDArray[np.complex128]:
        """Complex impedance R + jX."""
        return self.r_ohm + 1j * self.x_ohm


def estimate_impedance_direct(
    v_rms_v: ArrayLike,
    i_rms_a: ArrayLike,
    phase_rad: ArrayLike,
    *,
    min_current_a: float = DEFAULT_MIN_CURRENT_A,
    invalid: Literal["raise", "nan"] = "raise",
) -> ImpedanceEstimate:
    """Estimate R and X directly from V_rms, I_rms and phase (radians).

    A sample is *invalid* if any input is not finite, V <= 0, or I <= ``min_current_a``
    (division by zero or near-zero current).

    Parameters
    ----------
    invalid:
        ``"raise"`` (default) raises ``ValueError`` if any sample is invalid.
        ``"nan"`` returns NaN for invalid samples and processes the rest.
    """
    if invalid not in ("raise", "nan"):
        raise ValueError("invalid must be 'raise' or 'nan'")
    if not np.isfinite(min_current_a) or min_current_a <= 0:
        raise ValueError("min_current_a must be a positive finite number")

    v, i, phi = np.broadcast_arrays(
        np.asarray(v_rms_v, dtype=float),
        np.asarray(i_rms_a, dtype=float),
        np.asarray(phase_rad, dtype=float),
    )

    with np.errstate(invalid="ignore"):
        valid = np.isfinite(v) & np.isfinite(i) & np.isfinite(phi) & (v > 0) & (i > min_current_a)

    if invalid == "raise" and not np.all(valid):
        bad = int(np.size(valid) - np.count_nonzero(valid))
        raise ValueError(
            f"{bad} invalid sample(s): need finite inputs, V > 0 and I > {min_current_a} A"
        )

    safe_v = np.where(valid, v, 1.0)
    safe_i = np.where(valid, i, 1.0)
    safe_phi = np.where(valid, phi, 0.0)

    z_mag = safe_v / safe_i
    r = z_mag * np.cos(safe_phi)
    x = z_mag * np.sin(safe_phi)

    z_mag, r, x, phi_out = (np.where(valid, a, np.nan)[()] for a in (z_mag, r, x, phi))
    return ImpedanceEstimate(z_mag_ohm=z_mag, r_ohm=r, x_ohm=x, phase_rad=phi_out)
