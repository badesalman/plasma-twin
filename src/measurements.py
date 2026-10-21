"""Measurement data containers.

Stage 1 only defines the container. The noise model (configurable relative
noise on V and I, additive noise on phase, seeded RNG) is added in Stage 2.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


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
