"""Transparent, rule-based anomaly detection on estimated R and X.

This is NOT a trained machine-learning model. The rules are:

1. Baseline: mean and standard deviation of the estimates over the first
   ``baseline_steps`` samples (assumed to be normal operation).
2. Deviation rule: a sample *exceeds* when
       |R - R_base| > thr_R   or   |X - X_base| > thr_X   or   ||Z| - |Z|_base| > thr_Z
   with   thr = max(k_sigma * sigma_base, min_relative_change * |Z|_base).
   The floor stops the threshold from collapsing when the baseline is almost
   noise-free (e.g. a heavily filtered signal).
3. Optional absolute band on |Z|: exceed if |Z| < z_min or |Z| > z_max.
4. Persistence: an alarm is raised only after ``persistence`` consecutive
   exceeding samples (suppresses isolated noise spikes) and stays on while
   samples keep exceeding. No alarm is raised inside the baseline window.

Vocabulary used throughout the project:
* true disturbance  - ground-truth label from the simulator (``anomaly_true``)
* detected anomaly  - alarm raised by this detector
* false alarm       - alarm with no true disturbance (see metrics.py)

Sudden changes appear as an immediate deviation from the baseline; gradual
drift is caught once it has accumulated beyond the thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

BoolArray = NDArray[np.bool_]
FloatArray = NDArray[np.float64]

MIN_BASELINE_SAMPLES = 5


@dataclass(frozen=True)
class AnomalyConfig:
    """Detector settings."""

    baseline_steps: int = 50
    k_sigma: float = 5.0
    min_relative_change: float = 0.03
    persistence: int = 3
    z_min_ohm: float | None = None
    z_max_ohm: float | None = None

    def __post_init__(self) -> None:
        if int(self.baseline_steps) != self.baseline_steps or self.baseline_steps < MIN_BASELINE_SAMPLES:
            raise ValueError(f"baseline_steps must be an integer >= {MIN_BASELINE_SAMPLES}")
        if not np.isfinite(self.k_sigma) or self.k_sigma <= 0:
            raise ValueError("k_sigma must be positive and finite")
        if not np.isfinite(self.min_relative_change) or self.min_relative_change < 0:
            raise ValueError("min_relative_change must be non-negative and finite")
        if int(self.persistence) != self.persistence or self.persistence < 1:
            raise ValueError("persistence must be an integer >= 1")
        for name in ("z_min_ohm", "z_max_ohm"):
            value = getattr(self, name)
            if value is not None and (not np.isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be positive and finite (or None)")
        if self.z_min_ohm is not None and self.z_max_ohm is not None and self.z_min_ohm >= self.z_max_ohm:
            raise ValueError("z_min_ohm must be smaller than z_max_ohm")


@dataclass(frozen=True)
class AnomalyResult:
    """Detector output."""

    alarm: BoolArray
    exceeds: BoolArray
    baseline_r_ohm: float
    baseline_x_ohm: float
    baseline_z_ohm: float
    threshold_r_ohm: float
    threshold_x_ohm: float
    threshold_z_ohm: float
    n_nan_samples: int

    @property
    def events(self) -> list[tuple[int, int]]:
        """Alarm segments as half-open index ranges [start, end)."""
        return find_segments(self.alarm)


def find_segments(flags: ArrayLike) -> list[tuple[int, int]]:
    """Return contiguous True runs of a boolean array as half-open [start, end) pairs."""
    f = np.asarray(flags, dtype=bool)
    if f.ndim != 1:
        raise ValueError("flags must be one-dimensional")
    padded = np.concatenate(([False], f, [False])).astype(np.int8)
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    return list(zip(starts.tolist(), ends.tolist()))


def detect_anomalies(
    r_est_ohm: ArrayLike, x_est_ohm: ArrayLike, config: AnomalyConfig | None = None
) -> AnomalyResult:
    """Run the detector on a series of estimated R and X (NaN samples never exceed)."""
    config = config or AnomalyConfig()
    r = np.asarray(r_est_ohm, dtype=float)
    x = np.asarray(x_est_ohm, dtype=float)
    if r.ndim != 1 or r.shape != x.shape:
        raise ValueError("r_est_ohm and x_est_ohm must be one-dimensional arrays of equal length")
    if r.size <= config.baseline_steps:
        raise ValueError("series must be longer than baseline_steps")

    z = np.hypot(r, x)
    window = slice(0, config.baseline_steps)
    base_ok = np.isfinite(r[window]) & np.isfinite(x[window])
    if np.count_nonzero(base_ok) < MIN_BASELINE_SAMPLES:
        raise ValueError("not enough valid samples in the baseline window")

    r_b, x_b, z_b = r[window][base_ok], x[window][base_ok], z[window][base_ok]
    base_r, base_x, base_z = float(r_b.mean()), float(x_b.mean()), float(z_b.mean())
    floor = config.min_relative_change * base_z
    thr_r = max(config.k_sigma * float(r_b.std(ddof=1)), floor)
    thr_x = max(config.k_sigma * float(x_b.std(ddof=1)), floor)
    thr_z = max(config.k_sigma * float(z_b.std(ddof=1)), floor)

    with np.errstate(invalid="ignore"):
        exceeds = (
            (np.abs(r - base_r) > thr_r)
            | (np.abs(x - base_x) > thr_x)
            | (np.abs(z - base_z) > thr_z)
        )
        if config.z_min_ohm is not None:
            exceeds |= z < config.z_min_ohm
        if config.z_max_ohm is not None:
            exceeds |= z > config.z_max_ohm
    exceeds &= np.isfinite(r) & np.isfinite(x)
    exceeds[window] = False

    alarm = np.zeros_like(exceeds)
    run = 0
    for k, flag in enumerate(exceeds):
        run = run + 1 if flag else 0
        alarm[k] = run >= config.persistence

    return AnomalyResult(
        alarm=alarm,
        exceeds=exceeds,
        baseline_r_ohm=base_r,
        baseline_x_ohm=base_x,
        baseline_z_ohm=base_z,
        threshold_r_ohm=thr_r,
        threshold_x_ohm=thr_x,
        threshold_z_ohm=thr_z,
        n_nan_samples=int(np.count_nonzero(~(np.isfinite(r) & np.isfinite(x)))),
    )
