"""Error and detection metrics.

Undefined metrics are reported as ``None`` (never as 0 or NaN) so that a
missing value cannot be mistaken for a good one. NaN estimates are excluded
from error metrics and counted; if no valid pair remains, ValueError is raised.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from src.anomaly_detection import find_segments

FloatArray = NDArray[np.float64]


def _paired(true: ArrayLike, est: ArrayLike) -> tuple[FloatArray, FloatArray, int]:
    t = np.asarray(true, dtype=float)
    e = np.asarray(est, dtype=float)
    if t.shape != e.shape:
        raise ValueError("true and estimated arrays must have the same shape")
    if t.size == 0:
        raise ValueError("arrays must not be empty")
    ok = np.isfinite(t) & np.isfinite(e)
    if not np.any(ok):
        raise ValueError("no valid (finite) sample pairs")
    return t[ok], e[ok], int(t.size - np.count_nonzero(ok))


def rmse(true: ArrayLike, est: ArrayLike) -> float:
    """Root mean squared error (same unit as the inputs)."""
    t, e, _ = _paired(true, est)
    return float(np.sqrt(np.mean((e - t) ** 2)))


def mae(true: ArrayLike, est: ArrayLike) -> float:
    """Mean absolute error."""
    t, e, _ = _paired(true, est)
    return float(np.mean(np.abs(e - t)))


def relative_rmse(true: ArrayLike, est: ArrayLike, min_rms_true: float = 1e-9) -> float | None:
    """RMSE divided by the RMS of the true signal; None if the true signal is ~0."""
    t, e, _ = _paired(true, est)
    rms_true = float(np.sqrt(np.mean(t**2)))
    if rms_true < min_rms_true:
        return None
    return float(np.sqrt(np.mean((e - t) ** 2)) / rms_true)


def mape(true: ArrayLike, est: ArrayLike, min_abs_true: float) -> float | None:
    """Mean absolute percentage error over samples with |true| >= min_abs_true (None if none)."""
    t, e, _ = _paired(true, est)
    keep = np.abs(t) >= min_abs_true
    if not np.any(keep):
        return None
    return float(100.0 * np.mean(np.abs(e[keep] - t[keep]) / np.abs(t[keep])))


@dataclass(frozen=True)
class EstimationMetrics:
    """Accuracy of an (R, X) estimate against ground truth."""

    rmse_r_ohm: float
    rmse_x_ohm: float
    mae_r_ohm: float
    mae_x_ohm: float
    rmse_z_ohm: float
    relative_rmse_r: float | None
    relative_rmse_x: float | None
    n_samples: int
    n_excluded_nan: int


def estimation_metrics(
    r_true: ArrayLike, x_true: ArrayLike, r_est: ArrayLike, x_est: ArrayLike
) -> EstimationMetrics:
    """Compute RMSE/MAE for R and X and the complex-impedance RMSE |Z_est - Z_true|."""
    rt, re, _ = _paired(r_true, r_est)
    xt, xe, _ = _paired(x_true, x_est)
    r_all = np.asarray(r_true, dtype=float)
    x_all = np.asarray(x_true, dtype=float)
    ok = (
        np.isfinite(r_all) & np.isfinite(x_all)
        & np.isfinite(np.asarray(r_est, dtype=float)) & np.isfinite(np.asarray(x_est, dtype=float))
    )
    dz = (np.asarray(r_est, dtype=float)[ok] - r_all[ok]) + 1j * (
        np.asarray(x_est, dtype=float)[ok] - x_all[ok]
    )
    return EstimationMetrics(
        rmse_r_ohm=rmse(r_true, r_est),
        rmse_x_ohm=rmse(x_true, x_est),
        mae_r_ohm=mae(r_true, r_est),
        mae_x_ohm=mae(x_true, x_est),
        rmse_z_ohm=float(np.sqrt(np.mean(np.abs(dz) ** 2))),
        relative_rmse_r=relative_rmse(r_true, r_est),
        relative_rmse_x=relative_rmse(x_true, x_est),
        n_samples=int(np.count_nonzero(ok)),
        n_excluded_nan=int(ok.size - np.count_nonzero(ok)),
    )


@dataclass(frozen=True)
class SampleDetectionMetrics:
    """Per-sample confusion matrix of alarms against ground-truth labels."""

    tp: int
    fp: int
    fn: int
    tn: int
    precision: float | None
    recall: float | None
    f1: float | None
    false_positive_rate: float | None


def _ratio(num: float, den: float) -> float | None:
    return None if den == 0 else float(num / den)


def sample_detection_metrics(true_flags: ArrayLike, alarm_flags: ArrayLike) -> SampleDetectionMetrics:
    """Sample-level precision, recall, F1 and false-positive rate (None when undefined)."""
    t = np.asarray(true_flags, dtype=bool)
    a = np.asarray(alarm_flags, dtype=bool)
    if t.shape != a.shape or t.ndim != 1 or t.size == 0:
        raise ValueError("flags must be non-empty one-dimensional arrays of equal length")
    tp = int(np.sum(t & a))
    fp = int(np.sum(~t & a))
    fn = int(np.sum(t & ~a))
    tn = int(np.sum(~t & ~a))
    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, tp + fn)
    f1 = None if precision is None or recall is None or (precision + recall) == 0 else (
        2 * precision * recall / (precision + recall)
    )
    return SampleDetectionMetrics(tp, fp, fn, tn, precision, recall, f1, _ratio(fp, fp + tn))


@dataclass(frozen=True)
class EventDetectionMetrics:
    """Event-level detection: was each true disturbance detected, how fast, any false alarms?"""

    n_true_events: int
    n_detected_events: int
    event_recall: float | None
    detection_delays_steps: tuple[int, ...]
    mean_delay_steps: float | None
    n_alarm_events: int
    n_false_alarm_events: int


def event_detection_metrics(
    true_flags: ArrayLike, alarm_flags: ArrayLike, grace_steps: int = 0
) -> EventDetectionMetrics:
    """Match alarm segments to true disturbance segments.

    A true event [s, e) counts as detected if any alarm sample lies in
    [s, e + grace_steps). Its delay is (first such alarm sample) - s; an alarm
    that is already active at s gives delay 0. An alarm segment that does not
    overlap any true window [s, e + grace_steps) is a false-alarm event.
    """
    t = np.asarray(true_flags, dtype=bool)
    a = np.asarray(alarm_flags, dtype=bool)
    if t.shape != a.shape or t.ndim != 1 or t.size == 0:
        raise ValueError("flags must be non-empty one-dimensional arrays of equal length")
    if grace_steps < 0:
        raise ValueError("grace_steps must be non-negative")

    true_events = find_segments(t)
    alarm_events = find_segments(a)
    delays: list[int] = []
    for s, e in true_events:
        hits = np.flatnonzero(a[s : min(e + grace_steps, a.size)])
        if hits.size:
            delays.append(int(hits[0]))
    false_alarms = sum(
        1
        for a_s, a_e in alarm_events
        if not any(a_s < e + grace_steps and a_e > s for s, e in true_events)
    )
    return EventDetectionMetrics(
        n_true_events=len(true_events),
        n_detected_events=len(delays),
        event_recall=_ratio(len(delays), len(true_events)),
        detection_delays_steps=tuple(delays),
        mean_delay_steps=None if not delays else float(np.mean(delays)),
        n_alarm_events=len(alarm_events),
        n_false_alarm_events=false_alarms,
    )


def settling_delay(
    r_est: ArrayLike,
    x_est: ArrayLike,
    z_before: complex,
    z_after: complex,
    start_index: int,
    fraction: float = 0.5,
    hold: int = 3,
) -> int | None:
    """Samples after ``start_index`` until the estimate stays near the new impedance.

    "Near" means ||Z_est - Z_after|| <= fraction * ||Z_after - Z_before||, for ``hold``
    consecutive samples. Returns None if that never happens (or the step is zero).
    """
    if not 0 < fraction < 1:
        raise ValueError("fraction must be in (0, 1)")
    if hold < 1:
        raise ValueError("hold must be >= 1")
    r = np.asarray(r_est, dtype=float)
    x = np.asarray(x_est, dtype=float)
    if r.shape != x.shape or r.ndim != 1:
        raise ValueError("r_est and x_est must be one-dimensional arrays of equal length")
    step = abs(z_after - z_before)
    if step == 0:
        return None
    with np.errstate(invalid="ignore"):
        close = np.abs((r + 1j * x) - z_after) <= fraction * step   # NaN compares False
    for k in range(start_index, r.size - hold + 1):
        if np.all(close[k : k + hold]):
            return k - start_index
    return None
