import numpy as np
import pytest

from src.anomaly_detection import AnomalyConfig, detect_anomalies, find_segments


def _series(n=200, r0=10.0, x0=24.0):
    return np.full(n, r0), np.full(n, x0)


def test_find_segments():
    assert find_segments([0, 1, 1, 0, 0, 1, 0, 1, 1]) == [(1, 3), (5, 6), (7, 9)]
    assert find_segments([0, 0, 0]) == []
    assert find_segments([1, 1]) == [(0, 2)]
    with pytest.raises(ValueError):
        find_segments(np.zeros((2, 2)))


def test_stable_signal_raises_no_alarm():
    r, x = _series()
    res = detect_anomalies(r, x)
    assert not res.alarm.any()
    assert res.baseline_r_ohm == pytest.approx(10.0)
    assert res.baseline_z_ohm == pytest.approx(np.hypot(10, 24))


def test_step_is_detected_after_persistence_samples():
    r, x = _series()
    r[100:] += 5.0
    x[100:] += 4.0
    res = detect_anomalies(r, x, AnomalyConfig(persistence=3))
    assert not res.alarm[:102].any()
    assert res.alarm[102:].all()
    assert res.events == [(102, 200)]


def test_isolated_spikes_are_suppressed_by_persistence():
    r, x = _series()
    r[100] += 5.0
    assert not detect_anomalies(r, x, AnomalyConfig(persistence=3)).alarm.any()
    r[101] += 5.0
    assert not detect_anomalies(r, x, AnomalyConfig(persistence=3)).alarm.any()
    r[102] += 5.0
    res = detect_anomalies(r, x, AnomalyConfig(persistence=3))
    assert res.alarm[102] and not res.alarm[101] and not res.alarm[103]
    assert detect_anomalies(r, x, AnomalyConfig(persistence=1)).alarm[100]


def test_threshold_floor_when_baseline_is_noise_free():
    r, x = _series()
    res = detect_anomalies(r, x, AnomalyConfig(min_relative_change=0.03))
    expected = 0.03 * np.hypot(10, 24)
    assert res.threshold_r_ohm == pytest.approx(expected)
    # a change smaller than the floor is ignored, a larger one is flagged
    small = r.copy(); small[100:] += 0.5 * expected
    big = r.copy(); big[100:] += 1.5 * expected
    assert not detect_anomalies(small, x).alarm.any()
    assert detect_anomalies(big, x).alarm.any()


def test_larger_k_sigma_gives_fewer_alarms():
    rng = np.random.default_rng(1)
    r = 10.0 + rng.normal(0, 1.0, 300)
    x = np.full(300, 24.0)
    r[150:] += 4.0
    loose = detect_anomalies(r, x, AnomalyConfig(k_sigma=2.0, min_relative_change=0.0))
    strict = detect_anomalies(r, x, AnomalyConfig(k_sigma=10.0, min_relative_change=0.0))
    assert loose.alarm.sum() > 0
    assert strict.alarm.sum() == 0
    assert strict.threshold_r_ohm > loose.threshold_r_ohm


def test_absolute_impedance_band_rule():
    r, x = _series()
    z0 = np.hypot(10, 24)
    r[100:103] += 2.0                     # 7 % change: below the (huge) relative floor
    cfg = AnomalyConfig(min_relative_change=0.5, z_max_ohm=z0 + 0.5)
    assert detect_anomalies(r, x, cfg).alarm[102]
    assert not detect_anomalies(r, x, AnomalyConfig(min_relative_change=0.5)).alarm.any()
    # lower band above the baseline: every sample after the baseline window is out of band
    low = detect_anomalies(*_series(), AnomalyConfig(z_min_ohm=z0 + 1.0))
    assert not low.exceeds[:50].any()
    assert low.exceeds[50:].all()
    assert not low.alarm[:52].any()       # persistence of 3 -> first alarm at index 52
    assert low.alarm[52:].all()


def test_no_alarm_inside_baseline_window():
    r, x = _series()
    r[10:15] += 50.0
    res = detect_anomalies(r, x, AnomalyConfig(baseline_steps=50, k_sigma=50.0))
    assert not res.alarm[:50].any()


def test_nan_samples_never_exceed_and_are_counted():
    r, x = _series()
    r[100:110] = np.nan
    res = detect_anomalies(r, x)
    assert not res.alarm.any()
    assert res.n_nan_samples == 10


def test_invalid_inputs_raise():
    r, x = _series(n=40)
    with pytest.raises(ValueError):
        detect_anomalies(r[:50], x[:50], AnomalyConfig(baseline_steps=50))   # too short
    with pytest.raises(ValueError):
        detect_anomalies(r, x[:-1])
    r2 = r.copy(); r2[:48] = np.nan
    with pytest.raises(ValueError):
        detect_anomalies(r2, x, AnomalyConfig(baseline_steps=30))
    for bad in (dict(baseline_steps=2), dict(k_sigma=0.0), dict(persistence=0),
                dict(min_relative_change=-0.1), dict(z_min_ohm=-1.0),
                dict(z_min_ohm=10.0, z_max_ohm=5.0)):
        with pytest.raises(ValueError):
            AnomalyConfig(**bad)
