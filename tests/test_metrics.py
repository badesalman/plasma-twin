import numpy as np
import pytest

from src import metrics


def test_rmse_and_mae_known_values():
    true = [0.0, 0.0, 0.0, 0.0]
    est = [1.0, -1.0, 1.0, -1.0]
    assert metrics.rmse(true, est) == pytest.approx(1.0)
    assert metrics.mae(true, est) == pytest.approx(1.0)
    assert metrics.rmse([1, 2, 3], [1, 2, 6]) == pytest.approx(np.sqrt(3.0))
    assert metrics.mae([1, 2, 3], [1, 2, 6]) == pytest.approx(1.0)


def test_nan_pairs_are_excluded():
    assert metrics.rmse([1.0, 2.0, 3.0], [1.0, np.nan, 3.0]) == pytest.approx(0.0)


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        metrics.rmse([1, 2], [1, 2, 3])
    with pytest.raises(ValueError):
        metrics.rmse([], [])
    with pytest.raises(ValueError):
        metrics.rmse([1.0], [np.nan])


def test_relative_rmse_undefined_for_zero_signal():
    assert metrics.relative_rmse([0, 0, 0], [1, 1, 1]) is None
    assert metrics.relative_rmse([10, 10], [11, 11]) == pytest.approx(0.1)


def test_mape_masks_small_true_values():
    assert metrics.mape([0.0, 10.0], [5.0, 11.0], min_abs_true=1.0) == pytest.approx(10.0)
    assert metrics.mape([0.0, 0.1], [1.0, 1.0], min_abs_true=1.0) is None


def test_estimation_metrics_complex_error():
    m = metrics.estimation_metrics([3.0, 3.0], [4.0, 4.0], [3.0, 3.0], [5.0, 5.0])
    assert m.rmse_r_ohm == pytest.approx(0.0)
    assert m.rmse_x_ohm == pytest.approx(1.0)
    assert m.rmse_z_ohm == pytest.approx(1.0)
    assert m.n_samples == 2 and m.n_excluded_nan == 0
    m2 = metrics.estimation_metrics([3.0, 3.0], [4.0, 4.0], [3.0, np.nan], [4.0, 4.0])
    assert m2.n_samples == 1 and m2.n_excluded_nan == 1


def test_sample_detection_confusion_matrix():
    true = [0, 0, 1, 1, 0]
    alarm = [0, 1, 1, 0, 0]
    m = metrics.sample_detection_metrics(true, alarm)
    assert (m.tp, m.fp, m.fn, m.tn) == (1, 1, 1, 2)
    assert m.precision == pytest.approx(0.5)
    assert m.recall == pytest.approx(0.5)
    assert m.f1 == pytest.approx(0.5)
    assert m.false_positive_rate == pytest.approx(1 / 3)


def test_undefined_detection_metrics_are_none():
    no_alarm = metrics.sample_detection_metrics([0, 1, 1], [0, 0, 0])
    assert no_alarm.precision is None and no_alarm.f1 is None
    assert no_alarm.recall == 0.0
    no_events = metrics.sample_detection_metrics([0, 0, 0], [0, 1, 0])
    assert no_events.recall is None and no_events.precision == 0.0
    assert metrics.event_detection_metrics([0, 0, 0], [0, 0, 0]).event_recall is None
    with pytest.raises(ValueError):
        metrics.sample_detection_metrics([0, 1], [0])


def test_event_metrics_delay_and_false_alarms():
    true = np.zeros(50, bool); true[10:20] = True
    alarm = np.zeros(50, bool); alarm[13:16] = True; alarm[30:33] = True
    m = metrics.event_detection_metrics(true, alarm)
    assert m.n_true_events == 1 and m.n_detected_events == 1
    assert m.detection_delays_steps == (3,)
    assert m.mean_delay_steps == pytest.approx(3.0)
    assert m.n_alarm_events == 2 and m.n_false_alarm_events == 1
    assert m.event_recall == 1.0


def test_event_metrics_grace_period_and_missed_events():
    true = np.zeros(50, bool); true[10:20] = True
    late = np.zeros(50, bool); late[22:25] = True
    assert metrics.event_detection_metrics(true, late, grace_steps=0).n_detected_events == 0
    m = metrics.event_detection_metrics(true, late, grace_steps=5)
    assert m.n_detected_events == 1 and m.detection_delays_steps == (12,)
    assert m.n_false_alarm_events == 0
    with pytest.raises(ValueError):
        metrics.event_detection_metrics(true, late, grace_steps=-1)


def test_alarm_active_before_event_counts_with_zero_delay():
    true = np.zeros(30, bool); true[10:20] = True
    alarm = np.zeros(30, bool); alarm[5:15] = True
    m = metrics.event_detection_metrics(true, alarm)
    assert m.detection_delays_steps == (0,)
    assert m.n_false_alarm_events == 0


def test_settling_delay():
    est_r = np.array([0, 0, 0, 2, 4, 6, 8, 9, 10, 10], float)
    est_x = np.zeros(10)
    assert metrics.settling_delay(est_r, est_x, 0 + 0j, 10 + 0j, 3, fraction=0.5) == 2
    assert metrics.settling_delay(est_r, est_x, 0 + 0j, 10 + 0j, 3, fraction=0.2) == 3
    assert metrics.settling_delay(np.zeros(10), est_x, 0j, 10 + 0j, 3) is None
    assert metrics.settling_delay(est_r, est_x, 5 + 0j, 5 + 0j, 3) is None
    nan = est_r.copy(); nan[5:8] = np.nan          # NaN is never "close": no 3 valid samples in a row
    assert metrics.settling_delay(nan, est_x, 0j, 10 + 0j, 3, fraction=0.5) is None
    for bad in (dict(fraction=0.0), dict(fraction=1.0), dict(hold=0)):
        with pytest.raises(ValueError):
            metrics.settling_delay(est_r, est_x, 0j, 10 + 0j, 3, **bad)
