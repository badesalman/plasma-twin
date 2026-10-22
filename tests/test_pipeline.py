import numpy as np
import pytest

from src.measurements import NoiseConfig
from src.pipeline import run_experiment
from src.simulator import SimulationConfig


def test_noiseless_pipeline_is_exact():
    res = run_experiment(SimulationConfig(scenario="step", noise=NoiseConfig.preset("off")))
    assert res.metrics_direct.rmse_z_ohm == pytest.approx(0.0, abs=1e-9)
    assert res.events_direct.n_detected_events == 1
    assert res.events_direct.n_false_alarm_events == 0
    assert res.events_kalman.n_false_alarm_events == 0


def test_pipeline_is_deterministic():
    cfg = SimulationConfig(scenario="intermittent", noise=NoiseConfig.preset("medium"), seed=3)
    a, b = run_experiment(cfg), run_experiment(cfg)
    np.testing.assert_array_equal(a.kalman.r_ohm, b.kalman.r_ohm)
    np.testing.assert_array_equal(a.anomaly_direct.alarm, b.anomaly_direct.alarm)


def test_stable_load_low_noise_has_no_false_alarms():
    res = run_experiment(SimulationConfig(scenario="stable", seed=1))
    assert not res.simulation.anomaly_true.any()
    assert not res.anomaly_direct.alarm.any()
    assert not res.anomaly_kalman.alarm.any()
    assert res.detection_direct.recall is None        # no true events: recall undefined


def test_step_is_detected_by_both_detectors_at_low_noise():
    res = run_experiment(SimulationConfig(scenario="step", seed=1))
    assert res.events_direct.n_detected_events == 1
    assert res.events_kalman.n_detected_events == 1
    assert res.events_direct.n_false_alarm_events == 0


def test_dataframe_export_has_all_columns():
    df = run_experiment(SimulationConfig(n_steps=100)).to_dataframe()
    for col in ("r_true_ohm", "r_direct_ohm", "r_kalman_ohm", "x_kalman_ohm",
                "alarm_direct", "alarm_kalman", "anomaly_true"):
        assert col in df.columns
    assert len(df) == 100
