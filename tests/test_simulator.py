import numpy as np
import pytest

from src import physics
from src.measurements import NoiseConfig
from src.simulator import Scenario, SimulationConfig, simulate
from src.anomaly_detection import find_segments

NOISELESS = NoiseConfig.preset("off")


def cfg(**kw):
    kw.setdefault("noise", NOISELESS)
    return SimulationConfig(**kw)


def test_simulation_is_deterministic_for_fixed_seed():
    a = simulate(SimulationConfig(scenario="step", seed=11, noise=NoiseConfig.preset("medium")))
    b = simulate(SimulationConfig(scenario="step", seed=11, noise=NoiseConfig.preset("medium")))
    np.testing.assert_array_equal(a.v_meas_v, b.v_meas_v)
    np.testing.assert_array_equal(a.i_meas_a, b.i_meas_a)
    np.testing.assert_array_equal(a.phase_meas_rad, b.phase_meas_rad)


def test_different_seed_changes_measurements_but_not_truth():
    a = simulate(SimulationConfig(seed=1))
    b = simulate(SimulationConfig(seed=2))
    assert not np.array_equal(a.v_meas_v, b.v_meas_v)
    np.testing.assert_array_equal(a.r_true_ohm, b.r_true_ohm)


def test_stable_load_is_constant_and_has_no_true_anomaly():
    r = simulate(cfg(scenario="stable"))
    assert np.all(r.r_true_ohm == r.r_true_ohm[0])
    assert np.all(r.x_true_ohm == r.x_true_ohm[0])
    assert not np.any(r.anomaly_true)


def test_default_operating_point():
    r = simulate(cfg())
    assert r.x_true_ohm[0] == pytest.approx(24.0, abs=0.1)
    assert r.x_true_ohm[0] == pytest.approx(
        float(physics.reactance(13.56e6, 1.2e-6, 150e-12)))


def test_noiseless_measurements_equal_ideal_and_obey_ohms_law():
    r = simulate(cfg(scenario="drift"))
    np.testing.assert_array_equal(r.v_meas_v, r.v_ideal_v)
    np.testing.assert_allclose(r.v_ideal_v / r.i_ideal_a, r.z_mag_true_ohm)
    np.testing.assert_allclose(r.phase_true_rad, np.arctan2(r.x_true_ohm, r.r_true_ohm))


def test_step_changes_load_at_configured_index():
    c = cfg(scenario="step")
    r = simulate(c)
    k = c.step_index
    assert r.r_true_ohm[k - 1] == pytest.approx(10.0)
    assert r.r_true_ohm[k] == pytest.approx(15.0)
    assert r.x_true_ohm[k] > r.x_true_ohm[k - 1]
    assert not np.any(r.anomaly_true[:k])
    assert np.all(r.anomaly_true[k:])


def test_drift_is_monotonic_and_starts_after_baseline():
    c = cfg(scenario="drift")
    r = simulate(c)
    assert np.all(np.diff(r.r_true_ohm) >= 0)
    assert np.all(np.diff(r.x_true_ohm) >= 0)
    assert np.all(r.r_true_ohm[: c.baseline_steps] == r.r_true_ohm[0])
    assert r.r_true_ohm[-1] == pytest.approx(10.0 * 1.5)
    assert 0 < r.anomaly_true.sum() < c.n_steps


def test_noisy_scenario_keeps_load_but_ramps_noise():
    c = cfg(scenario="noisy")
    r = simulate(c)
    assert np.all(r.r_true_ohm == r.r_true_ohm[0])
    assert r.noise_scale[0] == 1.0 and r.noise_scale[c.baseline_steps - 1] == 1.0
    assert r.noise_scale[-1] == pytest.approx(c.noise_ramp_factor)
    assert np.all(np.diff(r.noise_scale) >= 0)
    assert not np.any(r.anomaly_true)


def test_noisy_scenario_has_larger_error_late_than_early():
    c = SimulationConfig(scenario="noisy", noise=NoiseConfig.preset("low"), n_steps=2000, seed=3)
    r = simulate(c)
    err = np.abs(r.v_meas_v - r.v_ideal_v)
    assert err[-300:].mean() > 2.0 * err[: c.baseline_steps].mean()


def test_intermittent_has_configured_number_and_length_of_pulses():
    c = cfg(scenario="intermittent", n_pulses=3, pulse_steps=30)
    r = simulate(c)
    segments = find_segments(r.anomaly_true)
    assert len(segments) == 3
    assert all(e - s == 30 for s, e in segments)
    assert segments[0][0] > c.baseline_steps
    # between pulses the load returns to its initial value
    assert r.r_true_ohm[segments[0][1]] == pytest.approx(10.0)


def test_config_dict_and_json_roundtrip():
    c = SimulationConfig(scenario="intermittent", seed=5, noise=NoiseConfig.preset("high"))
    assert SimulationConfig.from_dict(c.to_dict()) == c
    assert SimulationConfig.from_json(c.to_json()) == c


def test_scenario_accepts_string_and_rejects_unknown():
    assert SimulationConfig(scenario="drift").scenario is Scenario.DRIFT
    with pytest.raises(ValueError):
        SimulationConfig(scenario="chaos")


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(frequency_hz=0.0), dict(frequency_hz=-1e6), dict(v_rms_v=0.0), dict(v_rms_v=np.nan),
        dict(resistance_ohm=-1.0), dict(inductance_h=-1e-6), dict(capacitance_f=0.0),
        dict(n_steps=5), dict(n_steps=100.5), dict(dt_s=0.0), dict(seed=-1),
        dict(baseline_steps=2), dict(baseline_steps=600), dict(event_r_factor=0.0),
        dict(noise_ramp_factor=0.5), dict(label_tolerance=0.0),
        dict(scenario="step", event_start_fraction=0.05),
        dict(scenario="step", event_start_fraction=1.2),
        dict(scenario="intermittent", pulse_steps=500),
        dict(resistance_ohm=0.0, inductance_h=0.0, capacitance_f=1e300),
    ],
)
def test_invalid_configuration_raises(kwargs):
    with pytest.raises(ValueError):
        SimulationConfig(**kwargs)


def test_from_dict_rejects_unknown_keys():
    with pytest.raises(ValueError):
        SimulationConfig.from_dict({"not_a_parameter": 1})


def test_dataframe_columns_and_length():
    df = simulate(SimulationConfig(n_steps=100)).to_dataframe()
    assert len(df) == 100
    for col in ("time_s", "r_true_ohm", "x_true_ohm", "v_meas_v", "i_meas_a",
                "phase_meas_deg", "anomaly_true"):
        assert col in df.columns
