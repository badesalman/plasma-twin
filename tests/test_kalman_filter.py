import numpy as np
import pytest

from src.estimators import ImpedanceEstimate, estimate_impedance_direct
from src.kalman_filter import (
    KalmanConfig, KalmanFilter, measurement_covariance, run_kalman,
)
from src.measurements import NoiseConfig
from src.simulator import SimulationConfig, simulate
from src.metrics import rmse

LOW = NoiseConfig.preset("low")


def _direct(cfg):
    s = simulate(cfg)
    return s, estimate_impedance_direct(s.v_meas_v, s.i_meas_a, s.phase_meas_rad, invalid="nan")


def test_covariance_is_radial_tangential_at_zero_phase():
    noise = NoiseConfig(v_rel_std=0.01, i_rel_std=0.02, phase_std_rad=0.005)
    cov = measurement_covariance(10.0, 0.0, noise)
    assert cov[0, 0] == pytest.approx(100 * (0.01**2 + 0.02**2), rel=1e-6)   # radial = R direction
    assert cov[1, 1] == pytest.approx(100 * 0.005**2, rel=1e-6)               # tangential = X direction
    assert cov[0, 1] == pytest.approx(0.0, abs=1e-12)


def test_covariance_symmetric_and_positive_definite_for_any_phase():
    phis = np.linspace(-np.pi, np.pi, 25)
    cov = measurement_covariance(np.full(25, 30.0), phis, NoiseConfig.preset("high"))
    assert cov.shape == (25, 2, 2)
    np.testing.assert_allclose(cov, np.swapaxes(cov, -1, -2))
    assert np.all(np.linalg.eigvalsh(cov) > 0)


def test_covariance_matches_monte_carlo_for_small_noise():
    noise = NoiseConfig(v_rel_std=0.01, i_rel_std=0.01, phase_std_rad=0.01)
    z0, phi0 = 20.0, 0.7
    rng = np.random.default_rng(0)
    n = 300_000
    v = 100.0 * (1 + noise.v_rel_std * rng.standard_normal(n))
    i = (100.0 / z0) * (1 + noise.i_rel_std * rng.standard_normal(n))
    phi = phi0 + noise.phase_std_rad * rng.standard_normal(n)
    est = estimate_impedance_direct(v, i, phi)
    empirical = np.cov(np.vstack([est.r_ohm, est.x_ohm]))
    predicted = measurement_covariance(z0, phi0, noise)
    np.testing.assert_allclose(empirical, predicted, rtol=0.05, atol=1e-3)


def test_noiseless_constant_input_stays_constant():
    s, d = _direct(SimulationConfig(noise=NoiseConfig.preset("off")))
    k = run_kalman(d, NoiseConfig.preset("off"))
    np.testing.assert_allclose(k.r_ohm, s.r_true_ohm, atol=1e-9)
    np.testing.assert_allclose(k.x_ohm, s.x_true_ohm, atol=1e-9)


def test_filter_beats_direct_on_stable_load_over_several_seeds():
    d_err, k_err = [], []
    for seed in range(10):
        s, d = _direct(SimulationConfig(noise=LOW, seed=seed))
        k = run_kalman(d, LOW).as_estimate()
        d_err.append(rmse(s.r_true_ohm, d.r_ohm) + rmse(s.x_true_ohm, d.x_ohm))
        k_err.append(rmse(s.r_true_ohm, k.r_ohm) + rmse(s.x_true_ohm, k.x_ohm))
    assert np.mean(k_err) < 0.7 * np.mean(d_err)


def test_huge_process_noise_makes_filter_follow_measurements():
    _, d = _direct(SimulationConfig(noise=LOW, seed=1))
    k = run_kalman(d, LOW, KalmanConfig(process_std_ohm=1e3))
    np.testing.assert_allclose(k.r_ohm, d.r_ohm, rtol=1e-4)
    np.testing.assert_allclose(k.x_ohm, d.x_ohm, rtol=1e-4)


def test_zero_process_noise_converges_to_running_average():
    _, d = _direct(SimulationConfig(noise=LOW, seed=2))
    k = run_kalman(d, LOW, KalmanConfig(process_std_ohm=0.0))
    assert k.std_r_ohm[-1] < k.std_r_ohm[1]            # uncertainty keeps shrinking


def test_covariance_stays_symmetric_and_positive_semidefinite():
    rng = np.random.default_rng(0)
    kf = KalmanFilter([10.0, 24.0], np.diag([0.1, 0.1]), 0.05)
    for _ in range(5000):
        kf.predict()
        z = np.array([10.0, 24.0]) + rng.normal(0, 0.3, 2)
        r_meas = measurement_covariance(26.0, 1.17, LOW)
        kf.update(z, r_meas)
        assert np.allclose(kf.p, kf.p.T)
        assert np.all(np.linalg.eigvalsh(kf.p) >= -1e-12)
    assert np.all(np.isfinite(kf.x))


def test_nan_measurements_are_skipped_and_uncertainty_grows():
    s, d = _direct(SimulationConfig(noise=LOW, seed=4, n_steps=100))
    r, x = d.r_ohm.copy(), d.x_ohm.copy()
    r[50:60] = np.nan
    x[50:60] = np.nan
    gapped = ImpedanceEstimate(d.z_mag_ohm, r, x, d.phase_rad)
    k = run_kalman(gapped, LOW)
    assert np.all(np.isfinite(k.r_ohm))
    assert np.all(np.isnan(k.innovation_r_ohm[50:60]))
    assert k.std_r_ohm[59] > k.std_r_ohm[49]
    assert k.std_r_ohm[62] < k.std_r_ohm[59]


def test_leading_invalid_samples_give_nan_until_first_valid():
    s, d = _direct(SimulationConfig(noise=LOW, n_steps=100))
    r = d.r_ohm.copy(); r[:5] = np.nan
    k = run_kalman(ImpedanceEstimate(d.z_mag_ohm, r, d.x_ohm, d.phase_rad), LOW)
    assert np.all(np.isnan(k.r_ohm[:5]))
    assert np.all(np.isfinite(k.r_ohm[5:]))


def test_invalid_parameters_raise():
    with pytest.raises(ValueError):
        KalmanConfig(process_std_ohm=-1.0)
    with pytest.raises(ValueError):
        KalmanConfig(measurement_noise_scale=0.0)
    with pytest.raises(ValueError):
        KalmanConfig(initial_std_ohm=-1.0)
    with pytest.raises(ValueError):
        KalmanFilter([np.nan, 0.0], np.eye(2), 0.1)
    with pytest.raises(ValueError):
        KalmanFilter([0.0, 0.0], np.eye(2), -0.1)
    kf = KalmanFilter([0.0, 0.0], np.zeros((2, 2)), 0.0)
    with pytest.raises(ValueError):
        kf.update([1.0, 1.0], np.zeros((2, 2)))          # singular S
