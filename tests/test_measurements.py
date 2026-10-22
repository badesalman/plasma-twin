import numpy as np
import pytest

from src.measurements import MeasurementSeries, NoiseConfig, add_measurement_noise


def _noisy(seed, noise=None, n=50, scale=1.0):
    rng = np.random.default_rng(seed)
    return add_measurement_noise(np.full(n, 100.0), np.full(n, 4.0), np.full(n, 0.5),
                                 noise or NoiseConfig(), rng, scale)


def test_same_seed_is_reproducible():
    a, b = _noisy(7), _noisy(7)
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x, y)


def test_different_seeds_differ():
    assert not np.array_equal(_noisy(1)[0], _noisy(2)[0])


def test_zero_noise_returns_ideal_values():
    v, i, phi = _noisy(3, NoiseConfig.preset("off"))
    np.testing.assert_array_equal(v, 100.0)
    np.testing.assert_array_equal(i, 4.0)
    np.testing.assert_array_equal(phi, 0.5)


def test_noise_statistics_match_configuration():
    noise = NoiseConfig(v_rel_std=0.02, i_rel_std=0.03, phase_std_rad=0.01)
    v, i, phi = _noisy(0, noise, n=200_000)
    assert np.std(v / 100.0 - 1.0) == pytest.approx(0.02, rel=0.02)
    assert np.std(i / 4.0 - 1.0) == pytest.approx(0.03, rel=0.02)
    assert np.std(phi - 0.5) == pytest.approx(0.01, rel=0.02)
    assert abs(np.mean(v) / 100.0 - 1.0) < 1e-3


def test_noise_scale_scales_the_deviation():
    v1, _, _ = _noisy(5, scale=1.0)
    v3, _, _ = _noisy(5, scale=3.0)
    np.testing.assert_allclose(v3 - 100.0, 3.0 * (v1 - 100.0))


def test_same_underlying_draws_across_noise_levels():
    low = _noisy(9, NoiseConfig.preset("low"))[0] - 100.0
    high = _noisy(9, NoiseConfig.preset("high"))[0] - 100.0
    np.testing.assert_allclose(high, 5.0 * low)      # 5% vs 1% relative noise


def test_presets_and_invalid_inputs():
    assert NoiseConfig.preset("low").v_rel_std == pytest.approx(0.01)
    assert NoiseConfig.preset("low").phase_std_rad == pytest.approx(np.deg2rad(0.5))
    with pytest.raises(ValueError):
        NoiseConfig.preset("extreme")
    with pytest.raises(ValueError):
        NoiseConfig(v_rel_std=-0.1)
    with pytest.raises(ValueError):
        NoiseConfig(i_rel_std=0.9)
    with pytest.raises(ValueError):
        NoiseConfig(phase_std_rad=np.nan)
    with pytest.raises(ValueError):
        _noisy(0, scale=-1.0)


def test_measurement_series_shape_check():
    MeasurementSeries(np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3))
    with pytest.raises(ValueError):
        MeasurementSeries(np.zeros(3), np.zeros(2), np.zeros(3), np.zeros(3))
