import numpy as np
import pytest

from src import physics
from src.estimators import estimate_impedance_direct


@pytest.mark.parametrize(
    "r, x",
    [(50.0, 0.0), (0.0, 30.0), (0.0, -30.0), (3.0, 4.0), (3.0, -4.0), (0.5, 120.0), (1e3, -1e3)],
)
def test_noiseless_roundtrip(r, x):
    m = physics.ideal_measurements(100.0, r, x)
    est = estimate_impedance_direct(m.v_rms_v, m.i_rms_a, m.phase_rad)
    assert est.r_ohm == pytest.approx(r, abs=1e-9)
    assert est.x_ohm == pytest.approx(x, abs=1e-9)
    assert est.z_mag_ohm == pytest.approx(np.hypot(r, x))
    assert est.z_complex == pytest.approx(complex(r, x), abs=1e-9)


def test_known_values():
    # |Z| = 100/20 = 5, phi = 60 deg -> R = 2.5, X = 5*sin(60 deg)
    est = estimate_impedance_direct(100.0, 20.0, np.deg2rad(60.0))
    assert est.r_ohm == pytest.approx(2.5)
    assert est.x_ohm == pytest.approx(5 * np.sqrt(3) / 2)


def test_vectorised_round_trip():
    r = np.linspace(1, 100, 50)
    x = np.linspace(-80, 80, 50)
    m = physics.ideal_measurements(70.0, r, x)
    est = estimate_impedance_direct(m.v_rms_v, m.i_rms_a, m.phase_rad)
    np.testing.assert_allclose(est.r_ohm, r, atol=1e-9)
    np.testing.assert_allclose(est.x_ohm, x, atol=1e-9)


@pytest.mark.parametrize(
    "v, i, phi",
    [(10.0, 0.0, 0.1), (10.0, 1e-15, 0.1), (10.0, -1.0, 0.1), (0.0, 1.0, 0.1),
     (np.nan, 1.0, 0.1), (10.0, np.inf, 0.1), (10.0, 1.0, np.nan)],
)
def test_invalid_samples_raise_by_default(v, i, phi):
    with pytest.raises(ValueError):
        estimate_impedance_direct(v, i, phi)


def test_invalid_samples_become_nan_when_requested():
    v = np.array([10.0, 10.0, 10.0])
    i = np.array([2.0, 0.0, 5.0])
    phi = np.array([0.0, 0.0, 0.0])
    est = estimate_impedance_direct(v, i, phi, invalid="nan")
    assert est.r_ohm[0] == pytest.approx(5.0)
    assert np.isnan(est.r_ohm[1]) and np.isnan(est.x_ohm[1])
    assert est.r_ohm[2] == pytest.approx(2.0)


def test_bad_options_raise():
    with pytest.raises(ValueError):
        estimate_impedance_direct(1.0, 1.0, 0.0, invalid="ignore")
    with pytest.raises(ValueError):
        estimate_impedance_direct(1.0, 1.0, 0.0, min_current_a=0.0)


def test_negative_resistance_is_not_clipped():
    # a noisy phase beyond +/-90 degrees gives R < 0; the baseline must report it honestly
    est = estimate_impedance_direct(10.0, 1.0, np.deg2rad(100.0))
    assert est.r_ohm < 0
