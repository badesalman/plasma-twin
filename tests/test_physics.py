import numpy as np
import pytest

from src import physics

W = 1.0e7                      # rad/s, chosen so hand calculations are exact
F = W / (2 * np.pi)


def test_reactance_hand_calculation():
    # wL = 1e7 * 10e-6 = 100 ohm ; 1/(wC) = 1/(1e7 * 2e-9) = 50 ohm  ->  X = +50
    assert physics.reactance(F, 10e-6, 2e-9) == pytest.approx(50.0)


def test_reactance_zero_at_resonance():
    # wL = 100 ohm, 1/(wC) = 100 ohm
    assert physics.reactance(F, 10e-6, 1e-9) == pytest.approx(0.0, abs=1e-9)
    f0 = physics.resonant_frequency(10e-6, 1e-9)
    assert f0 == pytest.approx(F)


def test_rlc_impedance_complex_value():
    z = physics.rlc_impedance(F, 3.0, 10e-6, 2e-9)
    assert z == pytest.approx(3.0 + 50.0j)


def test_purely_resistive_case():
    mag = physics.impedance_magnitude(25.0, 0.0)
    assert mag == pytest.approx(25.0)
    assert physics.phase_angle(25.0, 0.0) == pytest.approx(0.0)


def test_purely_inductive_case():
    assert physics.phase_angle(0.0, 40.0) == pytest.approx(np.pi / 2)
    assert physics.reactance(F, 10e-6, 1e-3) > 0   # huge C -> inductive


def test_purely_capacitive_case():
    assert physics.phase_angle(0.0, -40.0) == pytest.approx(-np.pi / 2)
    assert physics.reactance(F, 0.0, 1e-9) == pytest.approx(-100.0)  # L = 0


def test_3_4_5_triangle_and_ohms_law():
    m = physics.ideal_measurements(100.0, 3.0, 4.0)
    assert m.i_rms_a == pytest.approx(20.0)                 # 100 V / 5 ohm
    assert m.phase_rad == pytest.approx(np.arctan2(4.0, 3.0))
    assert m.v_rms_v == pytest.approx(100.0)
    assert m.v_rms_v / m.i_rms_a == pytest.approx(physics.impedance_magnitude(3.0, 4.0))


def test_array_broadcasting():
    m = physics.ideal_measurements(50.0, np.array([1.0, 2.0, 3.0]), 4.0)
    assert m.i_rms_a.shape == (3,)
    assert m.v_rms_v.shape == (3,)


@pytest.mark.parametrize(
    "call",
    [
        lambda: physics.reactance(0.0, 1e-6, 1e-10),
        lambda: physics.reactance(-1.0, 1e-6, 1e-10),
        lambda: physics.reactance(1e6, -1e-6, 1e-10),
        lambda: physics.reactance(1e6, 1e-6, 0.0),
        lambda: physics.reactance(np.nan, 1e-6, 1e-10),
        lambda: physics.rlc_impedance(1e6, -1.0, 1e-6, 1e-10),
        lambda: physics.ideal_measurements(0.0, 1.0, 1.0),
        lambda: physics.ideal_measurements(10.0, -1.0, 1.0),
        lambda: physics.ideal_measurements(10.0, 0.0, 0.0),
        lambda: physics.ideal_measurements(10.0, np.inf, 1.0),
        lambda: physics.resonant_frequency(0.0, 1e-9),
    ],
)
def test_invalid_inputs_raise(call):
    with pytest.raises(ValueError):
        call()
