"""Experiment 1 - noiseless validation of the direct estimator.

Sweeps a grid of frequencies and physical R-L-C values, generates ideal
measurements, runs the direct estimator and reports the reconstruction error.
Run from the repository root:  python -m experiments.exp1_noiseless_validation
"""

from __future__ import annotations

import itertools

import numpy as np

from src import physics
from src.estimators import estimate_impedance_direct

FREQUENCIES_HZ = [1.0e6, 13.56e6, 27.12e6, 40.68e6]
RESISTANCES_OHM = [0.5, 5.0, 50.0, 500.0]
INDUCTANCES_H = [0.1e-6, 1e-6, 10e-6]
CAPACITANCES_F = [10e-12, 100e-12, 1000e-12]
V_RMS_V = 100.0


def main() -> None:
    grid = np.array(list(itertools.product(FREQUENCIES_HZ, RESISTANCES_OHM, INDUCTANCES_H, CAPACITANCES_F)))
    f, r, l, c = grid.T
    x_true = physics.reactance(f, l, c)
    m = physics.ideal_measurements(V_RMS_V, r, x_true)
    est = estimate_impedance_direct(m.v_rms_v, m.i_rms_a, m.phase_rad)

    abs_err_r = np.abs(est.r_ohm - r)
    abs_err_x = np.abs(est.x_ohm - x_true)
    z_true = np.hypot(r, x_true)
    rel_err_z = np.abs(est.z_complex - (r + 1j * x_true)) / z_true

    print(f"Grid size: {len(grid)} operating points")
    print(f"Max |R_est - R_true| : {abs_err_r.max():.3e} ohm")
    print(f"Max |X_est - X_true| : {abs_err_x.max():.3e} ohm")
    print(f"Max relative |Z| error: {rel_err_z.max():.3e}")
    print(f"|X_true| range: {np.abs(x_true).min():.3f} .. {np.abs(x_true).max():.1f} ohm")


if __name__ == "__main__":
    main()
