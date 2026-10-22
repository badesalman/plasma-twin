"""Experiment 2 - direct vs Kalman-filtered estimation over many random seeds.

Part A: every scenario x noise level, N_SEEDS seeds each. Reports the mean RMSE of
        the complex impedance |Z_est - Z_true| for both estimators and the share of
        seeds in which the filter is better.
Part B: effect of the filter's process noise q on a stable load and on a step.

Run from the repository root:  python -m experiments.exp2_noise_reduction
"""

from __future__ import annotations

import numpy as np

from src.kalman_filter import KalmanConfig
from src.measurements import NoiseConfig
from src.pipeline import run_experiment
from src.simulator import Scenario, SimulationConfig

N_SEEDS = 50
SCENARIOS = [Scenario.STABLE, Scenario.DRIFT, Scenario.STEP, Scenario.INTERMITTENT, Scenario.NOISY]
NOISE_LEVELS = ["low", "medium", "high"]
Q_VALUES = [0.005, 0.02, 0.05, 0.2, 1.0]


def part_a() -> None:
    print(f"Part A - {N_SEEDS} seeds per row, default filter (q = {KalmanConfig().process_std_ohm} ohm/step)")
    print(f"{'scenario':<13}{'noise':<8}{'RMSE_Z direct':>15}{'RMSE_Z Kalman':>15}{'Kalman better':>15}")
    for scenario in SCENARIOS:
        for level in NOISE_LEVELS:
            d, k = [], []
            for seed in range(N_SEEDS):
                cfg = SimulationConfig(scenario=scenario, noise=NoiseConfig.preset(level), seed=seed)
                res = run_experiment(cfg)
                d.append(res.metrics_direct.rmse_z_ohm)
                k.append(res.metrics_kalman.rmse_z_ohm)
            d, k = np.array(d), np.array(k)
            print(f"{scenario.value:<13}{level:<8}{d.mean():>12.3f} ohm{k.mean():>12.3f} ohm{100 * np.mean(k < d):>14.0f}%")
        print()


def part_b() -> None:
    print(f"Part B - process noise q sweep, medium noise, {N_SEEDS} seeds (mean RMSE_Z in ohm)")
    print(f"{'q [ohm/step]':<14}{'stable':>10}{'step':>10}")
    base = {}
    for scenario in (Scenario.STABLE, Scenario.STEP):
        base[scenario] = np.mean([
            run_experiment(SimulationConfig(scenario=scenario, noise=NoiseConfig.preset("medium"), seed=s))
            .metrics_direct.rmse_z_ohm for s in range(N_SEEDS)])
    print(f"{'direct':<14}{base[Scenario.STABLE]:>10.3f}{base[Scenario.STEP]:>10.3f}")
    for q in Q_VALUES:
        row = []
        for scenario in (Scenario.STABLE, Scenario.STEP):
            row.append(np.mean([
                run_experiment(
                    SimulationConfig(scenario=scenario, noise=NoiseConfig.preset("medium"), seed=s),
                    kalman_config=KalmanConfig(process_std_ohm=q),
                ).metrics_kalman.rmse_z_ohm for s in range(N_SEEDS)]))
        print(f"{q:<14}{row[0]:>10.3f}{row[1]:>10.3f}")


if __name__ == "__main__":
    part_a()
    part_b()
