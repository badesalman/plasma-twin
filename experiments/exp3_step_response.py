"""Experiment 3 - response to a sudden impedance step and anomaly detection.

For each noise level, N_SEEDS seeds of the 'step' scenario:
  * response delay: samples after the step until the estimate stays within 50 % (and 20 %)
    of the step size around the new impedance for 3 consecutive samples;
  * detection delay of the anomaly detector run on the direct and on the filtered estimate;
  * false-alarm events (alarm segments not overlapping the true disturbance).
Also: false alarms on a stable load and on the 'noisy' scenario (no true disturbance).

Run from the repository root:  python -m experiments.exp3_step_response
"""

from __future__ import annotations

import numpy as np

from src.measurements import NoiseConfig
from src.metrics import settling_delay
from src.pipeline import run_experiment
from src.simulator import Scenario, SimulationConfig

N_SEEDS = 50
NOISE_LEVELS = ["low", "medium", "high"]


def _fmt(values: list[int | None]) -> str:
    found = [v for v in values if v is not None]
    if not found:
        return "never"
    return f"{np.mean(found):5.1f} ({len(found)}/{len(values)})"


def step_response() -> None:
    print(f"Step response, {N_SEEDS} seeds per noise level (mean samples, (runs that got there/total))")
    header = f"{'noise':<8}{'estimator':<10}{'within 50%':>16}{'within 20%':>16}{'detected':>12}{'detect delay':>14}{'false alarms':>14}"
    print(header)
    for level in NOISE_LEVELS:
        acc = {"direct": {"r50": [], "r20": [], "ev": []}, "kalman": {"r50": [], "r20": [], "ev": []}}
        for seed in range(N_SEEDS):
            cfg = SimulationConfig(scenario=Scenario.STEP, noise=NoiseConfig.preset(level), seed=seed)
            res = run_experiment(cfg)
            sim = res.simulation
            k = cfg.step_index
            z_before = complex(sim.r_true_ohm[k - 1], sim.x_true_ohm[k - 1])
            z_after = complex(sim.r_true_ohm[k], sim.x_true_ohm[k])
            for name, est, ev in (("direct", res.direct, res.events_direct),
                                  ("kalman", res.kalman, res.events_kalman)):
                acc[name]["r50"].append(settling_delay(est.r_ohm, est.x_ohm, z_before, z_after, k, 0.5))
                acc[name]["r20"].append(settling_delay(est.r_ohm, est.x_ohm, z_before, z_after, k, 0.2))
                acc[name]["ev"].append(ev)
        for name in ("direct", "kalman"):
            events = acc[name]["ev"]
            detected = sum(e.n_detected_events for e in events)
            delays = [d for e in events for d in e.detection_delays_steps]
            false_alarms = sum(e.n_false_alarm_events for e in events)
            print(f"{level:<8}{name:<10}{_fmt(acc[name]['r50']):>16}{_fmt(acc[name]['r20']):>16}"
                  f"{detected:>9}/{N_SEEDS}{(np.mean(delays) if delays else float('nan')):>14.1f}{false_alarms:>14}")
        print()


def false_alarms() -> None:
    print(f"False alarms on scenarios with NO true disturbance ({N_SEEDS} seeds each)")
    print(f"{'scenario':<9}{'noise':<8}{'estimator':<10}{'runs with >=1 false alarm':>28}")
    for scenario in (Scenario.STABLE, Scenario.NOISY):
        for level in NOISE_LEVELS:
            counts = {"direct": 0, "kalman": 0}
            for seed in range(N_SEEDS):
                res = run_experiment(SimulationConfig(scenario=scenario, noise=NoiseConfig.preset(level), seed=seed))
                counts["direct"] += int(res.anomaly_direct.alarm.any())
                counts["kalman"] += int(res.anomaly_kalman.alarm.any())
            for name in ("direct", "kalman"):
                print(f"{scenario.value:<9}{level:<8}{name:<10}{counts[name]:>22}/{N_SEEDS}")


if __name__ == "__main__":
    step_response()
    false_alarms()
