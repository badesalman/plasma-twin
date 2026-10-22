"""End-to-end run: simulate -> direct estimate -> Kalman filter -> detect -> evaluate.

Keeps this orchestration out of the UI and out of the experiment scripts.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from src.anomaly_detection import AnomalyConfig, AnomalyResult, detect_anomalies
from src.estimators import ImpedanceEstimate, estimate_impedance_direct
from src.kalman_filter import KalmanConfig, KalmanResult, run_kalman
from src.measurements import NoiseConfig
from src.metrics import (
    EstimationMetrics,
    EventDetectionMetrics,
    SampleDetectionMetrics,
    estimation_metrics,
    event_detection_metrics,
    sample_detection_metrics,
)
from src.simulator import SimulationConfig, SimulationResult, simulate


@dataclass(frozen=True)
class ExperimentResult:
    """Everything produced by one run."""

    simulation: SimulationResult
    direct: ImpedanceEstimate
    kalman_raw: KalmanResult
    kalman: ImpedanceEstimate
    anomaly_direct: AnomalyResult
    anomaly_kalman: AnomalyResult
    metrics_direct: EstimationMetrics
    metrics_kalman: EstimationMetrics
    detection_direct: SampleDetectionMetrics
    detection_kalman: SampleDetectionMetrics
    events_direct: EventDetectionMetrics
    events_kalman: EventDetectionMetrics

    def to_dataframe(self) -> pd.DataFrame:
        """One row per sample: ground truth, measurements, both estimates and alarms."""
        df = self.simulation.to_dataframe()
        for name, est in (("direct", self.direct), ("kalman", self.kalman)):
            df[f"r_{name}_ohm"] = est.r_ohm
            df[f"x_{name}_ohm"] = est.x_ohm
            df[f"z_mag_{name}_ohm"] = est.z_mag_ohm
            df[f"phase_{name}_deg"] = np.rad2deg(est.phase_rad)
        df["alarm_direct"] = self.anomaly_direct.alarm
        df["alarm_kalman"] = self.anomaly_kalman.alarm
        return df


def run_experiment(
    sim_config: SimulationConfig,
    kalman_config: KalmanConfig | None = None,
    anomaly_config: AnomalyConfig | None = None,
    assumed_noise: NoiseConfig | None = None,
    grace_steps: int = 10,
) -> ExperimentResult:
    """Run the full pipeline.

    ``assumed_noise`` is the noise the Kalman filter believes in; by default it
    equals the simulated noise (a "matched" filter).
    """
    sim = simulate(sim_config)
    direct = estimate_impedance_direct(
        sim.v_meas_v, sim.i_meas_a, sim.phase_meas_rad, invalid="nan"
    )
    kal_raw = run_kalman(direct, assumed_noise or sim_config.noise, kalman_config or KalmanConfig())
    kal = kal_raw.as_estimate()

    acfg = anomaly_config or AnomalyConfig(baseline_steps=sim_config.baseline_steps)
    det_d = detect_anomalies(direct.r_ohm, direct.x_ohm, acfg)
    det_k = detect_anomalies(kal.r_ohm, kal.x_ohm, acfg)

    return ExperimentResult(
        simulation=sim,
        direct=direct,
        kalman_raw=kal_raw,
        kalman=kal,
        anomaly_direct=det_d,
        anomaly_kalman=det_k,
        metrics_direct=estimation_metrics(sim.r_true_ohm, sim.x_true_ohm, direct.r_ohm, direct.x_ohm),
        metrics_kalman=estimation_metrics(sim.r_true_ohm, sim.x_true_ohm, kal.r_ohm, kal.x_ohm),
        detection_direct=sample_detection_metrics(sim.anomaly_true, det_d.alarm),
        detection_kalman=sample_detection_metrics(sim.anomaly_true, det_k.alarm),
        events_direct=event_detection_metrics(sim.anomaly_true, det_d.alarm, grace_steps),
        events_kalman=event_detection_metrics(sim.anomaly_true, det_k.alarm, grace_steps),
    )


def compare_over_seeds(
    sim_config: SimulationConfig,
    seeds: Iterable[int],
    kalman_config: KalmanConfig | None = None,
    anomaly_config: AnomalyConfig | None = None,
    assumed_noise: NoiseConfig | None = None,
) -> pd.DataFrame:
    """Run the same configuration for several seeds and tabulate the complex-impedance RMSE.

    A single seed can be misleading (the filter may win or lose by chance), so this
    gives the distribution of the direct-vs-Kalman comparison for one configuration.
    """
    seed_list = [int(s) for s in seeds]
    if not seed_list:
        raise ValueError("at least one seed is required")
    rows = []
    for seed in seed_list:
        res = run_experiment(
            replace(sim_config, seed=seed), kalman_config, anomaly_config, assumed_noise
        )
        d, k = res.metrics_direct.rmse_z_ohm, res.metrics_kalman.rmse_z_ohm
        rows.append(
            {"seed": seed, "rmse_z_direct_ohm": d, "rmse_z_kalman_ohm": k, "kalman_better": bool(k < d)}
        )
    return pd.DataFrame(rows)
