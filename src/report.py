"""Tables, configuration export and the short summary report (no UI code here)."""

from __future__ import annotations

import json
from dataclasses import asdict

import numpy as np
import pandas as pd

from src import physics
from src.anomaly_detection import AnomalyConfig
from src.kalman_filter import KalmanConfig
from src.pipeline import ExperimentResult
from src.simulator import SimulationConfig

UNDEFINED = "undefined"


def format_optional(value: float | None, fmt: str = ".3f", suffix: str = "") -> str:
    """Format a number, or return 'undefined' for None (metric not defined)."""
    return UNDEFINED if value is None else f"{value:{fmt}}{suffix}"


def _delay_text(mean_steps: float | None, dt_s: float) -> str:
    if mean_steps is None:
        return UNDEFINED
    return f"{mean_steps:.1f} samples ({mean_steps * dt_s * 1000:.0f} ms)"


def metrics_table(result: ExperimentResult) -> pd.DataFrame:
    """Direct vs Kalman accuracy and detection metrics as a display table."""
    md, mk = result.metrics_direct, result.metrics_kalman
    sd, sk = result.detection_direct, result.detection_kalman
    ed, ek = result.events_direct, result.events_kalman
    dt = result.simulation.config.dt_s
    rows = [
        ("RMSE of R (ohm)", format_optional(md.rmse_r_ohm), format_optional(mk.rmse_r_ohm)),
        ("RMSE of X (ohm)", format_optional(md.rmse_x_ohm), format_optional(mk.rmse_x_ohm)),
        ("RMSE of Z, |Z_est - Z_true| (ohm)", format_optional(md.rmse_z_ohm), format_optional(mk.rmse_z_ohm)),
        ("MAE of R (ohm)", format_optional(md.mae_r_ohm), format_optional(mk.mae_r_ohm)),
        ("MAE of X (ohm)", format_optional(md.mae_x_ohm), format_optional(mk.mae_x_ohm)),
        ("Relative RMSE of R", format_optional(md.relative_rmse_r, ".2%"), format_optional(mk.relative_rmse_r, ".2%")),
        ("Relative RMSE of X", format_optional(md.relative_rmse_x, ".2%"), format_optional(mk.relative_rmse_x, ".2%")),
        ("Precision (samples)", format_optional(sd.precision), format_optional(sk.precision)),
        ("Recall (samples)", format_optional(sd.recall), format_optional(sk.recall)),
        ("F1 (samples)", format_optional(sd.f1), format_optional(sk.f1)),
        ("True disturbance events", str(ed.n_true_events), str(ek.n_true_events)),
        ("Events detected", f"{ed.n_detected_events} / {ed.n_true_events}", f"{ek.n_detected_events} / {ek.n_true_events}"),
        ("Mean detection delay", _delay_text(ed.mean_delay_steps, dt), _delay_text(ek.mean_delay_steps, dt)),
        ("False-alarm events", str(ed.n_false_alarm_events), str(ek.n_false_alarm_events)),
    ]
    return pd.DataFrame(rows, columns=["Metric", "Direct", "Kalman"]).set_index("Metric")


def experiment_config_dict(
    sim: SimulationConfig, kalman: KalmanConfig, anomaly: AnomalyConfig
) -> dict:
    """All settings needed to reproduce a run."""
    return {
        "simulation": sim.to_dict(),
        "kalman": asdict(kalman),
        "anomaly_detection": asdict(anomaly),
    }


def experiment_config_json(sim: SimulationConfig, kalman: KalmanConfig, anomaly: AnomalyConfig) -> str:
    return json.dumps(experiment_config_dict(sim, kalman, anomaly), indent=2)


def results_csv(result: ExperimentResult) -> str:
    """Per-sample results as CSV text."""
    return result.to_dataframe().to_csv(index=False)


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def build_summary_report(
    result: ExperimentResult, kalman: KalmanConfig, anomaly: AnomalyConfig
) -> str:
    """Short Markdown report of one run."""
    cfg = result.simulation.config
    x0 = float(physics.reactance(cfg.frequency_hz, cfg.inductance_h, cfg.capacitance_f))
    z0 = float(np.hypot(cfg.resistance_ohm, x0))
    f0 = (
        f"{float(physics.resonant_frequency(cfg.inductance_h, cfg.capacitance_f)) / 1e6:.3f} MHz"
        if cfg.inductance_h > 0
        else "undefined (L = 0)"
    )
    table = metrics_table(result)
    acc_rows = [[name, row["Direct"], row["Kalman"]] for name, row in table.iterrows()]
    n_alarm_d = len(result.anomaly_direct.events)
    n_alarm_k = len(result.anomaly_kalman.events)

    cfg_rows = [
        ["Scenario", cfg.scenario.value],
        ["Frequency", f"{cfg.frequency_hz / 1e6:.3f} MHz"],
        ["RMS voltage at the load", f"{cfg.v_rms_v:g} V"],
        ["Initial R, L, C", f"{cfg.resistance_ohm:g} ohm, {cfg.inductance_h * 1e6:g} uH, {cfg.capacitance_f * 1e12:g} pF"],
        ["Initial X / |Z| / series resonance", f"{x0:.2f} ohm / {z0:.2f} ohm / {f0}"],
        ["Steps x sampling interval", f"{cfg.n_steps} x {cfg.dt_s * 1000:g} ms"],
        ["Noise (V, I relative; phase)", f"{cfg.noise.v_rel_std:.2%}, {cfg.noise.i_rel_std:.2%}; {np.rad2deg(cfg.noise.phase_std_rad):.2f} deg"],
        ["Random seed", str(cfg.seed)],
        ["Kalman process std q / measurement scale", f"{kalman.process_std_ohm:g} ohm per step / {kalman.measurement_noise_scale:g}"],
        ["Detector k_sigma / persistence / min relative change", f"{anomaly.k_sigma:g} / {anomaly.persistence} / {anomaly.min_relative_change:.1%}"],
    ]
    return "\n".join(
        [
            "# PlasmaTwin - experiment summary",
            "",
            "> Educational simulation of an equivalent RF load impedance. It does not estimate plasma "
            "density, temperature or process quality.",
            "",
            "## Configuration",
            _md_table(["Parameter", "Value"], cfg_rows),
            "",
            "## Estimation accuracy and detection",
            _md_table(["Metric", "Direct", "Kalman"], acc_rows),
            "",
            "## Alarm segments",
            f"- Direct estimate: {n_alarm_d} alarm segment(s)",
            f"- Kalman estimate: {n_alarm_k} alarm segment(s)",
            "",
            "## Notes",
            "- Results come from one simulated run (one random seed); they can differ for other seeds.",
            "- 'undefined' means the metric has no mathematical meaning for this run "
            "(for example recall when no true disturbance exists).",
            "- Ground-truth disturbance: true |Z| deviating from its initial value by more than "
            f"{cfg.label_tolerance:.0%}.",
            "",
        ]
    )
