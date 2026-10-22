import json

import pytest

from src.anomaly_detection import AnomalyConfig
from src.kalman_filter import KalmanConfig
from src.measurements import NoiseConfig
from src.pipeline import compare_over_seeds, run_experiment
from src.report import (
    build_summary_report, experiment_config_dict, experiment_config_json,
    format_optional, metrics_table, results_csv,
)
from src.simulator import SimulationConfig


def _run(scenario="step", **kw):
    return run_experiment(SimulationConfig(scenario=scenario, seed=1, **kw))


def test_format_optional():
    assert format_optional(None) == "undefined"
    assert format_optional(0.12345) == "0.123"
    assert format_optional(0.5, ".1%") == "50.0%"


def test_metrics_table_shape_and_undefined_values():
    table = metrics_table(_run("stable"))
    assert list(table.columns) == ["Direct", "Kalman"]
    assert table.loc["Recall (samples)", "Direct"] == "undefined"      # no true disturbance
    assert table.loc["Mean detection delay", "Kalman"] == "undefined"
    step = metrics_table(_run("step"))
    assert step.loc["Events detected", "Direct"] == "1 / 1"
    assert "samples" in step.loc["Mean detection delay", "Direct"]


def test_config_export_roundtrip():
    sim = SimulationConfig(scenario="drift", seed=9, noise=NoiseConfig.preset("medium"))
    k, a = KalmanConfig(process_std_ohm=0.1), AnomalyConfig(k_sigma=4.0, z_max_ohm=60.0)
    data = json.loads(experiment_config_json(sim, k, a))
    assert SimulationConfig.from_dict(data["simulation"]) == sim
    assert KalmanConfig(**data["kalman"]) == k
    assert AnomalyConfig(**data["anomaly_detection"]) == a
    assert experiment_config_dict(sim, k, a)["simulation"]["seed"] == 9


def test_csv_export_has_one_row_per_sample():
    res = _run(n_steps=120)
    lines = results_csv(res).strip().splitlines()
    assert len(lines) == 121
    assert "r_kalman_ohm" in lines[0]


def test_summary_report_contains_key_sections_and_disclaimer():
    res = _run()
    text = build_summary_report(res, KalmanConfig(), AnomalyConfig())
    for fragment in ("# PlasmaTwin", "Educational simulation", "## Configuration",
                     "## Estimation accuracy", "13.560 MHz", "step", "RMSE of R"):
        assert fragment in text


def test_compare_over_seeds():
    df = compare_over_seeds(SimulationConfig(scenario="stable"), range(4))
    assert list(df["seed"]) == [0, 1, 2, 3]
    assert df["kalman_better"].all()
    with pytest.raises(ValueError):
        compare_over_seeds(SimulationConfig(), [])
