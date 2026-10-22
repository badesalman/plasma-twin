import pytest

go = pytest.importorskip("plotly.graph_objects")

from src import visualization as viz
from src.pipeline import run_experiment
from src.simulator import SimulationConfig


def _result():
    return run_experiment(SimulationConfig(scenario="intermittent", seed=2, n_steps=300))


def test_time_series_figures_have_three_traces_and_shading():
    df = _result().to_dataframe()
    for builder in (viz.plot_resistance, viz.plot_reactance,
                    viz.plot_impedance_magnitude, viz.plot_phase):
        fig = builder(df)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 3
        assert len(fig.layout.shapes) == 3            # one band per true disturbance


def test_other_figures_build():
    res = _result()
    df = res.to_dataframe()
    assert len(viz.plot_measurements(df).data) == 2
    assert len(viz.plot_anomaly_timeline(df).data) == 3
    assert len(viz.plot_error_over_time(df).data) == 2
    assert len(viz.plot_rmse_comparison(res).data) == 2
