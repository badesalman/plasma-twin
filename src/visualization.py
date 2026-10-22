"""Plotly figures for the monitoring dashboard.

All functions take the per-sample table from ``ExperimentResult.to_dataframe()``
(or the result itself) and return a ``plotly.graph_objects.Figure``. They contain
no Streamlit code, so they can be reused in notebooks or scripts.

Colour convention (same in every chart): green = true simulated value,
light red = direct estimate, blue = Kalman estimate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.anomaly_detection import find_segments
from src.pipeline import ExperimentResult

COLORS = {
    "true": "#2e7d32",
    "direct": "#e57373",
    "kalman": "#1565c0",
    "voltage": "#6a1b9a",
    "current": "#ef6c00",
}
NAMES = {
    "true": "True (simulated)",
    "direct": "Direct estimate",
    "kalman": "Kalman estimate",
}


def _style(fig: go.Figure, title: str, y_title: str, height: int = 340) -> go.Figure:
    fig.update_layout(
        title=title,
        xaxis_title="Time (s)",
        yaxis_title=y_title,
        height=height,
        margin=dict(l=10, r=10, t=70, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )
    return fig


def _shade_true_disturbances(fig: go.Figure, df: pd.DataFrame) -> None:
    """Grey bands where the simulator's ground truth says the load is disturbed."""
    time = df["time_s"].to_numpy()
    for start, end in find_segments(df["anomaly_true"].to_numpy()):
        fig.add_vrect(
            x0=time[start],
            x1=time[end - 1],
            fillcolor="rgba(120, 120, 120, 0.15)",
            line_width=0,
            layer="below",
        )


def _three_series(
    df: pd.DataFrame, columns: dict[str, str], title: str, y_title: str, shade: bool = True
) -> go.Figure:
    fig = go.Figure()
    widths = {"true": 2.5, "direct": 1.0, "kalman": 2.0}
    for key in ("true", "direct", "kalman"):
        fig.add_trace(
            go.Scatter(
                x=df["time_s"],
                y=df[columns[key]],
                mode="lines",
                name=NAMES[key],
                line=dict(color=COLORS[key], width=widths[key]),
            )
        )
    if shade:
        _shade_true_disturbances(fig, df)
    return _style(fig, title, y_title)


def plot_resistance(df: pd.DataFrame) -> go.Figure:
    return _three_series(
        df,
        {"true": "r_true_ohm", "direct": "r_direct_ohm", "kalman": "r_kalman_ohm"},
        "Equivalent resistance R", "R (ohm)",
    )


def plot_reactance(df: pd.DataFrame) -> go.Figure:
    return _three_series(
        df,
        {"true": "x_true_ohm", "direct": "x_direct_ohm", "kalman": "x_kalman_ohm"},
        "Equivalent reactance X", "X (ohm)",
    )


def plot_impedance_magnitude(df: pd.DataFrame) -> go.Figure:
    return _three_series(
        df,
        {"true": "z_mag_true_ohm", "direct": "z_mag_direct_ohm", "kalman": "z_mag_kalman_ohm"},
        "Impedance magnitude |Z|", "|Z| (ohm)",
    )


def plot_phase(df: pd.DataFrame) -> go.Figure:
    return _three_series(
        df,
        {"true": "phase_true_deg", "direct": "phase_direct_deg", "kalman": "phase_kalman_deg"},
        "Voltage-current phase angle", "Phase (deg)",
    )


def plot_measurements(df: pd.DataFrame) -> go.Figure:
    """Noisy RMS voltage (left axis) and RMS current (right axis)."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(x=df["time_s"], y=df["v_meas_v"], mode="lines", name="Voltage (measured)",
                   line=dict(color=COLORS["voltage"], width=1.2)),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=df["time_s"], y=df["i_meas_a"], mode="lines", name="Current (measured)",
                   line=dict(color=COLORS["current"], width=1.2)),
        secondary_y=True,
    )
    _style(fig, "Measured RMS voltage and current", "")
    fig.update_yaxes(title_text="V (V rms)", secondary_y=False)
    fig.update_yaxes(title_text="I (A rms)", secondary_y=True)
    return fig


def plot_anomaly_timeline(df: pd.DataFrame) -> go.Figure:
    """Three lanes: ground-truth disturbance, direct-estimate alarm, Kalman-estimate alarm."""
    lanes = [
        ("Kalman alarm", "alarm_kalman", 0, COLORS["kalman"]),
        ("Direct alarm", "alarm_direct", 1, COLORS["direct"]),
        ("True disturbance", "anomaly_true", 2, COLORS["true"]),
    ]
    fig = go.Figure()
    for name, column, lane, color in lanes:
        fig.add_trace(
            go.Scatter(
                x=df["time_s"],
                y=df[column].to_numpy(dtype=float) * 0.8 + lane,
                mode="lines",
                name=name,
                line=dict(color=color, width=2, shape="hv"),
                hovertemplate=name + "<extra></extra>",
            )
        )
    _style(fig, "Detected anomalies vs true disturbances", "")
    fig.update_layout(hovermode="closest")
    fig.update_yaxes(
        tickvals=[0.4, 1.4, 2.4],
        ticktext=[lane[0] for lane in lanes],
        range=[-0.2, 3.0],
    )
    return fig


def plot_error_over_time(df: pd.DataFrame) -> go.Figure:
    """Absolute complex-impedance error |Z_est - Z_true| per sample."""
    fig = go.Figure()
    for key in ("direct", "kalman"):
        error = np.hypot(
            df[f"r_{key}_ohm"] - df["r_true_ohm"], df[f"x_{key}_ohm"] - df["x_true_ohm"]
        )
        fig.add_trace(
            go.Scatter(x=df["time_s"], y=error, mode="lines", name=NAMES[key],
                       line=dict(color=COLORS[key], width=1.2 if key == "direct" else 2))
        )
    _shade_true_disturbances(fig, df)
    return _style(fig, "Estimation error over time", "|Z_est - Z_true| (ohm)")


def plot_rmse_comparison(result: ExperimentResult) -> go.Figure:
    """Bar chart of RMSE for R, X and Z, direct vs Kalman."""
    md, mk = result.metrics_direct, result.metrics_kalman
    categories = ["R", "X", "Z (complex)"]
    fig = go.Figure()
    fig.add_trace(go.Bar(name=NAMES["direct"], x=categories, marker_color=COLORS["direct"],
                         y=[md.rmse_r_ohm, md.rmse_x_ohm, md.rmse_z_ohm]))
    fig.add_trace(go.Bar(name=NAMES["kalman"], x=categories, marker_color=COLORS["kalman"],
                         y=[mk.rmse_r_ohm, mk.rmse_x_ohm, mk.rmse_z_ohm]))
    _style(fig, "RMSE: direct vs Kalman", "RMSE (ohm)", height=320)
    fig.update_layout(barmode="group", xaxis_title="Quantity", hovermode="closest")
    return fig
