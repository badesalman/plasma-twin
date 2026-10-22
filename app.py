"""PlasmaTwin - Streamlit dashboard.

Run from the repository root:   streamlit run app.py

This file only wires widgets to the numerical code in ``src/``; no estimation or
simulation logic lives here.
"""

from __future__ import annotations

import numpy as np
import streamlit as st

from src import physics
from src import visualization as viz
from src.anomaly_detection import AnomalyConfig
from src.kalman_filter import KalmanConfig
from src.measurements import NoiseConfig
from src.pipeline import compare_over_seeds, run_experiment
from src.report import (
    build_summary_report,
    experiment_config_json,
    metrics_table,
    results_csv,
)
from src.simulator import Scenario, SimulationConfig

st.set_page_config(page_title="PlasmaTwin", page_icon="⚡", layout="wide")

DEFAULT_SIM = SimulationConfig()
DEFAULT_KALMAN = KalmanConfig()
DEFAULT_ANOMALY = AnomalyConfig()

SCENARIO_LABELS = {
    Scenario.STABLE: "A - Stable operation",
    Scenario.DRIFT: "B - Gradual drift",
    Scenario.STEP: "C - Sudden impedance change",
    Scenario.NOISY: "D - Increasing measurement noise",
    Scenario.INTERMITTENT: "E - Intermittent disturbance",
}
SCENARIO_HELP = {
    Scenario.STABLE: "R and L stay constant. Any alarm here is a false alarm.",
    Scenario.DRIFT: "R and L change slowly and linearly after the baseline window.",
    Scenario.STEP: "R and L jump by a factor at a configurable moment.",
    Scenario.NOISY: "The true load is constant; the noise level ramps up. Any alarm is a false alarm.",
    Scenario.INTERMITTENT: "Several temporary R/L changes in otherwise stable operation.",
}


# --------------------------------------------------------------------------- sidebar
def build_configs() -> tuple[SimulationConfig, KalmanConfig, AnomalyConfig]:
    """Read all widgets and build validated configuration objects (SI units)."""
    with st.sidebar:
        st.header("Simulation controls")

        scenario = st.selectbox(
            "Operating scenario",
            options=list(Scenario),
            format_func=lambda s: SCENARIO_LABELS[s],
            index=0,
        )
        st.caption(SCENARIO_HELP[scenario])

        st.subheader("RF load (series R-L-C)")
        freq_mhz = st.number_input(
            "RF frequency (MHz)", min_value=0.1, max_value=200.0, value=13.56, step=0.01, format="%.2f"
        )
        v_rms = st.number_input(
            "RMS voltage at the load (V)", min_value=1.0, max_value=1000.0, value=100.0, step=1.0
        )
        r_ohm = st.number_input(
            "Initial resistance R (ohm)", min_value=0.0, max_value=1000.0, value=10.0, step=0.5
        )
        l_uh = st.number_input(
            "Inductance L (uH)", min_value=0.0, max_value=100.0, value=1.2, step=0.05, format="%.2f"
        )
        c_pf = st.number_input(
            "Capacitance C (pF)", min_value=1.0, max_value=10000.0, value=150.0, step=5.0
        )

        st.subheader("Time base")
        n_steps = st.slider("Number of simulation steps", min_value=100, max_value=3000, value=600, step=50)
        dt_ms = st.number_input(
            "Sampling interval (ms)", min_value=1.0, max_value=1000.0, value=10.0, step=1.0
        )
        baseline = st.slider(
            "Baseline window (steps, assumed normal)", min_value=10, max_value=200, value=50, step=5
        )

        st.subheader("Measurement noise")
        level = st.selectbox("Noise level", options=["off", "low", "medium", "high", "custom"], index=1)
        if level == "custom":
            v_pct = st.number_input("Voltage noise (% relative)", min_value=0.0, max_value=30.0, value=1.0, step=0.1)
            i_pct = st.number_input("Current noise (% relative)", min_value=0.0, max_value=30.0, value=1.0, step=0.1)
            phase_deg = st.number_input("Phase noise (deg)", min_value=0.0, max_value=28.0, value=0.5, step=0.1)
            noise = NoiseConfig(
                v_rel_std=v_pct / 100.0, i_rel_std=i_pct / 100.0, phase_std_rad=float(np.deg2rad(phase_deg))
            )
        else:
            noise = NoiseConfig.preset(level)
        seed = st.number_input("Random seed", min_value=0, max_value=2**31 - 1, value=0, step=1)

        with st.expander("Scenario parameters"):
            event_frac = st.slider("Step time (fraction of run)", min_value=0.1, max_value=0.9, value=0.5, step=0.05)
            event_r = st.number_input("Step / pulse R factor", min_value=0.1, max_value=5.0, value=1.5, step=0.05)
            event_l = st.number_input("Step / pulse L factor", min_value=0.5, max_value=2.0, value=1.04, step=0.01)
            drift_r = st.number_input("Drift of R over the run (fraction)", min_value=-0.9, max_value=3.0, value=0.5, step=0.05)
            drift_l = st.number_input("Drift of L over the run (fraction)", min_value=-0.5, max_value=1.0, value=0.05, step=0.01)
            ramp = st.number_input("Noise ramp factor (scenario D)", min_value=1.0, max_value=10.0, value=4.0, step=0.5)
            n_pulses = st.slider("Number of pulses (scenario E)", min_value=1, max_value=8, value=3)
            pulse_len = st.slider("Pulse length (steps)", min_value=5, max_value=100, value=30)
            tolerance = st.slider(
                "Ground-truth tolerance (relative |Z| change)", min_value=0.02, max_value=0.5, value=0.10, step=0.01
            )

        st.subheader("Kalman filter")
        q = st.number_input(
            "Process noise std q (ohm per step)", min_value=0.0, max_value=5.0,
            value=float(DEFAULT_KALMAN.process_std_ohm), step=0.01, format="%.3f",
        )
        r_scale = st.number_input(
            "Measurement covariance scale (1 = matched)", min_value=0.05, max_value=20.0, value=1.0, step=0.05
        )

        st.subheader("Anomaly detector")
        k_sigma = st.number_input(
            "Threshold k (x baseline std)", min_value=1.0, max_value=20.0,
            value=float(DEFAULT_ANOMALY.k_sigma), step=0.5,
        )
        persistence = st.slider("Consecutive samples required", min_value=1, max_value=10, value=DEFAULT_ANOMALY.persistence)
        floor_pct = st.number_input(
            "Minimum threshold (% of baseline |Z|)", min_value=0.0, max_value=20.0,
            value=DEFAULT_ANOMALY.min_relative_change * 100.0, step=0.5,
        )
        use_band = st.checkbox("Also monitor |Z| against an absolute band", value=False)
        z_min = z_max = None
        if use_band:
            z_min = st.number_input("Minimum allowed |Z| (ohm)", min_value=0.1, max_value=5000.0, value=15.0, step=1.0)
            z_max = st.number_input("Maximum allowed |Z| (ohm)", min_value=0.1, max_value=5000.0, value=40.0, step=1.0)

    sim = SimulationConfig(
        frequency_hz=freq_mhz * 1e6,
        v_rms_v=v_rms,
        resistance_ohm=r_ohm,
        inductance_h=l_uh * 1e-6,
        capacitance_f=c_pf * 1e-12,
        n_steps=int(n_steps),
        dt_s=dt_ms / 1000.0,
        scenario=scenario,
        noise=noise,
        seed=int(seed),
        baseline_steps=int(baseline),
        event_start_fraction=event_frac,
        event_r_factor=event_r,
        event_l_factor=event_l,
        drift_r_fraction=drift_r,
        drift_l_fraction=drift_l,
        noise_ramp_factor=ramp,
        n_pulses=int(n_pulses),
        pulse_steps=int(pulse_len),
        label_tolerance=tolerance,
    )
    kalman = KalmanConfig(process_std_ohm=q, measurement_noise_scale=r_scale)
    anomaly = AnomalyConfig(
        baseline_steps=int(baseline),
        k_sigma=k_sigma,
        min_relative_change=floor_pct / 100.0,
        persistence=int(persistence),
        z_min_ohm=z_min,
        z_max_ohm=z_max,
    )
    return sim, kalman, anomaly


# --------------------------------------------------------------------------- tabs
def tab_overview(result, sim: SimulationConfig) -> None:
    st.title("PlasmaTwin")
    st.markdown(
        "**Virtual RF load monitoring and impedance estimation.** A simulated series R-L-C load changes "
        "over time; noisy voltage, current and phase measurements are turned into estimates of the "
        "equivalent impedance Z = R + jX, tracked with a Kalman filter, and monitored for abnormal changes."
    )
    st.info(
        "Educational simulation of an electrical load. It does not estimate plasma density, plasma "
        "temperature or process quality - that would need validated physical models and real measurements."
    )

    x0 = float(physics.reactance(sim.frequency_hz, sim.inductance_h, sim.capacitance_f))
    z0 = float(np.hypot(sim.resistance_ohm, x0))
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("RF frequency", f"{sim.frequency_hz / 1e6:.2f} MHz")
    c2.metric("Initial reactance X", f"{x0:.2f} ohm")
    c3.metric("Initial |Z|", f"{z0:.2f} ohm")
    if sim.inductance_h > 0:
        c4.metric("Series resonance", f"{float(physics.resonant_frequency(sim.inductance_h, sim.capacitance_f)) / 1e6:.2f} MHz")
    else:
        c4.metric("Series resonance", "n/a (L = 0)")

    st.subheader("Current estimate (last sample, Kalman filter)")
    k, d, s = result.kalman, result.direct, result.simulation
    last = -1
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Resistance R", f"{k.r_ohm[last]:.2f} ohm",
              delta=f"{k.r_ohm[last] - s.r_true_ohm[last]:+.2f} vs true", delta_color="off")
    m2.metric("Reactance X", f"{k.x_ohm[last]:.2f} ohm",
              delta=f"{k.x_ohm[last] - s.x_true_ohm[last]:+.2f} vs true", delta_color="off")
    m3.metric("Impedance |Z|", f"{k.z_mag_ohm[last]:.2f} ohm",
              delta=f"{k.z_mag_ohm[last] - s.z_mag_true_ohm[last]:+.2f} vs true", delta_color="off")
    m4.metric("Phase angle", f"{np.rad2deg(k.phase_rad[last]):.1f} deg",
              delta=f"{np.rad2deg(k.phase_rad[last] - s.phase_true_rad[last]):+.1f} vs true", delta_color="off")
    st.caption(
        f"Direct estimate of the same sample: R = {d.r_ohm[last]:.2f} ohm, X = {d.x_ohm[last]:.2f} ohm. "
        "Deltas compare with the simulated ground truth, which a real system would not have."
    )

    st.subheader("Operating status")
    if result.anomaly_kalman.alarm[last]:
        st.error("ANOMALY: the filtered impedance currently deviates from the baseline beyond the thresholds.")
    else:
        st.success("NORMAL: the filtered impedance is within the thresholds around the baseline.")
    ev = result.events_kalman
    st.write(
        f"Alarm segments (filtered estimate): **{len(result.anomaly_kalman.events)}** - "
        f"true disturbance events in the simulation: **{ev.n_true_events}**, "
        f"false-alarm events: **{ev.n_false_alarm_events}**."
    )


def tab_monitoring(df) -> None:
    st.caption("Grey bands mark the simulated ground-truth disturbances. Hover for values.")
    a, b = st.columns(2)
    a.plotly_chart(viz.plot_resistance(df))
    b.plotly_chart(viz.plot_reactance(df))
    c, d = st.columns(2)
    c.plotly_chart(viz.plot_impedance_magnitude(df))
    d.plotly_chart(viz.plot_phase(df))
    e, f = st.columns(2)
    e.plotly_chart(viz.plot_measurements(df))
    f.plotly_chart(viz.plot_anomaly_timeline(df))


def tab_performance(result, df, sim, kalman, anomaly) -> None:
    st.subheader("Direct vs Kalman-filtered estimation (this run)")
    st.dataframe(metrics_table(result))
    st.caption("'undefined' means the metric has no meaning for this run, e.g. recall when no true disturbance exists.")
    a, b = st.columns(2)
    a.plotly_chart(viz.plot_rmse_comparison(result))
    b.plotly_chart(viz.plot_error_over_time(df))

    st.subheader("Is one seed representative?")
    st.write("Repeat the same configuration with different random seeds to see how stable the comparison is.")
    n_seeds = st.number_input("Number of seeds", min_value=5, max_value=100, value=30, step=5)
    if st.button("Run seed comparison"):
        with st.spinner("Running..."):
            table = compare_over_seeds(sim, range(int(n_seeds)), kalman, anomaly)
        better = int(table["kalman_better"].sum())
        st.write(
            f"Mean RMSE of Z - direct: **{table['rmse_z_direct_ohm'].mean():.3f} ohm**, "
            f"Kalman: **{table['rmse_z_kalman_ohm'].mean():.3f} ohm**. "
            f"The filter was better in **{better} of {len(table)}** seeds."
        )
        st.dataframe(table)


def tab_export(result, sim, kalman, anomaly) -> None:
    st.write("Download the run so it can be analysed or reproduced elsewhere.")
    c1, c2, c3 = st.columns(3)
    c1.download_button("Results (CSV)", data=results_csv(result), file_name="plasmatwin_results.csv", mime="text/csv")
    config_json = experiment_config_json(sim, kalman, anomaly)
    c2.download_button("Configuration (JSON)", data=config_json, file_name="plasmatwin_config.json", mime="application/json")
    report = build_summary_report(result, kalman, anomaly)
    c3.download_button("Summary report (Markdown)", data=report, file_name="plasmatwin_summary.md", mime="text/markdown")
    with st.expander("Preview: summary report"):
        st.markdown(report)
    with st.expander("Preview: configuration JSON"):
        st.code(config_json, language="json")


def tab_model() -> None:
    st.subheader("Model in brief")
    st.latex(r"Z = R + jX,\quad X = \omega L - \frac{1}{\omega C},\quad \omega = 2\pi f")
    st.latex(r"|Z| = \frac{V_{rms}}{I_{rms}},\quad \varphi = \operatorname{atan2}(X, R),\quad R = |Z|\cos\varphi,\ X = |Z|\sin\varphi")
    st.markdown(
        """
**Identifiability.** At one frequency only the equivalent R and X are observable. Many different
(L, C) pairs give the same X, so the physical L and C are used to *generate* the simulation but are
never claimed to be *recovered*.

**Direct estimator.** Applies the formulas above to each sample. Noise passes straight through.

**Kalman filter.** Random-walk state x = [R, X], F = H = I. Q = q² I; the measurement covariance is
obtained by first-order error propagation of the V, I and phase noise. That is an approximation,
because the noise of (R, X) is not exactly Gaussian. A larger q follows changes faster but smooths less.

**Anomaly detector.** Rule-based, not machine learning: deviation from a baseline beyond
max(k·σ, floor), persistence over several samples, optional absolute |Z| band.

**Known weakness.** The detector cannot tell *rising measurement noise* from a *real load change*
(scenario D produces false alarms). See `docs/limitations.md`.
        """
    )


# --------------------------------------------------------------------------- main
def main() -> None:
    try:
        sim, kalman, anomaly = build_configs()
        result = run_experiment(sim, kalman, anomaly)
    except ValueError as exc:
        st.error(f"Invalid configuration: {exc}")
        st.stop()
        return

    df = result.to_dataframe()
    overview, monitoring, performance, export, model = st.tabs(
        ["Overview", "Monitoring", "Performance analysis", "Export", "Model notes"]
    )
    with overview:
        tab_overview(result, sim)
    with monitoring:
        tab_monitoring(df)
    with performance:
        tab_performance(result, df, sim, kalman, anomaly)
    with export:
        tab_export(result, sim, kalman, anomaly)
    with model:
        tab_model()


main()
