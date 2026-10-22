# PlasmaTwin - Virtual RF Load Monitoring and Impedance Estimation

PlasmaTwin is a student engineering project that simulates an RF electrical load, produces noisy voltage, current and
phase measurements, estimates the load's equivalent impedance from them, tracks its changes with a Kalman filter,
and flags abnormal behaviour. Everything is validated against simulated ground truth with quantitative error metrics.

> **Scope.** This is an educational simulation of electrical load behaviour. It does **not** estimate plasma density,
> plasma temperature or process quality; that would require validated physical models and real experimental data.

## Problem statement

An RF generator delivers power to a load whose impedance is not constant. From measurements of RMS voltage, RMS current and
their phase difference, can the equivalent complex impedance $Z = R + jX$ be estimated reliably, tracked over time despite measurement noise,
and can sudden or abnormal changes be detected quickly without raising false alarms?

## Motivation

In industrial RF power delivery, for example RF supplies for plasma processing, the load impedance changes with
operating conditions. A mismatch between the source and the load reflects power. As an illustration for the default load
(Z ≈ 10 + j24 ohm) driven from a 50 ohm source without matching, the reflection coefficient is |Γ| ≈ 0.72, i.e. about 52 % of the
incident power would be reflected. Monitoring the impedance, and noticing when it changes, is therefore a basic
building block of controlling and protecting such systems. This project explores that building block in software, on synthetic data.

## Key features

* Series R-L-C equivalent load with physically meaningful default values (13.56 MHz, R = 10 ohm, L = 1.2 uH, C = 150 pF).
* Five time-dependent scenarios: stable, gradual drift, sudden step, increasing noise, intermittent disturbances.
* Configurable, reproducible (seeded) measurement noise on voltage, current and phase.
* Two estimators: direct (V/I and phase) and a Kalman filter on [R, X] with a measurement covariance derived by error propagation.
* Transparent rule-based anomaly detector with ground-truth labels, detection delay and false-alarm counting.
* Metrics that report `undefined` instead of misleading numbers.
* Interactive Streamlit dashboard with Plotly charts; export to CSV, JSON and a Markdown summary.
* Unit tests and three reproducible numerical experiments.

## Technology stack

Python 3.11+, NumPy, SciPy, Pandas, Plotly, Streamlit, Pytest. No cloud services, no paid APIs, no hardware.

## Mathematical model (short version)

$$X = \omega L - \frac{1}{\omega C},\quad Z = R + jX,\quad |Z| = \frac{V_{rms}}{I_{rms}},\quad \varphi = \operatorname{atan2}(X, R)$$
$$R_{est} = |Z_{est}|\cos\varphi_m,\qquad X_{est} = |Z_{est}|\sin\varphi_m$$

At a single frequency only R and X are identifiable; L and C cannot be recovered separately, so they only generate the simulation.
The Kalman filter uses a random-walk state $x = [R, X]^T$ ($F = H = I$, $Q = q^2 I$). Full derivation, assumptions and
approximations: [docs/mathematical_model.md](docs/mathematical_model.md).

## System architecture

```
app.py                 Streamlit UI (widgets -> configs -> pipeline -> charts)
src/physics.py         R-L-C model
src/measurements.py    noise model
src/simulator.py       scenarios and ground truth
src/estimators.py      direct estimator
src/kalman_filter.py   Kalman filter
src/anomaly_detection.py  rule-based detector
src/metrics.py         error and detection metrics
src/pipeline.py        end-to-end run, multi-seed comparison
src/report.py          tables and exports
src/visualization.py   Plotly figures
tests/                 pytest
experiments/           validation experiments
```

Details: [docs/architecture.md](docs/architecture.md).

## Installation (Windows, VS Code)

1. Install Python 3.11 or newer from python.org (tick **Add python.exe to PATH**) and, optionally, Git.
2. Open the project folder in VS Code and a terminal in it, then:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

If PowerShell refuses to run the activation script, run once:
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

## Running the application

```powershell
streamlit run app.py
```

A browser tab opens (default http://localhost:8501). Choose a scenario and parameters in the sidebar; the dashboard
updates immediately.

## Example simulation scenarios

| Scenario | What happens | What it shows |
|---|---|---|
| A - Stable | R and L constant | filtering reduces noise; any alarm is a false alarm |
| B - Drift | R and L change slowly | the filter tracks slow change; detection after the drift accumulates |
| C - Step | R ×1.5 and L ×1.04 at mid-run | response time and detection delay |
| D - Noisy | load constant, noise ramps up to 4× | robustness; exposes the detector's weakness |
| E - Intermittent | three temporary disturbances | repeated detection, filter lag at fast changes |

An example configuration is in [examples/example_config.json](examples/example_config.json).

## Screenshots

*(Screenshots will be added after running the application locally; place images in `docs/images/`.)*

| Overview | Monitoring | Performance |
|---|---|---|
| _to be added_ | _to be added_ | _to be added_ |

## Testing

```powershell
pytest
```

Tests cover impedance calculations for known R-L-C values, purely resistive/inductive/capacitive cases, the noiseless estimator,
deterministic simulation for a fixed seed, stable and sudden-change scenarios, Kalman filter stability, invalid inputs,
anomaly thresholds and metric formulas.

## Validation methodology

Run the experiments from the repository root (all seeded):

```powershell
python -m experiments.exp1_noiseless_validation
python -m experiments.exp2_noise_reduction
python -m experiments.exp3_step_response
```

The complete, unedited outputs are in [docs/validation_results.md](docs/validation_results.md). Summary of what they show
(50 seeds, default configuration):

* **Experiment 1 (noiseless):** over 144 operating points the direct estimator reproduces R and X to about 1e-12 ohm, i.e. rounding error.
* **Experiment 2 (noise reduction):** at medium noise, mean complex-impedance RMSE direct → Kalman: stable 0.861 → 0.189 ohm, drift 0.965 → 0.304,
  increasing noise 2.208 → 0.553. For step-like changes the filter helps less or loses: step 0.962 → 0.832 (medium) but 0.481 → 0.564 (low),
  intermittent 0.892 → 1.611 (medium). The process-noise sweep shows the trade-off: q = 0.005 gives 0.105 ohm on a stable load but 2.535 ohm on a step; q = 0.2 gives 0.347 and 0.521.
* **Experiment 3 (step response):** at low noise the direct estimate is within half the step size immediately, the Kalman estimate needs 6.1 samples on average;
  both detectors found the step in 50/50 runs (mean delay 2.0 vs 3.0 samples). At high noise the direct detector found it in 42/50 runs, the Kalman-based one in 50/50.
  On a stable load there were no false alarms at low and medium noise.
* **Weakness found:** in scenario D (stable load, rising noise) both detectors raised false alarms in most runs (32-50 of 50), because the thresholds are calibrated on the initial noise level.

## Known limitations

See [docs/limitations.md](docs/limitations.md). Main points: simplified series R-L-C load, single frequency (only R and X identifiable),
idealised voltage source, modelled (not calibrated) noise, first-order Kalman covariance, rule-based detector that confuses noise growth with load changes.

## Future improvements

* Detector that monitors the noise level (sliding-window variance or a Kalman innovation test) so that rising noise is not reported as a load change.
* Adaptive process noise q (or an interacting multiple-model filter) to reduce lag after steps while keeping smoothing in quiet periods.
* Multi-frequency measurements to explore which additional parameters become identifiable.
* Reflection coefficient, VSWR and a matching-network model.
* Validation on real VI-probe or network-analyser data, with calibration and systematic errors.

## Development note

This project was developed step by step with an AI assistant (Claude, Anthropic): the assistant wrote and explained code, and the author ran the
tests and experiments locally and reviewed the results.
