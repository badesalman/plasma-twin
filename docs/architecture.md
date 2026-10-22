# Architecture

```
plasma-twin/
├── app.py                 Streamlit UI only (widgets -> configs -> pipeline -> charts)
├── src/
│   ├── physics.py         R-L-C model, Z, |Z|, phase, ideal V/I (validated inputs)
│   ├── measurements.py    NoiseConfig, noise generation, MeasurementSeries
│   ├── simulator.py       SimulationConfig, scenarios A-E, ground truth, DataFrame
│   ├── estimators.py      direct estimator
│   ├── kalman_filter.py   KalmanConfig, measurement covariance, KalmanFilter, run_kalman
│   ├── anomaly_detection.py  AnomalyConfig, detect_anomalies, find_segments
│   ├── metrics.py         error, detection and step-response metrics
│   ├── pipeline.py        run_experiment, compare_over_seeds  (orchestration)
│   ├── report.py          tables, CSV/JSON export, Markdown summary
│   └── visualization.py   Plotly figures
├── tests/                 pytest unit tests
├── experiments/           reproducible validation scripts (Experiments 1-3)
├── examples/              example configuration (JSON)
└── docs/
```

## Data flow

```
SimulationConfig --simulate--> ground truth + ideal V, I, phase
                    + NoiseConfig/seed --> noisy V, I, phase
noisy V, I, phase --estimate_impedance_direct--> R_est, X_est  (direct)
direct estimate   --run_kalman (assumed noise)--> filtered R, X
each estimate     --detect_anomalies--> alarms
ground truth + estimates + alarms --metrics--> tables / figures / reports
```

## Design decisions

* **Numerics are separate from the UI.** `app.py` contains no estimation logic; everything it displays is produced by
  functions in `src/` that are unit-tested without Streamlit.
* **Frozen dataclasses for configuration.** Validation happens at construction, so an invalid configuration cannot start a run;
  configurations round-trip through JSON, which makes every exported run reproducible.
* **Ground truth is kept next to the observations** but only metrics and charts read it. The estimators and the detector
  never see it.
* **Fixed random draws per seed.** Results at different noise levels use the same underlying random numbers, so comparisons are not blurred by sampling luck.
* **`pipeline.py` and `report.py` were added** to the originally proposed structure so the UI and the experiment scripts share one orchestration path.
* **Explicit undefined metrics.** Functions return `None` rather than 0 or NaN when a metric has no meaning.
* **Experiments are scripts, not tests.** Tests check correctness quickly; the experiments report measured behaviour over many seeds.
