# Limitations

## Scientific scope

* This is an **educational simulation of an electrical load**. It does not estimate plasma density, electron temperature,
  or process quality. Doing so would need validated physical models and real experimental data.
* The load is a **series R-L-C** with prescribed time dependence. Real plasma loads are non-linear (harmonics), depend on
  power, pressure and gas, and are usually coupled through a matching network. None of this is modelled.
* **Single frequency, steady state.** Only R and X are identifiable; L and C cannot be separated (see `mathematical_model.md`).
* **Idealised source.** The voltage at the load is given. A real generator has source impedance and regulates power.
* **Noise model is a modelling choice.** Independent Gaussian noise with fixed standard deviations is not a calibrated model of a specific probe.
  Real systems also have systematic errors (calibration, probe phase error, drift) that are not simulated.
* **No time-domain waveform.** The simulation works with RMS values and phase, not sampled waveforms; no synchronous detection or filtering of raw signals is modelled.

## Estimation

* The direct estimator has a small ratio bias and can output negative R under noise; it is intentionally not clipped.
* The Kalman measurement covariance is a first-order approximation; the true noise of (R, X) is non-Gaussian.
* The filter assumes a **random-walk** state. After a sudden change it lags (measured in Experiment 3); with a low process noise q it can be worse than the direct estimate for step-like and intermittent disturbances (measured in Experiment 2).
* The filter is "matched" by default (it knows the simulated noise level). In practice the noise level would itself have to be estimated.

## Anomaly detection

* Rule-based thresholds on a baseline; **not** a trained model.
* **It cannot distinguish rising measurement noise from a real load change.** In scenario D (stable load, noise ramping up to 4x)
  the detector raised false alarms in almost every run (see `validation_results.md`). Possible improvements: monitor the noise level
  with a sliding window, adapt the thresholds, or test the Kalman innovations against their predicted covariance.
* The baseline window is assumed to be normal operation and is not checked.
* Defaults (k = 5, persistence 3, 3 % floor) were fixed in advance and not tuned to the results. They are not optimal for every scenario.
* Detection delay depends on the persistence and on the filter's lag.

## Experiments

* Results are for the configurations and 50 seeds used in the scripts; other parameters can give different conclusions.
* Ground truth labels depend on a tolerance (10 % of |Z|); changing it changes precision and recall.
