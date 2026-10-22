# Mathematical model

This document describes exactly what `src/` computes. Symbols use SI units.

## 1. Equivalent RF load

A series R-L-C circuit driven by a sinusoid at frequency $f$ (steady state, phasor description):

$$\omega = 2\pi f,\qquad X = \omega L - \frac{1}{\omega C},\qquad Z = R + jX$$

| Symbol | Meaning | Unit |
|---|---|---|
| R | resistance | ohm |
| L | inductance | henry |
| C | capacitance | farad |
| X | reactance (positive = inductive, negative = capacitive) | ohm |
| Z | complex impedance | ohm |

Series resonance ($X = 0$) occurs at $f_0 = 1/(2\pi\sqrt{LC})$.

### Default operating point

| Parameter | Value | Reason |
|---|---|---|
| f | 13.56 MHz | ISM frequency commonly used in industrial RF / plasma processing |
| R | 10 ohm | low resistance with a strongly reactive load, a typical situation for plasma-like loads |
| L | 1.2 uH | gives $\omega L \approx 102.2$ ohm |
| C | 150 pF | gives $1/(\omega C) \approx 78.2$ ohm |
| V_rms | 100 V | convenient scale; current ≈ 3.8 A, real power ≈ 148 W |

Result: $X \approx +24.0$ ohm, $|Z| \approx 26.0$ ohm, $\varphi \approx 67.4^\circ$, $f_0 \approx 11.86$ MHz.
These are illustrative values chosen for a readable simulation, not measurements of a real plasma.

## 2. Voltage, current, phase

$$|Z| = \sqrt{R^2 + X^2},\qquad \varphi = \operatorname{atan2}(X, R),\qquad I_{rms} = \frac{V_{rms}}{|Z|}$$

$\varphi$ is the phase of the voltage relative to the current. Real power in the load: $P = V_{rms} I_{rms}\cos\varphi = I_{rms}^2 R$.

**Assumption:** $V_{rms}$ is the voltage *across the load at the measurement plane*. A real generator has its own
source impedance and usually regulates power, not voltage; this is not modelled.

## 3. Identifiability

One measurement at one frequency yields one complex number $Z = R + jX$, i.e. two real numbers.
$X$ depends on $L$ and $C$ only through $\omega L - 1/(\omega C)$, so infinitely many $(L, C)$ pairs give the same $X$.
The estimators therefore recover **R and X**, never L and C separately. L and C are used only to *generate* the
simulation. Separating them would need additional assumptions or measurements at several frequencies.

## 4. Measurement noise

For each sample, independent zero-mean Gaussian noise with scale factor $s \ge 0$ (1 except in scenario D):

$$V_m = V(1 + \sigma_v s\,\epsilon_1),\quad I_m = I(1 + \sigma_i s\,\epsilon_2),\quad \varphi_m = \varphi + \sigma_\varphi s\,\epsilon_3,\qquad \epsilon_k \sim \mathcal N(0,1)$$

| Level | $\sigma_v$ | $\sigma_i$ | $\sigma_\varphi$ |
|---|---|---|---|
| low | 1 % | 1 % | 0.5° |
| medium | 2 % | 2 % | 1.0° |
| high | 5 % | 5 % | 2.5° |

This is a modelling choice (gain-type errors on amplitudes, a fixed phase uncertainty), not a model of a specific probe.
A seed always produces the same underlying $\epsilon_k$, whatever the noise level.

## 5. Direct estimator

$$|Z_{est}| = \frac{V_m}{I_m},\qquad R_{est} = |Z_{est}|\cos\varphi_m,\qquad X_{est} = |Z_{est}|\sin\varphi_m$$

Properties:

* exact for noiseless data (verified in Experiment 1 and in the tests);
* samples are processed independently, so the noise passes straight through;
* $V_m/I_m$ is a ratio of noisy numbers, so it has a small bias of order $\sigma^2$ (Jensen's inequality);
* $R_{est}$ can become negative when $|\varphi_m| > 90^\circ$ under noise. It is intentionally **not** clipped.

Invalid samples (non-finite values, $V \le 0$, $I \le$ a minimum current) either raise an error or become NaN.

## 6. Kalman filter

State $x_k = [R_k, X_k]^T$. Random-walk model:

$$x_k = F x_{k-1} + w_k,\quad F = I,\quad w_k \sim \mathcal N(0, Q),\quad Q = q^2 I$$
$$z_k = H x_k + v_k,\quad H = I,\quad v_k \sim \mathcal N(0, R_k),\quad z_k = [R_{est}, X_{est}]^T$$

Prediction: $\hat x_{k|k-1} = \hat x_{k-1}$, $\quad P_{k|k-1} = P_{k-1} + Q$.

Update: $S = P_{k|k-1} + R_k$, $\;K = P_{k|k-1}S^{-1}$, $\;\hat x_k = \hat x_{k|k-1} + K(z_k - \hat x_{k|k-1})$,
$\;P_k = (I-K)P_{k|k-1}(I-K)^T + K R_k K^T$ (Joseph form, keeps $P$ symmetric and positive semi-definite).

**Process noise $q$ (ohm per step).** How much R and X may wander between samples. Large $q$: the filter trusts the
measurements, follows changes quickly, smooths little. Small $q$: strong smoothing, but it lags behind real changes.
Default 0.05 ohm per step. Experiment 2 (part B) shows the trade-off with real numbers.

**Measurement covariance $R_k$ (delta method).** With $|Z| = V/I$ and
$u = [\cos\varphi, \sin\varphi]^T$, $n = [-\sin\varphi, \cos\varphi]^T$, first-order error propagation gives

$$R_k = |Z|^2(\sigma_v^2 + \sigma_i^2)\,uu^T + |Z|^2\sigma_\varphi^2\,nn^T$$

(amplitude noise acts along the radial direction, phase noise along the tangential direction). A unit test checks this
against a Monte-Carlo estimate for small noise.

**Approximation.** $R_k$ is evaluated at the *measured* $|Z|$ and $\varphi$, and the true noise of $(R_{est}, X_{est})$ is not exactly Gaussian
because of the non-linear transform; the approximation worsens at high noise. In scenario D the filter assumes a
constant noise level while the true noise grows (a deliberate model mismatch).

**Invalid measurements** (NaN) are skipped: the filter only predicts, so its uncertainty grows until the next valid sample.

## 7. Anomaly detection

Rule-based, no learning.

1. Baseline: mean and standard deviation of the estimate over the first `baseline_steps` samples.
2. A sample *exceeds* if $|R - R_b| > \tau_R$ or $|X - X_b| > \tau_X$ or $\big||Z| - |Z|_b\big| > \tau_Z$, with
   $\tau = \max(k\,\sigma_b,\; \rho\,|Z|_b)$. $k$ = `k_sigma`, $\rho$ = `min_relative_change`.
3. Optional absolute band on $|Z|$.
4. An alarm requires `persistence` consecutive exceeding samples; none is raised inside the baseline window.

Terminology: **true disturbance** = simulator ground truth (true $|Z|$ deviates from its initial value by more than
`label_tolerance`, default 10 %); **detected anomaly** = alarm; **false alarm** = alarm with no true disturbance.

## 8. Metrics

* RMSE, MAE of R and X; RMSE of $Z$: $\sqrt{\operatorname{mean}|Z_{est} - Z_{true}|^2}$.
* Relative RMSE = RMSE / RMS(true). Undefined (reported as `undefined`) if the true signal is zero.
* Sample level: precision, recall, F1, false-positive rate; undefined when the denominator is zero.
* Event level: a true event is detected if an alarm occurs inside it (plus a grace period); delay = first alarm − event start;
  alarm segments overlapping no true event are false-alarm events.
* Step response: samples until the estimate stays within a fraction (50 % or 20 %) of the step size around the new impedance for 3 consecutive samples.
