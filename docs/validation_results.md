# Validation results

Output of the experiment scripts, copied unchanged. Reproduce with the commands shown (everything is seeded, so the numbers should match).
Experiments use seeds 0-49 and the default configuration unless stated. The scripts and tests are the source of truth; re-run them after changing any code.

## Experiment 1 - noiseless validation

`python -m experiments.exp1_noiseless_validation`

```text
Grid size: 144 operating points
Max |R_est - R_true| : 1.752e-12 ohm
Max |X_est - X_true| : 3.638e-12 ohm
Max relative |Z| error: 3.029e-16
|X_true| range: 3.217 .. 15914.9 ohm
```

The direct estimator reconstructs the simulated equivalent impedance to floating-point rounding error over 144 combinations of frequency, R, L and C.

## Experiment 2 - direct vs Kalman over many seeds

`python -m experiments.exp2_noise_reduction`

```text
Part A - 50 seeds per row, default filter (q = 0.05 ohm/step)
scenario     noise     RMSE_Z direct  RMSE_Z Kalman  Kalman better
stable       low            0.430 ohm       0.125 ohm           100%
stable       medium         0.861 ohm       0.189 ohm           100%
stable       high           2.164 ohm       0.430 ohm           100%

drift        low            0.482 ohm       0.167 ohm           100%
drift        medium         0.965 ohm       0.304 ohm           100%
drift        high           2.425 ohm       0.838 ohm           100%

step         low            0.481 ohm       0.564 ohm             0%
step         medium         0.962 ohm       0.832 ohm           100%
step         high           2.417 ohm       1.465 ohm           100%

intermittent low            0.446 ohm       1.180 ohm             0%
intermittent medium         0.892 ohm       1.611 ohm             0%
intermittent high           2.242 ohm       2.125 ohm            92%

noisy        low            1.097 ohm       0.320 ohm           100%
noisy        medium         2.208 ohm       0.553 ohm           100%
noisy        high           5.832 ohm       2.137 ohm           100%

Part B - process noise q sweep, medium noise, 50 seeds (mean RMSE_Z in ohm)
q [ohm/step]      stable      step
direct             0.861     0.962
0.005              0.105     2.535
0.02               0.137     1.302
0.05               0.189     0.832
0.2                0.347     0.521
1.0                0.673     0.730
```

## Experiment 3 - step response and anomaly detection

`python -m experiments.exp3_step_response`

```text
Step response, 50 seeds per noise level (mean samples, (runs that got there/total))
noise   estimator       within 50%      within 20%    detected  detect delay  false alarms
low     direct         0.0 (50/50)     0.0 (50/50)       50/50           2.0             0
low     kalman         6.1 (50/50)    14.6 (50/50)       50/50           3.0             0

medium  direct         0.0 (50/50)     1.9 (50/50)       50/50           2.0             0
medium  kalman        13.2 (50/50)    30.3 (50/50)       50/50           4.7             0

high    direct         1.5 (50/50)    70.5 (44/50)       42/50          55.0             0
high    kalman        38.3 (50/50)    87.5 (50/50)       50/50          18.1             1

False alarms on scenarios with NO true disturbance (50 seeds each)
scenario noise   estimator    runs with >=1 false alarm
stable   low     direct                         0/50
stable   low     kalman                         0/50
stable   medium  direct                         0/50
stable   medium  kalman                         0/50
stable   high    direct                         0/50
stable   high    kalman                         2/50
noisy    low     direct                        47/50
noisy    low     kalman                        32/50
noisy    medium  direct                        47/50
noisy    medium  kalman                        49/50
noisy    high    direct                        48/50
noisy    high    kalman                        50/50
```
