"""Time-dependent RF load simulation with ground truth.

The simulator changes the *physical* parameters (R and L; C is constant) over
time. The equivalent reactance X(t) = w L(t) - 1/(w C) follows from them.
Ideal V, I and phase are computed from Z(t), then measurement noise is added.

Scenarios
---------
stable        R and L constant.
drift         R and L change linearly (starting after the baseline window).
step          R and L jump by a factor at ``event_start_fraction`` of the run.
noisy         True load constant; the noise level ramps up after the baseline window.
intermittent  Several rectangular disturbances (R and L factors) of fixed length.

Ground-truth label
------------------
``anomaly_true[k]`` is True when the true impedance differs from its initial
value by more than ``label_tolerance`` (relative to |Z0|). This is the
reference against which detected anomalies and false alarms are judged.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from src import physics
from src.measurements import NoiseConfig, add_measurement_noise

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

MIN_STEPS = 10


class Scenario(str, Enum):
    STABLE = "stable"
    DRIFT = "drift"
    STEP = "step"
    NOISY = "noisy"
    INTERMITTENT = "intermittent"


@dataclass(frozen=True)
class SimulationConfig:
    """All parameters of one simulation run (SI units)."""

    frequency_hz: float = physics.DEFAULT_FREQUENCY_HZ
    v_rms_v: float = 100.0
    resistance_ohm: float = 10.0
    inductance_h: float = 1.2e-6
    capacitance_f: float = 150e-12
    n_steps: int = 600
    dt_s: float = 0.01
    scenario: Scenario = Scenario.STABLE
    noise: NoiseConfig = NoiseConfig()
    seed: int = 0
    baseline_steps: int = 50
    event_start_fraction: float = 0.5
    event_r_factor: float = 1.5
    event_l_factor: float = 1.04
    drift_r_fraction: float = 0.5
    drift_l_fraction: float = 0.05
    noise_ramp_factor: float = 4.0
    n_pulses: int = 3
    pulse_steps: int = 30
    label_tolerance: float = 0.10

    def __post_init__(self) -> None:
        object.__setattr__(self, "scenario", Scenario(self.scenario))
        if isinstance(self.noise, dict):
            object.__setattr__(self, "noise", NoiseConfig(**self.noise))

        # physical validity (raises ValueError for bad frequency, R, L, C)
        physics.rlc_impedance(
            self.frequency_hz, self.resistance_ohm, self.inductance_h, self.capacitance_f
        )
        if not np.isfinite(self.v_rms_v) or self.v_rms_v <= 0:
            raise ValueError("v_rms_v must be a positive finite number")
        if self.resistance_ohm == 0 and abs(self._initial_reactance()) < physics.MIN_IMPEDANCE_OHM:
            raise ValueError("initial impedance must not be zero")

        if int(self.n_steps) != self.n_steps or self.n_steps < MIN_STEPS:
            raise ValueError(f"n_steps must be an integer >= {MIN_STEPS}")
        if not np.isfinite(self.dt_s) or self.dt_s <= 0:
            raise ValueError("dt_s must be positive and finite")
        if int(self.seed) != self.seed or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if int(self.baseline_steps) != self.baseline_steps or not 5 <= self.baseline_steps < self.n_steps:
            raise ValueError("baseline_steps must be an integer with 5 <= baseline_steps < n_steps")
        for name in ("event_r_factor", "event_l_factor", "label_tolerance"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if not np.isfinite(self.noise_ramp_factor) or self.noise_ramp_factor < 1:
            raise ValueError("noise_ramp_factor must be >= 1")
        for name in ("drift_r_fraction", "drift_l_fraction"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= -1:
                raise ValueError(f"{name} must be finite and > -1")

        if self.scenario is Scenario.STEP:
            if not 0 < self.event_start_fraction < 1:
                raise ValueError("event_start_fraction must be in (0, 1)")
            if self.step_index <= self.baseline_steps or self.step_index >= self.n_steps:
                raise ValueError("step must occur after the baseline window and before the end")
        if self.scenario is Scenario.INTERMITTENT:
            if self.n_pulses < 1 or self.pulse_steps < 1:
                raise ValueError("n_pulses and pulse_steps must be >= 1")
            if self.pulse_steps >= self._pulse_spacing():
                raise ValueError("pulses do not fit: reduce pulse_steps/n_pulses or increase n_steps")

    def _initial_reactance(self) -> float:
        return float(physics.reactance(self.frequency_hz, self.inductance_h, self.capacitance_f))

    def _pulse_spacing(self) -> int:
        return (self.n_steps - self.baseline_steps) // self.n_pulses

    @property
    def step_index(self) -> int:
        """Sample index at which the step scenario changes the load."""
        return int(round(self.event_start_fraction * self.n_steps))

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["scenario"] = self.scenario.value
        return data

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SimulationConfig":
        allowed = {f.name for f in fields(cls)}
        unknown = set(data) - allowed
        if unknown:
            raise ValueError(f"unknown configuration keys: {sorted(unknown)}")
        return cls(**data)

    @classmethod
    def from_json(cls, text: str) -> "SimulationConfig":
        return cls.from_dict(json.loads(text))


@dataclass(frozen=True)
class SimulationResult:
    """Ground truth, ideal values and noisy measurements of one run."""

    config: SimulationConfig
    time_s: FloatArray
    r_true_ohm: FloatArray
    l_true_h: FloatArray
    x_true_ohm: FloatArray
    z_mag_true_ohm: FloatArray
    phase_true_rad: FloatArray
    v_ideal_v: FloatArray
    i_ideal_a: FloatArray
    v_meas_v: FloatArray
    i_meas_a: FloatArray
    phase_meas_rad: FloatArray
    noise_scale: FloatArray
    anomaly_true: BoolArray

    def to_dataframe(self) -> pd.DataFrame:
        """Ground truth and measurements as a table (angles in degrees)."""
        return pd.DataFrame(
            {
                "time_s": self.time_s,
                "r_true_ohm": self.r_true_ohm,
                "x_true_ohm": self.x_true_ohm,
                "z_mag_true_ohm": self.z_mag_true_ohm,
                "phase_true_deg": np.rad2deg(self.phase_true_rad),
                "v_meas_v": self.v_meas_v,
                "i_meas_a": self.i_meas_a,
                "phase_meas_deg": np.rad2deg(self.phase_meas_rad),
                "noise_scale": self.noise_scale,
                "anomaly_true": self.anomaly_true,
            }
        )


def _build_profiles(cfg: SimulationConfig) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Return R(t), L(t) and the per-sample noise scale for the scenario."""
    n = cfg.n_steps
    k = np.arange(n)
    r = np.full(n, cfg.resistance_ohm, dtype=float)
    l = np.full(n, cfg.inductance_h, dtype=float)
    scale = np.ones(n)

    # progress 0..1 starting after the baseline window
    progress = np.clip((k - cfg.baseline_steps) / (n - 1 - cfg.baseline_steps), 0.0, 1.0)

    if cfg.scenario is Scenario.DRIFT:
        r = cfg.resistance_ohm * (1.0 + cfg.drift_r_fraction * progress)
        l = cfg.inductance_h * (1.0 + cfg.drift_l_fraction * progress)
    elif cfg.scenario is Scenario.STEP:
        after = k >= cfg.step_index
        r = np.where(after, cfg.resistance_ohm * cfg.event_r_factor, cfg.resistance_ohm)
        l = np.where(after, cfg.inductance_h * cfg.event_l_factor, cfg.inductance_h)
    elif cfg.scenario is Scenario.NOISY:
        scale = 1.0 + (cfg.noise_ramp_factor - 1.0) * progress
    elif cfg.scenario is Scenario.INTERMITTENT:
        spacing = cfg._pulse_spacing()
        active = np.zeros(n, dtype=bool)
        for p in range(cfg.n_pulses):
            start = cfg.baseline_steps + p * spacing + (spacing - cfg.pulse_steps) // 2
            active[start : start + cfg.pulse_steps] = True
        r = np.where(active, cfg.resistance_ohm * cfg.event_r_factor, cfg.resistance_ohm)
        l = np.where(active, cfg.inductance_h * cfg.event_l_factor, cfg.inductance_h)
    return r, l, scale


def simulate(config: SimulationConfig) -> SimulationResult:
    """Run one simulation. Deterministic for a given configuration (incl. seed)."""
    r_true, l_true, noise_scale = _build_profiles(config)
    x_true = physics.reactance(config.frequency_hz, l_true, config.capacitance_f)
    ideal = physics.ideal_measurements(config.v_rms_v, r_true, x_true)

    rng = np.random.default_rng(config.seed)
    v_meas, i_meas, phi_meas = add_measurement_noise(
        ideal.v_rms_v, ideal.i_rms_a, ideal.phase_rad, config.noise, rng, noise_scale
    )

    z_true = r_true + 1j * x_true
    z0 = z_true[0]
    anomaly_true = (np.abs(z_true - z0) / np.abs(z0)) > config.label_tolerance

    return SimulationResult(
        config=config,
        time_s=np.arange(config.n_steps) * config.dt_s,
        r_true_ohm=r_true,
        l_true_h=l_true,
        x_true_ohm=x_true,
        z_mag_true_ohm=np.abs(z_true),
        phase_true_rad=ideal.phase_rad,
        v_ideal_v=ideal.v_rms_v,
        i_ideal_a=ideal.i_rms_a,
        v_meas_v=v_meas,
        i_meas_a=i_meas,
        phase_meas_rad=phi_meas,
        noise_scale=noise_scale,
        anomaly_true=anomaly_true,
    )
