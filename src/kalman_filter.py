"""Kalman filter for tracking the equivalent impedance state x = [R, X]^T.

Model
-----
State transition (random walk):   x_k = F x_{k-1} + w_k,   F = I,  w_k ~ N(0, Q)
Measurement:                      z_k = H x_k + v_k,       H = I,  v_k ~ N(0, R_k)

where z_k = [R_direct, X_direct]^T is the direct estimate from V, I and phase.

Process covariance Q = q^2 * I (q = ``process_std_ohm``, ohm per step): how much
R and X are allowed to wander between samples. Larger q -> faster tracking,
less smoothing. Smaller q -> more smoothing, more lag after a real change.

Measurement covariance R_k is derived by first-order (delta-method) error
propagation of the V, I and phase noise through
    |Z| = V/I,   R = |Z| cos(phi),   X = |Z| sin(phi):
    relative noise of |Z|:  var_rel = sigma_v^2 + sigma_i^2
    radial variance:        |Z|^2 var_rel   (direction u = [cos phi, sin phi])
    tangential variance:    |Z|^2 sigma_phi^2 (direction n = [-sin phi, cos phi])
    R_k = var_radial u u^T + var_tangential n n^T
This is an approximation: the true noise of (R, X) is not exactly Gaussian
because of the non-linear transform, and R_k is evaluated at the *measured*
|Z| and phi (the true values are unknown to the filter).

The update uses the Joseph form, which keeps P symmetric and positive
semi-definite under rounding.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from src.estimators import ImpedanceEstimate
from src.measurements import NoiseConfig

FloatArray = NDArray[np.float64]

MIN_VARIANCE_OHM2: float = 1e-12
"""Small diagonal term keeping the covariances numerically positive definite."""


@dataclass(frozen=True)
class KalmanConfig:
    """Tuning parameters of the filter."""

    process_std_ohm: float = 0.05
    measurement_noise_scale: float = 1.0
    initial_std_ohm: float | None = None

    def __post_init__(self) -> None:
        if not np.isfinite(self.process_std_ohm) or self.process_std_ohm < 0:
            raise ValueError("process_std_ohm must be finite and non-negative")
        if not np.isfinite(self.measurement_noise_scale) or self.measurement_noise_scale <= 0:
            raise ValueError("measurement_noise_scale must be positive and finite")
        if self.initial_std_ohm is not None and (
            not np.isfinite(self.initial_std_ohm) or self.initial_std_ohm <= 0
        ):
            raise ValueError("initial_std_ohm must be positive and finite (or None)")


def measurement_covariance(
    z_mag_ohm: ArrayLike, phase_rad: ArrayLike, noise: NoiseConfig, scale: float = 1.0
) -> FloatArray:
    """First-order covariance of the direct (R, X) estimate; shape (..., 2, 2)."""
    z = np.asarray(z_mag_ohm, dtype=float)
    phi = np.asarray(phase_rad, dtype=float)
    var_rel = noise.v_rel_std**2 + noise.i_rel_std**2
    var_radial = scale * z**2 * var_rel
    var_tangential = scale * z**2 * noise.phase_std_rad**2
    c, s = np.cos(phi), np.sin(phi)
    cov = np.empty(np.shape(z) + (2, 2))
    cov[..., 0, 0] = var_radial * c * c + var_tangential * s * s + MIN_VARIANCE_OHM2
    cov[..., 1, 1] = var_radial * s * s + var_tangential * c * c + MIN_VARIANCE_OHM2
    cov[..., 0, 1] = cov[..., 1, 0] = (var_radial - var_tangential) * c * s
    return cov


class KalmanFilter:
    """Two-state random-walk Kalman filter (F = H = I)."""

    def __init__(self, x0: ArrayLike, p0: ArrayLike, process_std_ohm: float) -> None:
        self.x = np.asarray(x0, dtype=float).reshape(2).copy()
        self.p = np.asarray(p0, dtype=float).reshape(2, 2).copy()
        if not np.all(np.isfinite(self.x)) or not np.all(np.isfinite(self.p)):
            raise ValueError("initial state and covariance must be finite")
        if process_std_ohm < 0 or not np.isfinite(process_std_ohm):
            raise ValueError("process_std_ohm must be finite and non-negative")
        self.q = (process_std_ohm**2) * np.eye(2)

    def predict(self) -> None:
        """Prediction step: x stays (F = I), P grows by Q."""
        self.p = self.p + self.q

    def update(self, z: ArrayLike, r_meas: ArrayLike) -> FloatArray:
        """Update step with measurement z and covariance r_meas; returns the innovation."""
        z = np.asarray(z, dtype=float).reshape(2)
        r_meas = np.asarray(r_meas, dtype=float).reshape(2, 2)
        innovation = z - self.x
        s = self.p + r_meas
        try:
            gain = np.linalg.solve(s, self.p).T        # K = P S^-1 (S, P symmetric)
        except np.linalg.LinAlgError as exc:
            raise ValueError("innovation covariance is singular") from exc
        self.x = self.x + gain @ innovation
        i_k = np.eye(2) - gain
        self.p = i_k @ self.p @ i_k.T + gain @ r_meas @ gain.T   # Joseph form
        self.p = 0.5 * (self.p + self.p.T)
        return innovation


@dataclass(frozen=True)
class KalmanResult:
    """Filtered estimates (NaN before the first valid measurement)."""

    r_ohm: FloatArray
    x_ohm: FloatArray
    std_r_ohm: FloatArray
    std_x_ohm: FloatArray
    innovation_r_ohm: FloatArray
    innovation_x_ohm: FloatArray

    def as_estimate(self) -> ImpedanceEstimate:
        """View as an ImpedanceEstimate (|Z| and phase derived from the filtered R, X)."""
        return ImpedanceEstimate(
            z_mag_ohm=np.hypot(self.r_ohm, self.x_ohm),
            r_ohm=self.r_ohm,
            x_ohm=self.x_ohm,
            phase_rad=np.arctan2(self.x_ohm, self.r_ohm),
        )


def run_kalman(
    direct: ImpedanceEstimate, assumed_noise: NoiseConfig, config: KalmanConfig | None = None
) -> KalmanResult:
    """Filter a series of direct estimates.

    Samples where the direct estimate is NaN (invalid) are skipped: the filter
    only predicts, so its uncertainty grows until the next valid measurement.
    ``assumed_noise`` is what the filter *believes* about the sensor noise.
    """
    config = config or KalmanConfig()
    r_d = np.atleast_1d(np.asarray(direct.r_ohm, dtype=float))
    x_d = np.atleast_1d(np.asarray(direct.x_ohm, dtype=float))
    z_mag = np.atleast_1d(np.asarray(direct.z_mag_ohm, dtype=float))
    phi = np.atleast_1d(np.asarray(direct.phase_rad, dtype=float))
    n = r_d.size

    valid = np.isfinite(r_d) & np.isfinite(x_d) & np.isfinite(z_mag) & np.isfinite(phi)
    cov = np.full((n, 2, 2), np.nan)
    cov[valid] = measurement_covariance(
        z_mag[valid], phi[valid], assumed_noise, config.measurement_noise_scale
    )

    out = {name: np.full(n, np.nan) for name in ("r", "x", "sr", "sx", "ir", "ix")}
    kf: KalmanFilter | None = None
    for k in range(n):
        if kf is None:
            if not valid[k]:
                continue
            p0 = (
                cov[k]
                if config.initial_std_ohm is None
                else (config.initial_std_ohm**2) * np.eye(2)
            )
            kf = KalmanFilter([r_d[k], x_d[k]], p0, config.process_std_ohm)
        else:
            kf.predict()
            if valid[k]:
                innovation = kf.update([r_d[k], x_d[k]], cov[k])
                out["ir"][k], out["ix"][k] = innovation
        out["r"][k], out["x"][k] = kf.x
        out["sr"][k], out["sx"][k] = np.sqrt(np.diag(kf.p))

    return KalmanResult(
        r_ohm=out["r"],
        x_ohm=out["x"],
        std_r_ohm=out["sr"],
        std_x_ohm=out["sx"],
        innovation_r_ohm=out["ir"],
        innovation_x_ohm=out["ix"],
    )
