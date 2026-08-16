"""
Olfactory Inertial Odometry (OIO) -- reference implementation.

Implements the concept line developed in:

  * France, Kondaveeti, Daescu, "Olfactory Inertial Odometry: Methods for
    Calibration and Drift Compensation" (arXiv:2506.04539, IEEE INERTIAL 2025)
    -- sensor calibration/differencing and the use of olfactory signals as an
    exteroceptive reference for inertial drift.
  * France et al., "Chasing Ghosts: A Simulation-to-Real Olfactory Navigation
    Stack" (arXiv:2602.19577) -- dual-timescale EMA bout detection
    (github.com/KordelFranceTech/ChasingGhosts).

The idea, compressed: an IMU drifts without bound; odor bouts are recurring
exteroceptive events tied to world structure (the plume), so they can bound
that drift the way loop closures bound SLAM drift.  Two corrections are used:

  heading  -- the anemometer measures wind in the BODY frame; the world wind
              bearing is slowly varying.  The mismatch between (measured body
              bearing + estimated heading) and the world bearing is a direct
              heading-error observation, applied as a complementary filter.
  crosswind - bout onsets happen inside the plume envelope.  The estimator
              maintains a running estimate of the plume axis (crosswind
              coordinate of historical bouts) and pulls the crosswind position
              toward it at each new bout.  Downwind drift is NOT observable
              from bouts alone -- the tests assert exactly that asymmetry
              rather than pretending otherwise.

This is a faithful, testable rendering of the published concept, not a claim
of the papers' exact production algorithms.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


# ---------------------------------------------------------------------------
# Bout detection -- dual-timescale EMA (Chasing Ghosts)
# ---------------------------------------------------------------------------
@dataclass
class BoutDetectorConfig:
    tau_fast_s: float = 0.5
    tau_slow_s: float = 8.0
    k_sigma: float = 3.0        # onset threshold in units of noise sigma
    hysteresis: float = 0.5     # offset threshold = k_sigma * hysteresis
    noise_sigma: float = 1e-3   # of the deflection signal; calibrate in clean air


class BoutDetector:
    """Fast-EMA minus slow-EMA exceeding a noise-referenced threshold marks a
    bout (whiff encounter). The slow EMA doubles as the baseline tracker, so
    the detector is inherently drift-tolerant -- the property that makes it
    usable on MOX hardware at all."""

    def __init__(self, cfg: BoutDetectorConfig):
        self.cfg = cfg
        self.reset()

    def reset(self) -> None:
        self.fast = 0.0
        self.slow = 0.0
        self.in_bout = False
        self.n_bouts = 0

    def step(self, x: float, dt: float) -> dict:
        c = self.cfg
        af = 1.0 - math.exp(-dt / c.tau_fast_s)
        asl = 1.0 - math.exp(-dt / c.tau_slow_s)
        self.fast += af * (x - self.fast)
        self.slow += asl * (x - self.slow)
        d = self.fast - self.slow
        onset = False
        if not self.in_bout and d > c.k_sigma * c.noise_sigma:
            self.in_bout, onset = True, True
            self.n_bouts += 1
        elif self.in_bout and d < c.k_sigma * c.hysteresis * c.noise_sigma:
            self.in_bout = False
        return {"bout": self.in_bout, "onset": onset, "excess": d}


# ---------------------------------------------------------------------------
# Platform presets -- IMU/motion characteristics per robot class
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PlatformSpec:
    """IMU error model + motion character for a robot class. Values are
    representative consumer-grade MEMS figures (evidence: ASSUMED --
    replace per platform; the POINT is the relative structure: legged
    platforms add gait vibration, UAVs add rotor downwash at the sensor)."""
    name: str
    accel_bias_ms2: float
    accel_noise_ms2: float          # per-sample white noise (1 sigma)
    gyro_bias_rad_s: float
    gyro_noise_rad_s: float
    gait_hz: float = 0.0            # 0 = no gait vibration
    gait_accel_ms2: float = 0.0
    sensor_flow_mps: float = 0.3
    """Airflow over the sensor element from platform motion/rotors. Feeds the
    MOX tau flow correction -- a UAV's downwash makes its nose FASTER."""


PLATFORMS = {
    "uav": PlatformSpec("uav", 0.03, 0.06, 0.002, 0.004,
                        gait_hz=0.0, gait_accel_ms2=0.0, sensor_flow_mps=2.0),
    "quadruped": PlatformSpec("quadruped", 0.05, 0.10, 0.003, 0.005,
                              gait_hz=2.5, gait_accel_ms2=0.8, sensor_flow_mps=0.5),
    "biped": PlatformSpec("biped", 0.05, 0.12, 0.003, 0.006,
                          gait_hz=1.8, gait_accel_ms2=1.2, sensor_flow_mps=0.4),
    "arm": PlatformSpec("arm", 0.01, 0.02, 0.001, 0.002,
                        gait_hz=0.0, gait_accel_ms2=0.0, sensor_flow_mps=0.2),
    # 'arm' note: a manipulator's base does not translate, so OIO degenerates
    # to end-effector pose refinement + concentration mapping; see
    # examples/03_olfactory_inertial_odometry.py --platform arm.
}


# ---------------------------------------------------------------------------
# The estimator
# ---------------------------------------------------------------------------
@dataclass
class OIOConfig:
    platform: str = "quadruped"
    wind_world_bearing_rad: float = 0.0   # slowly-varying map prior (mean wind)
    k_heading: float = 0.05               # complementary gain, per step at bout
    k_crosswind: float = 0.15             # position pull toward plume axis
    min_wind_mps: float = 0.3             # below this the anemometer is noise
    detector: BoutDetectorConfig = field(default_factory=BoutDetectorConfig)


class OlfactoryInertialOdometry:
    """
    2-D planar estimator (x = downwind, y = crosswind, heading).

    step() consumes raw IMU + anemometer + olfactory deflection and returns
    the corrected pose estimate alongside the pure dead-reckoning estimate,
    so the improvement is always measurable.
    """

    def __init__(self, cfg: OIOConfig, seed: int = 0):
        self.cfg = cfg
        self.spec = PLATFORMS[cfg.platform]
        self.rng = np.random.default_rng(seed)
        self.detector = BoutDetector(cfg.detector)
        self.reset()

    def reset(self) -> None:
        self.detector.reset()
        self.p_dr = np.zeros(2)      # dead-reckoning position
        self.p_oio = np.zeros(2)     # corrected position
        self.h_dr = 0.0              # dead-reckoning heading
        self.h_oio = 0.0
        self.v = np.zeros(2)
        self._axis_est = None        # crosswind coordinate of the plume axis
        self._t = 0.0

    # -- IMU simulation belongs to the caller in Isaac; provided here so the
    # -- standalone examples exercise realistic error growth.
    def simulate_imu(self, true_accel: np.ndarray, true_omega: float,
                     dt: float) -> tuple[np.ndarray, float]:
        s = self.spec
        gait = 0.0
        if s.gait_hz > 0:
            gait = s.gait_accel_ms2 * math.sin(2 * math.pi * s.gait_hz * self._t)
        accel = (true_accel + s.accel_bias_ms2 + gait
                 + s.accel_noise_ms2 * self.rng.standard_normal(2))
        omega = (true_omega + s.gyro_bias_rad_s
                 + s.gyro_noise_rad_s * self.rng.standard_normal())
        return accel, omega

    def step(self, accel_body: np.ndarray, omega: float,
             wind_body: np.ndarray, deflection: float, dt: float) -> dict:
        cfg = self.cfg
        self._t += dt

        # --- dead reckoning (both estimates share it) -----------------------
        self.h_dr += omega * dt
        self.h_oio += omega * dt
        c, s = math.cos(self.h_oio), math.sin(self.h_oio)
        a_world = np.array([c * accel_body[0] - s * accel_body[1],
                            s * accel_body[0] + c * accel_body[1]])
        self.v += a_world * dt
        self.p_dr += self.v * dt
        self.p_oio += self.v * dt

        # --- olfactory event ------------------------------------------------
        det = self.detector.step(deflection, dt)

        # --- heading correction from the anemometer -------------------------
        wind_speed = float(np.linalg.norm(wind_body))
        if wind_speed > cfg.min_wind_mps:
            bearing_body = math.atan2(wind_body[1], wind_body[0])
            bearing_world_est = self.h_oio + bearing_body
            err = _angle_wrap(cfg.wind_world_bearing_rad - bearing_world_est)
            # trust the wind reference more during a bout: inside the plume the
            # local wind is the transporting flow, not a stray eddy
            gain = cfg.k_heading * (2.0 if det["bout"] else 1.0)
            self.h_oio += gain * err

        # --- crosswind correction at bout onsets ----------------------------
        if det["onset"]:
            y = self.p_oio[1]
            if self._axis_est is None:
                self._axis_est = y
            else:
                self._axis_est += 0.2 * (y - self._axis_est)
            self.p_oio[1] += cfg.k_crosswind * (self._axis_est - self.p_oio[1])

        return {"p_dr": self.p_dr.copy(), "p_oio": self.p_oio.copy(),
                "h_dr": self.h_dr, "h_oio": self.h_oio, **det}


def _angle_wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi
