"""
Baseline agents for PlumeNav.

CastAndSurge is the classic moth-derived strategy (surge upwind during a
whiff, cast crosswind with growing amplitude when the plume is lost) -- the
standard non-learning baseline in plume-tracing work and the behaviour DRL
agents rediscover (Singh et al., Nature Machine Intelligence 5:58-70, 2023).
Any learned policy shipped with this package must beat it to be worth
shipping; `tests/test_env_and_agents.py` holds CastAndSurge itself to beating
RandomAgent, which keeps the environment honest (if random wins, the
observation is broken, not the agent).
"""

from __future__ import annotations

import math

import numpy as np


class RandomAgent:
    def __init__(self, action_space, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.space = action_space

    def act(self, obs: np.ndarray) -> np.ndarray:
        lo, hi = self.space.low, self.space.high
        return (lo + self.rng.random(lo.shape) * (hi - lo)).astype(np.float32)

    def reset(self) -> None:
        pass


class CastAndSurge:
    """
    Detection -> surge upwind. Loss -> cast crosswind, amplitude growing with
    time since last detection. Uses only what the observation provides:
    MOX deflection channels and body-frame wind. No ground truth, no position.
    """

    def __init__(self, detect_threshold: float = 0.01,
                 v_surge: float = 1.0, v_cast: float = 0.6,
                 cast_period0_s: float = 2.0, dt: float = 0.05, seed: int = 0):
        self.thr = detect_threshold
        self.v_surge, self.v_cast = v_surge, v_cast
        self.cast_period0 = cast_period0_s
        self.dt = dt
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self) -> None:
        self.t_since_hit = 1e9
        self.cast_sign = 1.0 if self.rng.random() < 0.5 else -1.0
        self.t_cast = 0.0

    def act(self, obs: np.ndarray) -> np.ndarray:
        defl = obs[:6]
        wind_body = obs[7:9]           # wind TOWARD +x_body means headwind
        detected = float(defl.max()) > self.thr

        if detected:
            self.t_since_hit = 0.0
        else:
            self.t_since_hit += self.dt

        # bearing of the UPWIND direction in body frame
        upwind = math.atan2(-wind_body[1], -wind_body[0])

        if self.t_since_hit < 1.0:
            # SURGE: steer to the upwind bearing, full speed
            turn = np.clip(2.0 * upwind, -1.5, 1.5)
            return np.array([self.v_surge, turn], np.float32)

        # CAST: crosswind zigzag, period (and thus amplitude) growing with
        # time lost -- the counterturning schedule moths use
        period = self.cast_period0 * (1.0 + 0.25 * min(self.t_since_hit, 20.0))
        self.t_cast += self.dt
        if self.t_cast > period:
            self.t_cast = 0.0
            self.cast_sign *= -1.0
        crosswind = _wrap(upwind + self.cast_sign * math.pi / 2.0)
        turn = np.clip(2.0 * crosswind, -1.5, 1.5)
        return np.array([self.v_cast, turn], np.float32)


class StereoCastAndSurge:
    """Cast-and-surge steered by the STEREO cue -- the behaviour the dev
    kit's second chemical sensor exists to enable.

    Follows the olfaction-only navigation formulation of Chasing Ghosts
    (France et al., arXiv:2602.19577, Sec. III-E1/III-F): the inter-sensor
    ONSET TIME LAG resolves the plume's incident angle into a steering bias
    (Eqs. 3-4), with a deadband where the vehicle holds course -- large
    corrections are casting-like, small ones surging-like. Time lag, not
    amplitude difference, on purpose: the two dies carry independent
    unit-to-unit calibration (R0, sensitivity), so a raw left-minus-right
    amplitude cue contains a CONSTANT bias that swamps the true stereo
    signal at small baselines and steers the robot off the plume line --
    measured here at 0.04 m baseline, and the reason the paper's hardware
    formulation is lag-based (whiff arrival order is calibration-invariant).
    Surge/cast switching uses the paper's dual-timescale
    divergence-vs-signal-line filter (Eqs. 5-7, `DivergenceSignal`), and
    the agent DECLARES the source found by the paper's max-concentration
    confidence rule (Eqs. 9-11, `SourceDeclaration`) -- read by
    `run_episode` as the honest, sensor-only alternative to the env's
    privileged distance test.

    At a 0.04 m baseline, 20 Hz sampling and ~1 m/s wind the lag is often
    below one control tick, in which case the cue reads zero and behaviour
    degrades gracefully to plain upwind cast-and-surge -- that resolution
    limit is real physics, not a bug; widen the baseline or slow the wind
    to resolve finer angles.

    Uses only what the observation provides: 6 deflection channels
    (chem_left_*, chem_right_*) and body-frame wind. No ground truth.
    """

    def __init__(self, detect_threshold: float = 0.01,
                 v_surge: float = 1.0, v_cast: float = 0.6,
                 lag_gain: float = 12.0,       # rad/s of turn bias per second of lag
                 lag_window_s: float = 1.0,    # onsets further apart are unrelated
                 lag_hold_s: float = 1.5,      # how long one lag sample steers
                 cast_period0_s: float = 2.0, dt: float = 0.05, seed: int = 0):
        from ..oio.oio import DivergenceSignal
        from .declaration import SourceDeclaration
        self.thr = detect_threshold
        self.v_surge, self.v_cast = v_surge, v_cast
        self.lag_gain = lag_gain
        self.lag_window = lag_window_s
        self.lag_hold = lag_hold_s
        self.cast_period0 = cast_period0_s
        self.dt = dt
        self.rng = np.random.default_rng(seed)
        self._make = lambda: (DivergenceSignal(),
                              SourceDeclaration(min_signal=detect_threshold))
        self.reset()

    def reset(self) -> None:
        self.momentum, self.declare = self._make()
        self.t = 0.0
        self.t_since_hit = 1e9
        self.cast_sign = 1.0 if self.rng.random() < 0.5 else -1.0
        self.t_cast = 0.0
        self._above = [False, False]        # left, right currently above thr
        self._t_onset = [None, None]        # latest onset time per side
        self._bias = 0.0                    # current steering bias [rad/s]
        self._t_bias = -1e9                 # when that bias was set

    @property
    def declared(self) -> bool:
        return self.declare.declared()

    def _update_lag_bias(self, l_now: bool, r_now: bool) -> None:
        """Paper Eqs. 3-4, first order: the side that smells the odor front
        FIRST is the side the plume comes from; the onset lag maps to a turn
        bias toward it. Whiff arrival order survives per-die calibration
        differences that raw amplitude comparison does not."""
        for i, now in enumerate((l_now, r_now)):
            if now and not self._above[i]:
                self._t_onset[i] = self.t
            self._above[i] = now
        tl, tr = self._t_onset
        if tl is not None and tr is not None:
            lag = tr - tl                        # >0: left first -> turn left
            if 0.0 < abs(lag) <= self.lag_window:
                self._bias = float(np.clip(self.lag_gain * lag, -0.8, 0.8))
                self._t_bias = self.t
                self._t_onset = [None, None]     # consume this onset pair
        if self.t - self._t_bias > self.lag_hold:
            self._bias = 0.0                     # stale cue: hold the wind line

    def act(self, obs: np.ndarray) -> np.ndarray:
        left, right = obs[:3], obs[3:6]
        wind_body = obs[7:9]
        peak = float(max(left.max(), right.max()))
        self.t += self.dt
        mom = self.momentum.step(peak, self.dt)
        self.declare.observe(peak, dt=self.dt)   # decimated to ~1 Hz inside
        self._update_lag_bias(float(left.max()) > self.thr,
                              float(right.max()) > self.thr)

        if peak > self.thr:
            self.t_since_hit = 0.0
        else:
            self.t_since_hit += self.dt

        upwind = math.atan2(-wind_body[1], -wind_body[0])

        # Contact window as in CastAndSurge; the divergence signal line
        # (paper: positive deviation = plume entry) EXTENDS the surge while
        # odor momentum is still building, rather than vetoing it.
        in_contact = (self.t_since_hit < 1.0
                      or (mom["surging"] and self.t_since_hit < 5.0))
        if in_contact:
            # SURGE upwind with the lag-derived lateral bias; zero bias
            # (simultaneous onsets, or none resolved) holds the wind line,
            # which is the paper's deadband behaviour.
            turn = np.clip(2.0 * upwind + self._bias, -1.5, 1.5)
            return np.array([self.v_surge, turn], np.float32)

        # CAST: crosswind zigzag with growing period, as in CastAndSurge.
        period = self.cast_period0 * (1.0 + 0.25 * min(self.t_since_hit, 20.0))
        self.t_cast += self.dt
        if self.t_cast > period:
            self.t_cast = 0.0
            self.cast_sign *= -1.0
        crosswind = _wrap(upwind + self.cast_sign * math.pi / 2.0)
        turn = np.clip(2.0 * crosswind, -1.5, 1.5)
        return np.array([self.v_cast, turn], np.float32)


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def run_episode(env, agent, seed: int | None = None) -> dict:
    """Run one episode. An agent may expose a boolean `declared` property --
    the sensor-only "I found it" of Chasing Ghosts Sec. III-H (the UAV's
    'land' action). When it turns True the episode ends there and the
    ground-truth distance at that moment scores it, so declaration quality
    is measured instead of bypassed."""
    obs, _ = env.reset(seed=seed)
    agent.reset()
    total, steps = 0.0, 0
    while True:
        obs, r, term, trunc, info = env.step(agent.act(obs))
        total += r
        steps += 1
        declared = bool(getattr(agent, "declared", False))
        if term or trunc or declared:
            return {"success": bool((term or declared)
                                    and info["dist"] < env.cfg.success_radius + 1e-6),
                    "steps": steps, "reward": total, "final_dist": info["dist"],
                    "declared": declared}
