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


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def run_episode(env, agent, seed: int | None = None) -> dict:
    obs, _ = env.reset(seed=seed)
    agent.reset()
    total, steps = 0.0, 0
    while True:
        obs, r, term, trunc, info = env.step(agent.act(obs))
        total += r
        steps += 1
        if term or trunc:
            return {"success": bool(term and info["dist"] < env.cfg.success_radius + 1e-6),
                    "steps": steps, "reward": total, "final_dist": info["dist"]}
