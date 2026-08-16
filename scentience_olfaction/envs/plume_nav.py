"""
PlumeNav: a Gymnasium environment for olfactory source localisation.

No Isaac required -- this is the standalone training/benchmark path, in the
lineage of the Scentience `scentience-plume-envs` suite and the Chasing Ghosts
Gymnasium stack (France et al., arXiv:2602.19577), with the filament plume and
the full virtual device replacing analytic fields.

Observation (Box, float32) -- deliberately hardware-shaped:
    [0:6]  baseline-tracked MOX deflection, 6 channels (drift-invariant)
    [6]    d(deflection)/dt of the max channel
    [7:9]  wind vector in BODY frame (simulated anemometer)
    [9]    sin(heading), [10] cos(heading)
  Ground-truth concentration is NOT in the observation, by design.

Action (Box, float32): [forward speed 0..v_max, turn rate -w_max..w_max]

Reward: sparse success bonus + potential-based shaping on distance to source
(shaping keeps the sparse optimum unchanged; Ng et al. 1999).

Episode ends on: source reached (success), out of domain, or timeout.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as e:  # pragma: no cover
    raise ImportError("PlumeNav needs gymnasium: pip install gymnasium") from e

from ..plume.filament import FilamentPlume, FilamentPlumeConfig
from ..sensors.device_np import DeviceState, ScentienceV1


@dataclass
class PlumeNavConfig:
    plume: FilamentPlumeConfig = field(default_factory=lambda: FilamentPlumeConfig(
        source_pos=(0.0, 0.0, 1.0), release_rate_hz=40.0,
        turbulence_intensity=0.30, lagrangian_timescale=1.5,
        meander_std_rad=0.22, meander_timescale=15.0,
        gamma=2.0e-3, sigma0=0.05, max_filaments=6000, max_age_s=40.0,
        ppm_center_initial=300.0,   # inside the MiCS-6814 dynamic range
        domain_min=(-5.0, -12.0, 0.0), domain_max=(30.0, 12.0, 4.0)))
    sensor_profile: str = "fast_modulated"
    dt: float = 0.05
    warmup_s: float = 20.0          # plume spin-up before the robot moves
    v_max: float = 1.0
    w_max: float = 1.5
    z: float = 1.0                  # planar navigation at sensor height
    success_radius: float = 1.0
    timeout_s: float = 120.0
    spawn_box: tuple = ((12.0, -6.0), (24.0, 6.0))
    shaping_scale: float = 0.05
    baseline_tau_s: float = 20.0


class PlumeNavEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, cfg: PlumeNavConfig | None = None, seed: int | None = None):
        self.cfg = cfg or PlumeNavConfig()
        self.observation_space = spaces.Box(-np.inf, np.inf, (11,), np.float32)
        self.action_space = spaces.Box(
            np.array([0.0, -self.cfg.w_max], np.float32),
            np.array([self.cfg.v_max, self.cfg.w_max], np.float32))
        self._plume: FilamentPlume | None = None
        self._seed0 = seed

    # ------------------------------------------------------------------ reset
    def reset(self, *, seed: int | None = None, options=None):
        super().reset(seed=seed if seed is not None else self._seed0)
        s = int(self.np_random.integers(2**31 - 1))
        self._plume = FilamentPlume(self.cfg.plume, seed=s)
        self._device = ScentienceV1(self.cfg.sensor_profile, seed=s + 1)
        self._baseline = None   # initialised to the first reading -- starting
        self._last_defl = np.zeros(6)  # at zero would clip deflection to 0
                                       # for a full baseline time constant

        for _ in range(int(self.cfg.warmup_s / self.cfg.dt)):
            self._plume.step(self.cfg.dt)

        (x0, y0), (x1, y1) = self.cfg.spawn_box
        self._pos = np.array([self.np_random.uniform(x0, x1),
                              self.np_random.uniform(y0, y1)])
        self._heading = self.np_random.uniform(-math.pi, math.pi)
        self._t = 0.0
        self._prev_dist = self._dist_to_source()
        return self._obs(), {}

    # ------------------------------------------------------------------- step
    def step(self, action):
        cfg = self.cfg
        v = float(np.clip(action[0], 0.0, cfg.v_max))
        w = float(np.clip(action[1], -cfg.w_max, cfg.w_max))
        self._heading = _wrap(self._heading + w * cfg.dt)
        self._pos += v * cfg.dt * np.array([math.cos(self._heading),
                                            math.sin(self._heading)])
        self._plume.step(cfg.dt)
        self._t += cfg.dt

        d = self._dist_to_source()
        # potential-based shaping preserves the optimal policy (Ng et al. 1999)
        reward = cfg.shaping_scale * (self._prev_dist - d)
        self._prev_dist = d

        terminated = truncated = False
        if d < cfg.success_radius:
            reward += 10.0
            terminated = True
        lo, hi = np.asarray(cfg.plume.domain_min), np.asarray(cfg.plume.domain_max)
        if np.any(self._pos < lo[:2]) or np.any(self._pos > hi[:2]):
            reward -= 5.0
            terminated = True
        if self._t >= cfg.timeout_s:
            truncated = True
        return self._obs(), reward, terminated, truncated, {"dist": d}

    # ------------------------------------------------------------------- obs
    def _obs(self) -> np.ndarray:
        cfg = self.cfg
        p3 = np.array([self._pos[0], self._pos[1], cfg.z])
        conc = dict(zip(self._plume.species_names,
                        self._plume.sample_species(p3[None, :])[0]))
        r = self._device.step(conc, cfg.dt, DeviceState(flow_mps=0.5))
        mox = np.array([r[c] for c in ("mics1_red", "mics1_nh3", "mics1_ox",
                                       "mics2_red", "mics2_nh3", "mics2_ox")])
        # slow-EMA baseline tracker (what firmware runs); deflection below it
        if self._baseline is None:
            self._baseline = mox.copy()
        a = 1.0 - math.exp(-cfg.dt / cfg.baseline_tau_s)
        self._baseline += a * (mox - self._baseline)
        defl = np.maximum(self._baseline - mox, 0.0)
        ddt = float((defl - self._last_defl).max() / cfg.dt)
        self._last_defl = defl

        wind = self._plume.airflow.velocity(p3[None, :])[0][:2]
        c, s = math.cos(-self._heading), math.sin(-self._heading)
        wind_body = np.array([c * wind[0] - s * wind[1], s * wind[0] + c * wind[1]])
        return np.concatenate([defl, [ddt], wind_body,
                               [math.sin(self._heading), math.cos(self._heading)]]
                              ).astype(np.float32)

    def _dist_to_source(self) -> float:
        src = np.asarray(self.cfg.plume.source_pos[:2])
        return float(np.linalg.norm(self._pos - src))


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi
