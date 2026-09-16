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

Reward: success bonus and optional distance-potential shaping. Match
shaping_gamma to the learner's discount for the potential-based guarantee.

Episode ends on: source reached (success), out of domain, or timeout.
"""

from __future__ import annotations

import math
from copy import deepcopy
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
    shaping_gamma: float = 1.0
    """Must equal the learner discount; terminal-state potential is zero."""
    baseline_tau_s: float = 20.0
    wind_speed_range: tuple[float, float] | None = None
    source_strength_range: tuple[float, float] | None = None
    """Optional episode randomization ranges, m/s and centre ppm respectively."""
    stereo_baseline_m: float = 0.04
    """Centre-to-centre spacing of the two MiCS-6814 dies (stereo olfaction):
    chem_left_* samples the LEFT of the agent's heading, chem_right_* the RIGHT. The
    inter-die concentration difference is the cue the kit's second sensor
    exists to provide. 0.04 m is ASSUMED typical dev-board spacing -- measure
    your kit. Set 0.0 for the old mono behaviour (both dies at the centre)."""


class PlumeNavEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(self, cfg: PlumeNavConfig | None = None, seed: int | None = None,
                 render_mode: str | None = None):
        self.cfg = deepcopy(cfg) if cfg is not None else PlumeNavConfig()
        cfg = self.cfg
        for name in ("dt", "v_max", "w_max", "success_radius", "timeout_s", "baseline_tau_s"):
            if not np.isfinite(getattr(cfg, name)) or getattr(cfg, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not 0 <= cfg.shaping_gamma <= 1 or not np.isfinite(cfg.shaping_scale):
            raise ValueError("shaping_gamma must be in [0,1] and shaping_scale finite")
        if not np.isfinite(cfg.warmup_s) or cfg.warmup_s < 0:
            raise ValueError("warmup_s must be finite and nonnegative")
        if not np.isfinite(cfg.stereo_baseline_m) or cfg.stereo_baseline_m < 0:
            raise ValueError("stereo_baseline_m must be finite and nonnegative")
        spawn = np.asarray(cfg.spawn_box, float)
        if spawn.shape != (2, 2) or not np.isfinite(spawn).all() or np.any(spawn[0] > spawn[1]):
            raise ValueError("spawn_box must be ordered finite XY bounds")
        for name in ("wind_speed_range", "source_strength_range"):
            bounds = getattr(cfg, name)
            if bounds is not None and (len(bounds) != 2 or not np.isfinite(bounds).all()
                                       or not 0 <= bounds[0] <= bounds[1]):
                raise ValueError(f"{name} must be nonnegative ordered bounds")
        if render_mode not in (None, "rgb_array"):
            raise ValueError("render_mode must be None or 'rgb_array'")
        self.render_mode = render_mode
        self.observation_space = spaces.Box(-np.inf, np.inf, (11,), np.float32)
        self.action_space = spaces.Box(
            np.array([0.0, -self.cfg.w_max], np.float32),
            np.array([self.cfg.v_max, self.cfg.w_max], np.float32))
        self._plume: FilamentPlume | None = None
        self._seed0 = seed
        self._done = True

    # ------------------------------------------------------------------ reset
    def reset(self, *, seed: int | None = None, options=None):
        super().reset(seed=seed if seed is not None else self._seed0)
        self._seed0 = None  # constructor seed initializes a stream, not every episode
        s = int(self.np_random.integers(2**31 - 1))
        plume_cfg = deepcopy(self.cfg.plume)
        if self.cfg.wind_speed_range is not None:
            direction = np.asarray(plume_cfg.wind_mean, float)
            norm = np.linalg.norm(direction)
            direction = direction / norm if norm > 0 else np.array([1.0, 0.0, 0.0])
            plume_cfg.wind_mean = tuple(direction * self.np_random.uniform(*self.cfg.wind_speed_range))
        if self.cfg.source_strength_range is not None:
            if plume_cfg.emitters is not None:
                raise ValueError("source_strength_range requires the single-source configuration")
            plume_cfg.ppm_center_initial = self.np_random.uniform(*self.cfg.source_strength_range)
        self._plume = FilamentPlume(plume_cfg, seed=s)
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
        self._initial_dist = self._prev_dist
        self._path_length = 0.0
        self._hits = 0
        self._steps = 0
        self._done = False
        return self._obs(), self._info(False, False)

    # ------------------------------------------------------------------- step
    def step(self, action):
        cfg = self.cfg
        if self._done:
            raise RuntimeError("Call reset() before stepping a new episode")
        action = np.asarray(action, dtype=float)
        if action.shape != (2,) or not np.isfinite(action).all():
            raise ValueError("action must contain finite [forward_speed, turn_rate]")
        v = float(np.clip(action[0], 0.0, cfg.v_max))
        w = float(np.clip(action[1], -cfg.w_max, cfg.w_max))
        self._heading = _wrap(self._heading + w * cfg.dt)
        self._pos += v * cfg.dt * np.array([math.cos(self._heading),
                                            math.sin(self._heading)])
        self._plume.step(cfg.dt)
        self._t += cfg.dt
        self._path_length += v * cfg.dt
        self._steps += 1

        d = self._dist_to_source()
        reward = 0.0
        terminated = truncated = False
        success = d < cfg.success_radius
        if success:
            reward += 10.0
            terminated = True
        lo, hi = np.asarray(cfg.plume.domain_min), np.asarray(cfg.plume.domain_max)
        if np.any(self._pos < lo[:2]) or np.any(self._pos > hi[:2]):
            reward -= 5.0
            terminated = True
        if self._t + 1e-9 >= cfg.timeout_s and not terminated:
            truncated = True
        # Time limits bootstrap; true terminal states have zero potential.
        potential_next = 0.0 if terminated else -d
        reward += cfg.shaping_scale * (cfg.shaping_gamma * potential_next + self._prev_dist)
        self._prev_dist = d
        self._done = terminated or truncated
        obs = self._obs()
        self._hits += int(float(obs[:6].max()) > 0.01)
        return obs, reward, terminated, truncated, self._info(success, truncated)

    # ------------------------------------------------------------------- obs
    def _obs(self) -> np.ndarray:
        cfg = self.cfg
        p3 = np.array([self._pos[0], self._pos[1], cfg.z])

        def conc_at(p):
            return dict(zip(self._plume.species_names,
                            self._plume.sample_species(p[None, :])[0]))

        if cfg.stereo_baseline_m > 0.0:
            # Stereo: die 1 left of heading, die 2 right. The L/R difference
            # is the lateralisation cue -- see PlumeNavConfig.stereo_baseline_m.
            half = 0.5 * cfg.stereo_baseline_m
            left = half * np.array([-math.sin(self._heading),
                                    math.cos(self._heading), 0.0])
            r = self._device.step(conc_at(p3 + left), cfg.dt,
                                  DeviceState(flow_mps=0.5),
                                  conc_ppm_2=conc_at(p3 - left))
        else:
            r = self._device.step(conc_at(p3), cfg.dt, DeviceState(flow_mps=0.5))
        mox = np.array([r[c] for c in ("chem_left_red", "chem_left_nh3", "chem_left_ox",
                                       "chem_right_red", "chem_right_nh3", "chem_right_ox")])
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
        sources = [np.asarray(e.position[:2]) for e in self._plume.emitters]
        if not sources:
            raise ValueError("Navigation requires at least one emitter")
        return min(float(np.linalg.norm(self._pos - src)) for src in sources)

    def _info(self, success: bool, timeout: bool) -> dict:
        shortest = max(0.0, self._initial_dist - self.cfg.success_radius)
        spl = (shortest / max(shortest, self._path_length, 1e-9)) if success else 0.0
        return {"dist": self._dist_to_source(), "is_success": bool(success),
                "timeout": bool(timeout), "elapsed_s": self._t,
                "path_length_m": self._path_length, "spl": spl,
                "sensor_contact_fraction": self._hits / max(self._steps, 1)}

    def render(self):
        if self.render_mode != "rgb_array" or self._plume is None:
            return None
        from ..visualization import render_navigation
        return render_navigation(self._plume, self._pos, self._heading, self.cfg.z)


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi
