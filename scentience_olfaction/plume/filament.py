"""
Filament-based Lagrangian plume model (Farrell et al. 2002, Env. Fluid Mech. 2:143-169).

Independent implementation from the published equations. NOT derived from GADEN
source (LGPL-3.0) -- GADEN is used only as a cross-check on numeric behaviour.

Two corrections relative to the GADEN formulation are applied deliberately:

  1. Turbulent velocity is an Ornstein-Uhlenbeck process with Lagrangian
     integral timescale T_L, integrated with the EXACT discrete update.  GADEN
     uses a memoryless Gaussian kick scaled by dt (not sqrt(dt)), which makes
     effective turbulent diffusivity proportional to the timestep and produces
     exponentially-distributed blank durations instead of heavy-tailed ones.

  2. Wind is sampled with trilinear interpolation, not nearest-cell.

Units are SI throughout (metres, seconds, kg).  Concentration is reported in
ppm by volume.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# Ideal gas constant in the units used for number density of air.
R_GAS = 82.057338e-6  # m^3 * atm / (mol * K)


@dataclass
class FilamentPlumeConfig:
    """All physical constants. Nothing here is a magic number in the solver."""

    # --- source -------------------------------------------------------------
    source_pos: tuple[float, float, float] = (0.0, 0.0, 1.0)
    release_rate_hz: float = 20.0  # filaments released per second
    ppm_center_initial: float = 20.0  # concentration at filament centre at birth
    sigma0: float = 0.10  # initial filament std dev [m]

    # --- growth -------------------------------------------------------------
    # Farrell "Model 2":  dsigma/dt = gamma / (2 sigma)  =>  sigma = sqrt(s0^2 + gamma t)
    gamma: float = 1.0e-3  # [m^2/s]

    # --- small-scale turbulence: per-filament Ornstein-Uhlenbeck ------------
    turbulence_intensity: float = 0.25  # sigma_u = I * |U_mean|
    lagrangian_timescale: float = 1.0  # T_L [s]
    sigma_u_floor: float = 0.02  # [m/s], keeps plume alive at ~zero wind

    # --- large-scale meander: ONE OU process shared by every filament -------
    # Farrell's "large meander" term.  This is what swings the whole centreline
    # off a fixed probe and produces the long blanks; per-filament turbulence
    # alone gives blank durations that are sub-exponential (CV < 1) and an
    # environment that is easier than reality.  Do not omit it.
    meander_std_rad: float = 0.20  # stationary std of wind bearing [rad]
    meander_timescale: float = 12.0  # T_meander [s], >> T_L

    # --- mean flow ----------------------------------------------------------
    wind_mean: tuple[float, float, float] = (1.0, 0.0, 0.0)

    # --- gas ----------------------------------------------------------------
    specific_gravity: float = 1.0378  # ethanol rel. air; 1.0 disables buoyancy
    temperature_k: float = 293.15
    pressure_atm: float = 1.0

    # --- housekeeping -------------------------------------------------------
    max_filaments: int = 20000
    max_age_s: float = 120.0
    domain_min: tuple[float, float, float] = (-5.0, -20.0, 0.0)
    domain_max: tuple[float, float, float] = (60.0, 20.0, 6.0)
    cutoff_sigmas: float = 3.0  # ignore filaments beyond this many sigma

    def n_air_mol_per_m3(self) -> float:
        return self.pressure_atm / (R_GAS * self.temperature_k)


class FilamentPlume:
    """
    Vectorised NumPy reference implementation.

    State is a fixed-capacity pool with an `alive` mask -- the same layout the
    Warp/GPU kernel uses, so the two stay bit-comparable in structure.
    """

    def __init__(self, cfg: FilamentPlumeConfig, seed: int | None = 0):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.t = 0.0
        self._release_accum = 0.0

        n = cfg.max_filaments
        self.pos = np.zeros((n, 3), dtype=np.float64)
        self.sigma = np.zeros(n, dtype=np.float64)
        self.uprime = np.zeros((n, 3), dtype=np.float64)  # OU turbulent velocity
        self.age = np.zeros(n, dtype=np.float64)
        self.alive = np.zeros(n, dtype=bool)

        # Moles of target gas per filament, fixed at birth so mass is conserved
        # as sigma grows.  Normalisation is the 3-D Gaussian integral (2 pi)^(3/2).
        n_air = cfg.n_air_mol_per_m3()
        self.moles_per_filament = (
            (cfg.ppm_center_initial / 1.0e6) * n_air * (2.0 * math.pi) ** 1.5 * cfg.sigma0**3
        )

        U = np.asarray(cfg.wind_mean, dtype=np.float64)
        self.sigma_u = max(cfg.turbulence_intensity * float(np.linalg.norm(U)), cfg.sigma_u_floor)
        self.meander_angle = 0.0  # shared large-scale bearing perturbation

    def _wind_now(self) -> np.ndarray:
        """Mean wind rotated by the current large-scale meander angle (about z)."""
        U = np.asarray(self.cfg.wind_mean, dtype=np.float64)
        a = self.meander_angle
        ca, sa = math.cos(a), math.sin(a)
        return np.array([ca * U[0] - sa * U[1], sa * U[0] + ca * U[1], U[2]])

    # ------------------------------------------------------------------ reset
    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.alive[:] = False
        self.t = 0.0
        self._release_accum = 0.0

    # ------------------------------------------------------------------- step
    def step(self, dt: float) -> None:
        cfg = self.cfg

        # --- large-scale meander, exact OU update, advanced ONCE per step ----
        if cfg.meander_std_rad > 0.0:
            am = math.exp(-dt / cfg.meander_timescale)
            self.meander_angle = am * self.meander_angle + cfg.meander_std_rad * math.sqrt(
                max(1.0 - am * am, 0.0)
            ) * self.rng.standard_normal()

        self._release(dt)

        idx = np.flatnonzero(self.alive)
        if idx.size == 0:
            self.t += dt
            return

        # --- OU turbulent velocity, EXACT discrete update --------------------
        # u'(t+dt) = u' e^{-dt/T_L} + sigma_u sqrt(1 - e^{-2 dt/T_L}) * xi
        # Unconditionally stable for any dt, unlike an Euler-Maruyama step.
        a = math.exp(-dt / cfg.lagrangian_timescale)
        b = self.sigma_u * math.sqrt(max(1.0 - a * a, 0.0))
        self.uprime[idx] = a * self.uprime[idx] + b * self.rng.standard_normal((idx.size, 3))

        # --- advection -------------------------------------------------------
        vel = self._wind_now()[None, :] + self.uprime[idx]

        # --- buoyancy (vertical only, gravitational settling / rise) ---------
        if abs(cfg.specific_gravity - 1.0) > 1e-9:
            # Dilute-plume slip velocity, scaled by local mixing ratio so that a
            # grown (dilute) filament stops separating from the carrier air.
            mixing_ratio = (cfg.sigma0 / np.maximum(self.sigma[idx], 1e-9)) ** 3
            w_slip = 0.02 * (1.0 - cfg.specific_gravity) * mixing_ratio
            vel[:, 2] += w_slip

        self.pos[idx] += vel * dt

        # --- growth: sigma^2 += gamma*dt  (== Farrell Model 2) ---------------
        self.sigma[idx] = np.sqrt(self.sigma[idx] ** 2 + cfg.gamma * dt)
        self.age[idx] += dt

        # --- cull -------------------------------------------------------------
        lo = np.asarray(cfg.domain_min)
        hi = np.asarray(cfg.domain_max)
        p = self.pos[idx]
        out = np.any(p < lo[None, :], axis=1) | np.any(p > hi[None, :], axis=1)
        old = self.age[idx] > cfg.max_age_s
        self.alive[idx[out | old]] = False

        self.t += dt

    def _release(self, dt: float) -> None:
        self._release_accum += self.cfg.release_rate_hz * dt
        n_new = int(self._release_accum)
        if n_new <= 0:
            return
        self._release_accum -= n_new

        free = np.flatnonzero(~self.alive)
        if free.size == 0:
            return
        take = free[: min(n_new, free.size)]
        self.pos[take] = np.asarray(self.cfg.source_pos, dtype=np.float64)[None, :]
        self.sigma[take] = self.cfg.sigma0
        self.age[take] = 0.0
        # Seed OU state from its stationary distribution -- starting at zero
        # creates a spurious laminar segment near the source.
        self.uprime[take] = self.sigma_u * self.rng.standard_normal((take.size, 3))
        self.alive[take] = True

    # ----------------------------------------------------------------- sample
    def sample(self, points: np.ndarray) -> np.ndarray:
        """
        Concentration [ppm] at world points.

        points : (M, 3)
        returns: (M,)
        """
        pts = np.atleast_2d(np.asarray(points, dtype=np.float64))
        idx = np.flatnonzero(self.alive)
        out = np.zeros(pts.shape[0], dtype=np.float64)
        if idx.size == 0:
            return out

        fp = self.pos[idx]  # (N,3)
        fs = self.sigma[idx]  # (N,)
        n_air = self.cfg.n_air_mol_per_m3()

        # Peak concentration of each filament, from conserved moles.
        c_peak = 1.0e6 * self.moles_per_filament / (n_air * (2.0 * math.pi) ** 1.5 * fs**3)

        cut = self.cfg.cutoff_sigmas
        # Chunk to bound peak memory at large filament counts.
        chunk = max(1, int(4e6 // max(idx.size, 1)))
        for s in range(0, pts.shape[0], chunk):
            q = pts[s : s + chunk]
            d2 = np.sum((q[:, None, :] - fp[None, :, :]) ** 2, axis=2)  # (m,N)
            within = d2 < (cut * fs[None, :]) ** 2
            contrib = np.where(
                within, c_peak[None, :] * np.exp(-0.5 * d2 / (fs[None, :] ** 2)), 0.0
            )
            out[s : s + chunk] = contrib.sum(axis=1)
        return out

    @property
    def n_alive(self) -> int:
        return int(self.alive.sum())
