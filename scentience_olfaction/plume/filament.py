"""
Filament-based Lagrangian plume model (Farrell et al. 2002, Env. Fluid
Mech. 2:143-169), multi-species, obstacle-aware.

Independent implementation from the published equations.  NOT derived from
GADEN source (GADEN is LGPL-3.0); GADEN is used only as a behavioural
cross-check.  Design notes and deviations are documented in
docs/CHEMICAL_MODEL.md.

Two deliberate corrections relative to common practice:

  1. Small-scale turbulent velocity is an Ornstein-Uhlenbeck process with
     Lagrangian timescale T_L, integrated with the EXACT discrete update
     (u' <- a u' + sigma_u sqrt(1-a^2) xi, a = exp(-dt/T_L)).  A memoryless
     kick scaled by dt makes turbulent diffusivity depend on the timestep.
  2. A shared large-scale bearing meander (one OU process, advanced once per
     step) produces the heavy-tailed blank durations real plumes have.
     Ablating it drops blank-duration CV from 2.31 to 0.96 -- see
     tests/test_plume_gate.py.

Units are SI (m, s, kg); concentrations reported in ppm by volume.
This NumPy implementation is the SPECIFICATION; transport/filament_warp.py is
the fast path and tests/test_warp_parity.py keeps them honest.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..airflow.fields import AirflowField, UniformAirflow
from ..chemistry.registry import DEFAULT_REGISTRY, SpeciesRegistry
from ..emitters.emitters import PointEmitter

R_GAS = 82.057338e-6  # m^3 atm / (mol K)
GAUSS3D = (2.0 * math.pi) ** 1.5  # normalisation of an isotropic 3-D Gaussian


@dataclass
class FilamentPlumeConfig:
    """Physical constants. Nothing in the solver is a magic number."""

    # --- sources ------------------------------------------------------------
    emitters: list = None
    """List of PointEmitter/LineEmitter/BoxEmitter. If None, a single
    PointEmitter is built from the legacy single-source fields below."""

    # legacy single-source convenience (kept for API stability)
    source_pos: tuple[float, float, float] = (0.0, 0.0, 1.0)
    release_rate_hz: float = 20.0
    ppm_center_initial: float = 20.0
    sigma0: float = 0.10
    species: str = "ethanol"

    # --- growth -------------------------------------------------------------
    gamma: float = 1.0e-3
    """Farrell 'Model 2' growth: dsigma/dt = gamma/(2 sigma). [m^2/s]
    Farrell reports 1e-3 for outdoor pheromone plumes; evidence: ASSUMED for
    indoor scenes until fitted (see provenance in docs)."""

    # --- small-scale turbulence (per filament OU) ---------------------------
    turbulence_intensity: float = 0.25
    lagrangian_timescale: float = 1.0
    sigma_u_floor: float = 0.02

    # --- mean flow / large-scale meander ------------------------------------
    wind_mean: tuple[float, float, float] = (1.0, 0.0, 0.0)
    meander_std_rad: float = 0.20
    meander_timescale: float = 12.0

    # --- environment --------------------------------------------------------
    temperature_k: float = 293.15
    pressure_atm: float = 1.0

    # --- buoyancy -----------------------------------------------------------
    buoyancy_model: str = "none"
    """'none' (default) or 'slip'. The slip model is a phenomenological
    dilution-scaled settling/rise term, NOT derived physics -- it exists so
    dense (CO2) or light (H2, CH4) plumes are not pretending to be neutrally
    buoyant, and it is off by default precisely because it is undefended.
    See CHEMICAL_MODEL.md 'What is deliberately not modelled'."""
    slip_speed_scale: float = 0.02

    # --- housekeeping -------------------------------------------------------
    max_filaments: int = 20000
    max_age_s: float = 120.0
    domain_min: tuple[float, float, float] = (-5.0, -20.0, 0.0)
    domain_max: tuple[float, float, float] = (60.0, 20.0, 6.0)
    cutoff_sigmas: float = 3.0

    def n_air_mol_per_m3(self) -> float:
        return self.pressure_atm / (R_GAS * self.temperature_k)

    def build_emitters(self) -> list:
        if self.emitters is not None:
            return list(self.emitters)
        return [PointEmitter(position=self.source_pos, species=self.species,
                             release_rate_hz=self.release_rate_hz,
                             ppm_center_initial=self.ppm_center_initial,
                             sigma0=self.sigma0)]


class FilamentPlume:
    """
    Vectorised NumPy reference implementation.

    State is a fixed-capacity pool with an `alive` mask -- the same layout the
    Warp kernel uses, so the two stay structurally comparable.

    Optional collaborators:
      airflow   : AirflowField (default UniformAirflow built from cfg); owns
                  mean flow + large-scale meander.
      occupancy : OccupancyGrid; enables wall collision (slide) and
                  line-of-sight gating of concentration.
    """

    def __init__(self, cfg: FilamentPlumeConfig, seed: int | None = 0,
                 airflow: AirflowField | None = None, occupancy=None,
                 registry: SpeciesRegistry | None = None):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.registry = registry or DEFAULT_REGISTRY
        self.occupancy = occupancy
        self.airflow = airflow or UniformAirflow(
            mean=cfg.wind_mean, meander_std_rad=cfg.meander_std_rad,
            meander_timescale_s=cfg.meander_timescale,
            seed=(seed or 0) + 991)

        self.emitters = cfg.build_emitters()
        self.species_names = sorted({e.species for e in self.emitters})
        self._spec_index = {s: i for i, s in enumerate(self.species_names)}
        for s in self.species_names:
            self.registry.get(s)  # fail fast on unknown species

        self.t = 0.0
        n = cfg.max_filaments
        self.pos = np.zeros((n, 3))
        self.sigma = np.zeros(n)
        self.uprime = np.zeros((n, 3))
        self.age = np.zeros(n)
        self.moles = np.zeros(n)     # per-filament, so decay and per-emitter
        self.spec = np.zeros(n, np.int32)   # strengths are representable
        self.alive = np.zeros(n, bool)

        U = np.asarray(cfg.wind_mean, np.float64)
        self.sigma_u = max(cfg.turbulence_intensity * float(np.linalg.norm(U)),
                           cfg.sigma_u_floor)
        self._decay = np.array([self.registry.get(s).decay_rate_per_s
                                for s in self.species_names])
        self._sg = np.array([self.registry.get(s).specific_gravity
                             for s in self.species_names])

    # ------------------------------------------------------------------ reset
    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.alive[:] = False
        self.t = 0.0
        self.airflow.reset(seed if seed is None else seed + 991)
        for e in self.emitters:
            e.reset()

    # ------------------------------------------------------------------- step
    def step(self, dt: float) -> None:
        cfg = self.cfg
        self.airflow.step(dt)
        self._release(dt)

        idx = np.flatnonzero(self.alive)
        if idx.size == 0:
            self.t += dt
            return

        # --- small-scale OU turbulence, exact discrete update ----------------
        a = math.exp(-dt / cfg.lagrangian_timescale)
        b = self.sigma_u * math.sqrt(max(1.0 - a * a, 0.0))
        self.uprime[idx] = a * self.uprime[idx] + b * self.rng.standard_normal((idx.size, 3))

        # --- advection --------------------------------------------------------
        vel = self.airflow.velocity(self.pos[idx]) + self.uprime[idx]

        # --- optional buoyancy slip (OFF by default; see config docstring) ----
        if cfg.buoyancy_model == "slip":
            # Phenomenological: slip velocity scaled by dilution (sigma0/sigma)^3
            # so a grown, dilute filament stops separating from the carrier air.
            # This is NOT derived physics -- see CHEMICAL_MODEL.md.
            dil = (cfg.sigma0 / np.maximum(self.sigma[idx], 1e-9)) ** 3
            vel[:, 2] += cfg.slip_speed_scale * (1.0 - self._sg[self.spec[idx]]) * dil

        p_new = self.pos[idx] + vel * dt

        # --- walls: slide + outlet culling ------------------------------------
        if self.occupancy is not None:
            p_new, outlet = self.occupancy.collide_and_slide(self.pos[idx], p_new)
            self.alive[idx[outlet]] = False
        self.pos[idx] = p_new

        # --- growth + decay ----------------------------------------------------
        self.sigma[idx] = np.sqrt(self.sigma[idx] ** 2 + cfg.gamma * dt)
        lam = self._decay[self.spec[idx]]
        if np.any(lam > 0):
            self.moles[idx] *= np.exp(-lam * dt)
        self.age[idx] += dt

        # --- domain / age culling ---------------------------------------------
        lo, hi = np.asarray(cfg.domain_min), np.asarray(cfg.domain_max)
        p = self.pos[idx]
        gone = (np.any(p < lo, 1) | np.any(p > hi, 1) | (self.age[idx] > cfg.max_age_s))
        self.alive[idx[gone]] = False
        self.t += dt

    def _release(self, dt: float) -> None:
        n_air = self.cfg.n_air_mol_per_m3()
        for e in self.emitters:
            n_new = e.n_release(self.t, dt, self.rng)
            if n_new <= 0:
                continue
            free = np.flatnonzero(~self.alive)
            take = free[: min(n_new, free.size)]
            if take.size == 0:
                continue
            self.pos[take] = e.sample_positions(take.size, self.rng)
            self.sigma[take] = e.sigma0
            self.age[take] = 0.0
            self.moles[take] = (e.ppm_center_initial / 1e6) * n_air * GAUSS3D * e.sigma0 ** 3
            self.spec[take] = self._spec_index[e.species]
            # OU seeded from its stationary distribution -- starting at zero
            # creates a spurious laminar segment near the source.
            self.uprime[take] = self.sigma_u * self.rng.standard_normal((take.size, 3))
            self.alive[take] = True

    # ----------------------------------------------------------------- sample
    def sample_species(self, points: np.ndarray) -> np.ndarray:
        """
        Per-species concentration [ppm] at world points.
        points: (M, 3) -> returns (M, S), S = len(self.species_names).

        Superposition with a 3-sigma cutoff; if an occupancy grid is present,
        filaments without line of sight to the query point contribute zero
        (the cheap stand-in for 'gas does not diffuse through walls').
        """
        pts = np.atleast_2d(np.asarray(points, np.float64))
        S = len(self.species_names)
        out = np.zeros((pts.shape[0], S))
        idx = np.flatnonzero(self.alive)
        if idx.size == 0:
            return out

        fp, fs = self.pos[idx], self.sigma[idx]
        fm, fsp = self.moles[idx], self.spec[idx]
        n_air = self.cfg.n_air_mol_per_m3()
        c_peak = 1.0e6 * fm / (n_air * GAUSS3D * fs ** 3)
        cut2 = (self.cfg.cutoff_sigmas * fs) ** 2

        chunk = max(1, int(4e6 // max(idx.size, 1)))
        for s0 in range(0, pts.shape[0], chunk):
            q = pts[s0:s0 + chunk]
            d2 = np.sum((q[:, None, :] - fp[None, :, :]) ** 2, axis=2)
            within = d2 < cut2[None, :]
            if self.occupancy is not None:
                for qi in range(q.shape[0]):
                    cand = np.flatnonzero(within[qi])
                    if cand.size:
                        los = self.occupancy.line_of_sight_batch(q[qi], fp[cand])
                        within[qi, cand[~los]] = False
            contrib = np.where(within,
                               c_peak[None, :] * np.exp(-0.5 * d2 / fs[None, :] ** 2), 0.0)
            for si in range(S):
                m = fsp == si
                out[s0:s0 + chunk, si] = contrib[:, m].sum(axis=1)
        return out

    def sample(self, points: np.ndarray) -> np.ndarray:
        """Total concentration [ppm], summed over species: (M,).
        Kept for API stability with single-species callers."""
        return self.sample_species(points).sum(axis=1)

    def total_moles(self) -> np.ndarray:
        """Per-species moles currently in the domain -- the mass-conservation
        observable the tests audit."""
        out = np.zeros(len(self.species_names))
        idx = np.flatnonzero(self.alive)
        np.add.at(out, self.spec[idx], self.moles[idx])
        return out

    @property
    def n_alive(self) -> int:
        return int(self.alive.sum())
