"""
GPU filament plume, NVIDIA Warp. Batched over parallel environments.

Implements the uniform-wind, single-species, legacy point-source subset of
plume/filament.py. Unsupported physics is rejected explicitly. See
docs/TRANSPORT.md and tests/test_transport_contract.py for the shared contract.

Layout is a fixed-capacity pool with an `alive` mask, shape (n_envs, capacity).
No dynamic allocation: compaction is a CPU idiom and stalls the GPU.

Warp ships inside Isaac Sim as `omni.warp.core`, so this adds no dependency.
`wp.from_torch` / `wp.to_torch` are zero-copy on CUDA, so sensor output reaches
the policy tensor without a device round trip.
"""

from __future__ import annotations

import math

import numpy as np
import warp as wp

from ..chemistry.registry import DEFAULT_REGISTRY

wp.init()

TWO_PI_32 = wp.constant(6.2831853071795864769)
SQRT_2PI_CUBED = wp.constant(15.749609945722419)  # (2 pi)^(3/2)


# ---------------------------------------------------------------- kernels ---
@wp.kernel
def k_advect(
    pos: wp.array2d(dtype=wp.vec3),
    sigma: wp.array2d(dtype=wp.float32),
    uprime: wp.array2d(dtype=wp.vec3),
    age: wp.array2d(dtype=wp.float32),
    alive: wp.array2d(dtype=wp.int32),
    wind: wp.array(dtype=wp.vec3),  # per-env, already meander-rotated
    dt: wp.float32,
    ou_a: wp.float32,  # exp(-dt/T_L)
    ou_b: wp.float32,  # sigma_u * sqrt(1 - a^2)
    gamma: wp.float32,
    max_age: wp.float32,
    dmin: wp.vec3,
    dmax: wp.vec3,
    seed: wp.int32,
    step: wp.array(dtype=wp.int32),
    counts: wp.array2d(dtype=wp.int64),
):
    e, i = wp.tid()
    if alive[e, i] == 0:
        return

    # --- small-scale turbulence: OU, EXACT discrete update -------------------
    # u' <- a u' + sigma_u sqrt(1-a^2) xi.  Unconditionally stable for any dt.
    # An Euler-Maruyama step, or a kick scaled by dt instead of sqrt(dt), makes
    # effective diffusivity depend on the timestep. Do not "simplify" this.
    st = wp.rand_init(seed, (e * 1000003 + i) * 7919 + step[e])
    up = uprime[e, i]
    up = ou_a * up + ou_b * wp.vec3(wp.randn(st), wp.randn(st), wp.randn(st))
    uprime[e, i] = up

    p = pos[e, i] + (wind[e] + up) * dt
    pos[e, i] = p

    # --- growth: sigma^2 += gamma dt   (Farrell "Model 2") -------------------
    s = sigma[e, i]
    sigma[e, i] = wp.sqrt(s * s + gamma * dt)

    a = age[e, i] + dt
    age[e, i] = a

    if (
        p[0] < dmin[0] or p[1] < dmin[1] or p[2] < dmin[2]
        or p[0] > dmax[0] or p[1] > dmax[1] or p[2] > dmax[2]
    ):
        alive[e, i] = 0
        wp.atomic_add(counts, e, 3, wp.int64(1))
    elif a > max_age:
        alive[e, i] = 0
        wp.atomic_add(counts, e, 4, wp.int64(1))


@wp.kernel
def k_release(
    pos: wp.array2d(dtype=wp.vec3),
    sigma: wp.array2d(dtype=wp.float32),
    uprime: wp.array2d(dtype=wp.vec3),
    age: wp.array2d(dtype=wp.float32),
    alive: wp.array2d(dtype=wp.int32),
    cursor: wp.array(dtype=wp.int32),  # per-env round-robin write head
    source: wp.array(dtype=wp.vec3),
    n_new: wp.array(dtype=wp.int32),
    sigma0: wp.float32,
    sigma_u: wp.float32,
    capacity: wp.int32,
    seed: wp.int32,
    step: wp.array(dtype=wp.int32),
    counts: wp.array2d(dtype=wp.int64),
):
    # One writer per environment scans at most capacity slots. Never overwrite
    # live gas, and never launch n_new threads onto modulo-aliased slots.
    e = wp.tid()
    requested = n_new[e]
    if requested <= 0:
        return
    accepted = int(0)
    start = cursor[e]
    for j in range(capacity):
        if accepted >= requested:
            break
        slot = (start + j) % capacity
        if alive[e, slot] == 0:
            st = wp.rand_init(seed + 104729, (e * 1000003 + slot) * 6871 + step[e])
            pos[e, slot] = source[e]
            sigma[e, slot] = sigma0
            age[e, slot] = 0.0
            uprime[e, slot] = sigma_u * wp.vec3(wp.randn(st), wp.randn(st), wp.randn(st))
            alive[e, slot] = 1
            cursor[e] = (slot + 1) % capacity
            accepted += 1
    counts[e, 0] += wp.int64(requested)
    counts[e, 1] += wp.int64(accepted)
    counts[e, 2] += wp.int64(requested - accepted)


@wp.kernel
def k_sample(
    pos: wp.array2d(dtype=wp.vec3),
    sigma: wp.array2d(dtype=wp.float32),
    alive: wp.array2d(dtype=wp.int32),
    probes: wp.array(dtype=wp.vec3),  # one probe per env
    out: wp.array(dtype=wp.float32),
    moles_per_filament: wp.float32,
    inv_n_air: wp.float32,
    cutoff: wp.float32,
    capacity: wp.int32,
    background: wp.float32,
):
    """
    One thread per (env, probe).  Brute-force reduction over the filament pool.

    With a few thousand filaments and a handful of sensors this beats building
    an acceleration structure -- the pool fits in cache and there is no
    build cost per step.  Switch to wp.HashGrid only for dense field
    visualisation, not for sensor queries.
    """
    e = wp.tid()
    q = probes[e]
    acc = background
    for i in range(capacity):
        if alive[e, i] == 0:
            continue
        s = sigma[e, i]
        d = pos[e, i] - q
        d2 = wp.dot(d, d)
        r = cutoff * s
        if d2 < r * r:
            peak = 1.0e6 * moles_per_filament * inv_n_air / (SQRT_2PI_CUBED * s * s * s)
            acc += peak * wp.exp(-0.5 * d2 / (s * s))
    out[e] = acc


# ------------------------------------------------------------------ driver ---
class WarpFilamentPlume:
    """Batched common subset; float32 physics and independent environment state."""

    def __init__(self, cfg, n_envs: int = 1, device: str | None = None, seed: int = 0,
                 registry=None, airflow=None, occupancy=None):
        cfg.validate()
        if isinstance(n_envs, bool) or not isinstance(n_envs, (int, np.integer)) or n_envs <= 0:
            raise ValueError("n_envs must be a positive integer")
        if cfg.emitters is not None or airflow is not None or occupancy is not None:
            raise NotImplementedError("Warp supports legacy point sources and uniform wind only; "
                                      "use FilamentPlume for emitters, airflow, or occupancy")
        if cfg.buoyancy_model != "none":
            raise NotImplementedError("Warp does not implement buoyancy; use FilamentPlume")
        registry = registry or DEFAULT_REGISTRY
        species = registry.get(cfg.species)
        if species.decay_rate_per_s != 0:
            raise NotImplementedError("Warp does not implement decay; use FilamentPlume")
        self.species_names = [species.name]
        background = cfg.canonical_background(registry)
        if background.keys() - {species.name}:
            raise NotImplementedError("Warp background must match its single source species; "
                                      "use FilamentPlume for multiple species")
        self.background_ppm = background.get(species.name, 0.0)
        self.cfg = cfg
        self.n_envs = n_envs
        self.device = device or ("cuda" if wp.get_cuda_device_count() else "cpu")
        self.seed = seed
        self.capacity = cfg.max_filaments
        self.step_count = 0
        self.t = 0.0
        self._release_accum = np.zeros(n_envs)
        self.env_time_s = np.zeros(n_envs)
        self._rng_steps = np.zeros(n_envs, dtype=np.int32)

        n, c = n_envs, self.capacity
        d = self.device
        self.pos = wp.zeros((n, c), dtype=wp.vec3, device=d)
        self.sigma = wp.zeros((n, c), dtype=wp.float32, device=d)
        self.uprime = wp.zeros((n, c), dtype=wp.vec3, device=d)
        self.age = wp.zeros((n, c), dtype=wp.float32, device=d)
        self.alive = wp.zeros((n, c), dtype=wp.int32, device=d)
        self.cursor = wp.zeros(n, dtype=wp.int32, device=d)
        self.out = wp.zeros(n, dtype=wp.float32, device=d)
        self._steps = wp.zeros(n, dtype=wp.int32, device=d)
        self._n_new = wp.zeros(n, dtype=wp.int32, device=d)
        # emitted, accepted, rejected, domain-culled, age-culled packet counts
        self._counts = wp.zeros((n, 5), dtype=wp.int64, device=d)

        src = np.tile(np.asarray(cfg.source_pos, np.float32), (n, 1))
        self.source = wp.array(src, dtype=wp.vec3, device=d)
        self.wind = wp.array(np.tile(cfg.wind_mean, (n, 1)).astype(np.float32),
                             dtype=wp.vec3, device=d)
        self.probes = wp.zeros(n, dtype=wp.vec3, device=d)

        n_air = cfg.n_air_mol_per_m3()
        self.inv_n_air = 1.0 / n_air
        self.moles_per_filament = cfg.build_emitters()[0].moles_per_filament(n_air)
        U = np.asarray(cfg.wind_mean, np.float64)
        self.sigma_u = max(
            cfg.turbulence_intensity * float(np.linalg.norm(U)), cfg.sigma_u_floor
        )
        # Large-scale meander: one shared OU bearing per env, on host (scalar,
        # negligible cost, and keeps the RNG stream reproducible).
        self.meander = np.zeros(n_envs, np.float64)
        self._mrng = [np.random.default_rng(seed + 991 + e) for e in range(n_envs)]

    def set_probes(self, p: np.ndarray) -> None:
        if np.shape(p) != (self.n_envs, 3) or not np.isfinite(p).all():
            raise ValueError("probes must be finite with shape (n_envs, 3)")
        self.probes.assign(np.ascontiguousarray(p, dtype=np.float32))

    def step(self, dt: float) -> None:
        n, h = self.cfg.substeps(dt)
        if not math.isfinite(self.t + dt):
            raise ValueError("plume time must remain finite")
        if self.cfg.release_rate_hz * h + 1 > np.iinfo(np.int32).max:
            raise ValueError("too many releases per internal step for Warp int32 counts")
        for _ in range(n):
            self._step(h)

    def _step(self, dt: float) -> None:
        cfg = self.cfg
        # --- large-scale meander, advanced once per step -------------------
        if cfg.meander_std_rad > 0.0:
            am = math.exp(-dt / cfg.meander_timescale)
            self.meander = am * self.meander + cfg.meander_std_rad * math.sqrt(
                max(1.0 - am * am, 0.0)
            ) * np.array([rng.standard_normal() for rng in self._mrng])
        U = np.asarray(cfg.wind_mean, np.float64)
        ca, sa = np.cos(self.meander), np.sin(self.meander)
        w = np.stack([ca * U[0] - sa * U[1], sa * U[0] + ca * U[1], np.full(self.n_envs, U[2])], 1)
        self.wind.assign(np.ascontiguousarray(w, dtype=np.float32))

        # --- release --------------------------------------------------------
        self._release_accum += cfg.release_rate_hz * dt
        tol = 8 * (np.spacing(np.maximum(1.0, self._release_accum)) +
                   cfg.release_rate_hz * np.spacing(self.env_time_s + dt))
        n_new = np.floor(self._release_accum + tol).astype(np.int32)
        self._steps.assign(self._rng_steps)
        if np.any(n_new > 0):
            self._release_accum = np.maximum(0.0, self._release_accum - n_new)
            self._n_new.assign(n_new)
            wp.launch(
                k_release, dim=self.n_envs,
                inputs=[self.pos, self.sigma, self.uprime, self.age, self.alive,
                        self.cursor, self.source, self._n_new, cfg.sigma0, self.sigma_u,
                        self.capacity, self.seed, self._steps, self._counts],
                device=self.device,
            )

        # --- advect ---------------------------------------------------------
        a = math.exp(-dt / cfg.lagrangian_timescale)
        b = self.sigma_u * math.sqrt(max(1.0 - a * a, 0.0))
        wp.launch(
            k_advect, dim=(self.n_envs, self.capacity),
            inputs=[self.pos, self.sigma, self.uprime, self.age, self.alive, self.wind,
                    float(dt), float(a), float(b), float(cfg.gamma), float(cfg.max_age_s),
                    wp.vec3(*[float(v) for v in cfg.domain_min]),
                    wp.vec3(*[float(v) for v in cfg.domain_max]),
                    self.seed, self._steps, self._counts],
            device=self.device,
        )
        self.step_count += 1
        self._rng_steps += 1
        self.env_time_s += dt
        self.t += dt

    def sample(self) -> np.ndarray:
        wp.launch(
            k_sample, dim=self.n_envs,
            inputs=[self.pos, self.sigma, self.alive, self.probes, self.out,
                    float(self.moles_per_filament), float(self.inv_n_air),
                    float(self.cfg.cutoff_sigmas), self.capacity, float(self.background_ppm)],
            device=self.device,
        )
        return self.out.numpy()

    @property
    def n_alive(self) -> np.ndarray:
        return self.alive.numpy().sum(axis=1)

    # ------------------------------------------------------ torch interop ---
    # Zero-copy on CUDA. On CPU Warp falls back to a copy, which is fine --
    # the CPU path exists for CI parity, not for throughput.
    def set_probes_torch(self, p) -> None:
        import torch
        if tuple(p.shape) != (self.n_envs, 3) or not bool(torch.isfinite(p).all()):
            raise ValueError("probes must be finite with shape (n_envs, 3)")
        p = p.to(dtype=torch.float32)
        wp.copy(self.probes, wp.from_torch(p.contiguous(), dtype=wp.vec3))

    def sample_torch(self):
        """(n_envs, 1) ppm as a torch tensor on the sim device.

        Shaped (n, S) for forward compatibility with multi-species; S == 1
        until the per-species pool lands. Returning (n,) here would force a
        breaking change on every caller later.
        """
        wp.launch(
            k_sample, dim=self.n_envs,
            inputs=[self.pos, self.sigma, self.alive, self.probes, self.out,
                    float(self.moles_per_filament), float(self.inv_n_air),
                    float(self.cfg.cutoff_sigmas), self.capacity, float(self.background_ppm)],
            device=self.device,
        )
        return wp.to_torch(self.out).unsqueeze(-1)

    def wind_torch(self):
        """Simulated anemometer: (n_envs, 3) local wind."""
        return wp.to_torch(self.wind)

    def total_moles(self) -> np.ndarray:
        return (self.n_alive * self.moles_per_filament)[:, None]

    def diagnostics(self) -> dict:
        """Same keys as NumPy; arrays have shape (n_envs, 1). Host snapshot."""
        counts = self._counts.numpy()
        out = {"species_names": tuple(self.species_names), "retained_mol": self.total_moles()}
        for col, name in enumerate(("emitted", "released", "pool_rejected", "domain", "age")):
            out[name + "_mol"] = counts[:, col:col+1] * self.moles_per_filament
            if col < 3:
                out[name + "_filaments"] = counts[:, col:col+1].copy()
        out["outlet_mol"] = np.zeros((self.n_envs, 1))
        out["decayed_mol"] = np.zeros((self.n_envs, 1))
        out["pending_mol"] = self._release_accum[:, None] * self.moles_per_filament
        out["balance_error_mol"] = out["emitted_mol"] - sum(out[k] for k in (
            "retained_mol", "pool_rejected_mol", "decayed_mol", "outlet_mol", "domain_mol", "age_mol"))
        return out

    def reset(self, env_ids=None, seed: int | None = None) -> None:
        """Clear selected environments; seed replays a FULL reset only.

        Unseeded resets continue random streams. Partial resets clear fractional
        releases and diagnostics without disturbing any other environment.
        Source/probe positions and configuration are retained.
        """
        if seed is not None and env_ids is not None:
            raise ValueError("seed is supported only for a full reset (env_ids=None)")
        idx = np.arange(self.n_envs) if env_ids is None else np.asarray(list(env_ids))
        if idx.size == 0:
            return
        if (idx.ndim != 1 or not np.issubdtype(idx.dtype, np.integer) or
                np.any(idx < 0) or np.any(idx >= self.n_envs)):
            raise ValueError("env_ids must contain valid integer environment indices")
        if seed is not None:
            self.seed = seed
            self._rng_steps[:] = 0
            self._mrng = [np.random.default_rng(seed + 991 + e) for e in range(self.n_envs)]
        for state in (self.pos, self.sigma, self.uprime, self.age, self.alive, self.cursor,
                      self.out, self._counts, self._n_new):
            a = state.numpy()
            a[idx] = 0
            state.assign(a)
        self.meander[idx] = 0.0
        self._release_accum[idx] = 0.0
        self.env_time_s[idx] = 0.0
        wind = self.wind.numpy()
        wind[idx] = self.cfg.wind_mean
        self.wind.assign(wind)
        if env_ids is None:
            self.step_count = 0
            self.t = 0.0
