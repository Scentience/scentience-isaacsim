"""
GPU filament plume, NVIDIA Warp. Batched over parallel environments.

Physics is identical to plume/filament.py (the NumPy reference); that file is
the specification and this one is the fast path.  `tests/test_warp_parity.py`
asserts they agree.  Never change one without the other.

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
    step: wp.int32,
):
    e, i = wp.tid()
    if alive[e, i] == 0:
        return

    # --- small-scale turbulence: OU, EXACT discrete update -------------------
    # u' <- a u' + sigma_u sqrt(1-a^2) xi.  Unconditionally stable for any dt.
    # An Euler-Maruyama step, or a kick scaled by dt instead of sqrt(dt), makes
    # effective diffusivity depend on the timestep. Do not "simplify" this.
    st = wp.rand_init(seed, (e * 1000003 + i) * 7919 + step)
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
        a > max_age
        or p[0] < dmin[0] or p[1] < dmin[1] or p[2] < dmin[2]
        or p[0] > dmax[0] or p[1] > dmax[1] or p[2] > dmax[2]
    ):
        alive[e, i] = 0


@wp.kernel
def k_release(
    pos: wp.array2d(dtype=wp.vec3),
    sigma: wp.array2d(dtype=wp.float32),
    uprime: wp.array2d(dtype=wp.vec3),
    age: wp.array2d(dtype=wp.float32),
    alive: wp.array2d(dtype=wp.int32),
    cursor: wp.array(dtype=wp.int32),  # per-env round-robin write head
    source: wp.array(dtype=wp.vec3),
    n_new: wp.int32,
    sigma0: wp.float32,
    sigma_u: wp.float32,
    capacity: wp.int32,
    seed: wp.int32,
    step: wp.int32,
):
    e, j = wp.tid()
    if j >= n_new:
        return
    slot = (cursor[e] + j) % capacity
    st = wp.rand_init(seed + 104729, (e * 1000003 + slot) * 6871 + step)
    pos[e, slot] = source[e]
    sigma[e, slot] = sigma0
    age[e, slot] = 0.0
    # Seed OU from its stationary distribution. Starting at zero creates a
    # spurious laminar segment near the source that shows up in the statistics.
    uprime[e, slot] = sigma_u * wp.vec3(wp.randn(st), wp.randn(st), wp.randn(st))
    alive[e, slot] = 1


@wp.kernel
def k_advance_cursor(cursor: wp.array(dtype=wp.int32), n_new: wp.int32, capacity: wp.int32):
    e = wp.tid()
    cursor[e] = (cursor[e] + n_new) % capacity


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
    acc = float(0.0)
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
    """Batched filament plume. Mirrors plume/filament.py exactly."""

    def __init__(self, cfg, n_envs: int = 1, device: str | None = None, seed: int = 0):
        self.cfg = cfg
        self.n_envs = n_envs
        self.device = device or ("cuda" if wp.get_cuda_device_count() else "cpu")
        self.seed = seed
        self.capacity = cfg.max_filaments
        self.step_count = 0
        self.t = 0.0
        self._release_accum = 0.0

        n, c = n_envs, self.capacity
        d = self.device
        self.pos = wp.zeros((n, c), dtype=wp.vec3, device=d)
        self.sigma = wp.zeros((n, c), dtype=wp.float32, device=d)
        self.uprime = wp.zeros((n, c), dtype=wp.vec3, device=d)
        self.age = wp.zeros((n, c), dtype=wp.float32, device=d)
        self.alive = wp.zeros((n, c), dtype=wp.int32, device=d)
        self.cursor = wp.zeros(n, dtype=wp.int32, device=d)
        self.out = wp.zeros(n, dtype=wp.float32, device=d)

        src = np.tile(np.asarray(cfg.source_pos, np.float32), (n, 1))
        self.source = wp.array(src, dtype=wp.vec3, device=d)
        self.wind = wp.zeros(n, dtype=wp.vec3, device=d)
        self.probes = wp.zeros(n, dtype=wp.vec3, device=d)

        n_air = cfg.n_air_mol_per_m3()
        self.inv_n_air = 1.0 / n_air
        self.moles_per_filament = (
            (cfg.ppm_center_initial / 1.0e6) * n_air * (2.0 * math.pi) ** 1.5 * cfg.sigma0**3
        )
        U = np.asarray(cfg.wind_mean, np.float64)
        self.sigma_u = max(
            cfg.turbulence_intensity * float(np.linalg.norm(U)), cfg.sigma_u_floor
        )
        # Large-scale meander: one shared OU bearing per env, on host (scalar,
        # negligible cost, and keeps the RNG stream reproducible).
        self.meander = np.zeros(n_envs, np.float64)
        self._mrng = np.random.default_rng(seed + 991)

    def set_probes(self, p: np.ndarray) -> None:
        self.probes.assign(np.ascontiguousarray(p, dtype=np.float32))

    def step(self, dt: float) -> None:
        cfg = self.cfg
        # --- large-scale meander, advanced once per step -------------------
        if cfg.meander_std_rad > 0.0:
            am = math.exp(-dt / cfg.meander_timescale)
            self.meander = am * self.meander + cfg.meander_std_rad * math.sqrt(
                max(1.0 - am * am, 0.0)
            ) * self._mrng.standard_normal(self.n_envs)
        U = np.asarray(cfg.wind_mean, np.float64)
        ca, sa = np.cos(self.meander), np.sin(self.meander)
        w = np.stack([ca * U[0] - sa * U[1], sa * U[0] + ca * U[1], np.full(self.n_envs, U[2])], 1)
        self.wind.assign(np.ascontiguousarray(w, dtype=np.float32))

        # --- release --------------------------------------------------------
        self._release_accum += cfg.release_rate_hz * dt
        n_new = int(self._release_accum)
        if n_new > 0:
            self._release_accum -= n_new
            wp.launch(
                k_release, dim=(self.n_envs, n_new),
                inputs=[self.pos, self.sigma, self.uprime, self.age, self.alive,
                        self.cursor, self.source, n_new, cfg.sigma0, self.sigma_u,
                        self.capacity, self.seed, self.step_count],
                device=self.device,
            )
            wp.launch(k_advance_cursor, dim=self.n_envs,
                      inputs=[self.cursor, n_new, self.capacity], device=self.device)

        # --- advect ---------------------------------------------------------
        a = math.exp(-dt / cfg.lagrangian_timescale)
        b = self.sigma_u * math.sqrt(max(1.0 - a * a, 0.0))
        wp.launch(
            k_advect, dim=(self.n_envs, self.capacity),
            inputs=[self.pos, self.sigma, self.uprime, self.age, self.alive, self.wind,
                    float(dt), float(a), float(b), float(cfg.gamma), float(cfg.max_age_s),
                    wp.vec3(*[float(v) for v in cfg.domain_min]),
                    wp.vec3(*[float(v) for v in cfg.domain_max]),
                    self.seed, self.step_count],
            device=self.device,
        )
        self.step_count += 1
        self.t += dt

    def sample(self) -> np.ndarray:
        wp.launch(
            k_sample, dim=self.n_envs,
            inputs=[self.pos, self.sigma, self.alive, self.probes, self.out,
                    float(self.moles_per_filament), float(self.inv_n_air),
                    float(self.cfg.cutoff_sigmas), self.capacity],
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
                    float(self.cfg.cutoff_sigmas), self.capacity],
            device=self.device,
        )
        return wp.to_torch(self.out).unsqueeze(-1)

    def wind_torch(self):
        """Simulated anemometer: (n_envs, 3) local wind."""
        return wp.to_torch(self.wind)

    def reset(self, env_ids=None) -> None:
        """Clear filaments. `env_ids=None` resets every environment."""
        a = self.alive.numpy()
        if env_ids is None:
            a[:] = 0
            self.meander[:] = 0.0
            self._release_accum = 0.0
        else:
            idx = np.asarray(list(env_ids), dtype=np.int64)
            a[idx] = 0
            self.meander[idx] = 0.0
        self.alive.assign(a)
