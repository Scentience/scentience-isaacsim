"""
Airflow fields.

The transport engine asks one question: velocity at these points, now.
Everything else -- meander, turbulence, CFD import -- is an implementation
detail of the field.  Fields own their time evolution via `step(dt)`.

Division of labour with the plume (documented because it is easy to get
wrong): the LARGE-scale meander and the field's mean structure live HERE;
the SMALL-scale per-filament Ornstein-Uhlenbeck turbulence lives in the
transport, because it is per-filament state.  Putting meander per filament
averages it out of existence; putting per-filament turbulence in the field
would correlate all filaments.  Both mistakes produce a plume that fails the
realism gate (sub-exponential blank durations).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


class AirflowField:
    def step(self, dt: float) -> None:  # noqa: B027
        pass

    def velocity(self, points: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def reset(self, seed: int | None = None) -> None:  # noqa: B027
        pass


@dataclass
class UniformAirflow(AirflowField):
    """Uniform mean wind with an Ornstein-Uhlenbeck bearing meander.

    The meander is ONE shared process (rotation about +z), the exact discrete
    OU update, unconditionally stable for any dt. This is the component whose
    ablation drops blank-duration CV from 2.31 to 0.96 -- see
    tests/test_plume_gate.py::test_meander_ablation_fails_the_gate.
    """
    mean: tuple[float, float, float] = (1.0, 0.0, 0.0)
    meander_std_rad: float = 0.22
    meander_timescale_s: float = 15.0
    seed: int = 0

    def __post_init__(self):
        self.reset(self.seed)

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._angle = 0.0

    def step(self, dt: float) -> None:
        if self.meander_std_rad <= 0.0:
            return
        a = math.exp(-dt / self.meander_timescale_s)
        self._angle = a * self._angle + self.meander_std_rad * math.sqrt(
            max(1.0 - a * a, 0.0)) * self._rng.standard_normal()

    def velocity(self, points: np.ndarray) -> np.ndarray:
        n = np.atleast_2d(points).shape[0]
        U = np.asarray(self.mean, np.float64)
        ca, sa = math.cos(self._angle), math.sin(self._angle)
        v = np.array([ca * U[0] - sa * U[1], sa * U[0] + ca * U[1], U[2]])
        return np.tile(v, (n, 1))


@dataclass
class GridAirflow(AirflowField):
    """Precomputed velocity field on a regular grid, trilinear interpolation.

    This is the import path for CFD (OpenFOAM et al.): resample the solver
    output onto a regular grid offline, save as .npz, load here. Trilinear --
    never nearest-cell -- because nearest-cell advection imprints grid-aligned
    artefacts on every filament trajectory.
    """
    origin: np.ndarray = None
    cell_size: float = 0.25
    u: np.ndarray = None  # (Nx, Ny, Nz, 3)

    @classmethod
    def from_npz(cls, path: str) -> "GridAirflow":
        d = np.load(path)
        return cls(origin=d["origin"], cell_size=float(d["cell_size"]), u=d["u"])

    def velocity(self, points: np.ndarray) -> np.ndarray:
        p = (np.atleast_2d(points) - self.origin) / self.cell_size - 0.5
        dims = np.asarray(self.u.shape[:3])
        i0 = np.floor(p).astype(int)
        f = p - i0
        i0 = np.clip(i0, 0, dims - 2)
        f = np.clip(f, 0.0, 1.0)
        out = np.zeros((p.shape[0], 3))
        for dx in (0, 1):
            for dy in (0, 1):
                for dz in (0, 1):
                    w = (np.where(dx, f[:, 0], 1 - f[:, 0]) *
                         np.where(dy, f[:, 1], 1 - f[:, 1]) *
                         np.where(dz, f[:, 2], 1 - f[:, 2]))
                    out += w[:, None] * self.u[i0[:, 0] + dx, i0[:, 1] + dy, i0[:, 2] + dz]
        return out


def potential_flow(occ, mean_wind: tuple[float, float, float],
                   n_iter: int = 600) -> GridAirflow:
    """
    Obstacle-aware mean flow without CFD: solve the Laplace equation for the
    velocity potential phi on the occupancy grid, u = grad(phi).

    Boundary conditions: far-field phi = U . x on the domain boundary
    (uniform inflow), homogeneous Neumann on obstacle cells (no penetration),
    enforced by averaging over free neighbours only in the Jacobi sweep.

    What you get: smooth, curl-free, divergence-free flow that goes AROUND
    obstacles. What you do not get: wakes, separation, recirculation -- the
    interesting parts of real indoor flow. Documented honestly in
    CHEMICAL_MODEL.md; use RANS import when those matter.
    """
    from ..geometry.occupancy import OBSTACLE

    g = occ.grid
    dims = np.asarray(g.shape)
    U = np.asarray(mean_wind, np.float64)
    idx = np.stack(np.meshgrid(*[np.arange(d) for d in dims], indexing="ij"), -1)
    x = occ.origin + (idx + 0.5) * occ.cell_size
    phi = x @ U  # initial guess = far-field potential
    free = g != OBSTACLE

    fixed = np.zeros(g.shape, bool)  # Dirichlet on the domain boundary
    fixed[0, :, :] = fixed[-1, :, :] = True
    fixed[:, 0, :] = fixed[:, -1, :] = True
    fixed[:, :, 0] = fixed[:, :, -1] = True
    phi_b = phi.copy()

    for _ in range(n_iter):
        acc = np.zeros_like(phi)
        cnt = np.zeros_like(phi)
        for ax in range(3):
            for sh in (1, -1):
                nb_phi = np.roll(phi, sh, axis=ax)
                nb_free = np.roll(free, sh, axis=ax)
                # roll wraps around; kill wrapped contributions
                sl = [slice(None)] * 3
                sl[ax] = 0 if sh == 1 else -1
                nb_free = nb_free.copy()
                nb_free[tuple(sl)] = False
                acc += np.where(nb_free, nb_phi, 0.0)
                cnt += nb_free
        upd = np.divide(acc, np.maximum(cnt, 1), out=np.zeros_like(acc), where=cnt > 0)
        phi = np.where(free & ~fixed & (cnt > 0), upd, phi)
        phi[fixed] = phi_b[fixed]

    # Gradient with Neumann faces: where a neighbour is an obstacle, substitute
    # the centre value so the difference across that face is zero (no
    # penetration). Naive central differences read phi inside obstacles and
    # produce spurious near-wall velocity spikes.
    u = np.zeros((*g.shape, 3))
    h = 2.0 * occ.cell_size
    for ax in range(3):
        fwd_phi, fwd_free = np.roll(phi, -1, axis=ax), np.roll(free, -1, axis=ax)
        bwd_phi, bwd_free = np.roll(phi, 1, axis=ax), np.roll(free, 1, axis=ax)
        u[..., ax] = (np.where(fwd_free, fwd_phi, phi) -
                      np.where(bwd_free, bwd_phi, phi)) / h
    u[~free] = 0.0
    # kill the one-cell rim where central differences wrapped
    for ax in range(3):
        sl = [slice(None)] * 3
        for edge in (0, -1):
            sl_edge = list(sl)
            sl_edge[ax] = edge
            u[tuple(sl_edge)] = U  # boundary cells: far-field wind
    return GridAirflow(origin=occ.origin, cell_size=occ.cell_size, u=u)
