"""
Occupancy grid: the plume's knowledge of where walls are.

Pipeline (mirrors what any dispersion simulator needs, implemented from first
principles -- no third-party dispersion code is used):

  1. triangles -> voxels   separating-axis triangle/box overlap test
                           (Akenine-Moller's 13-axis method, implemented from
                           the published algorithm description)
  2. flood fill            BFS from a user-supplied known-empty point marks
                           FREE; anything unreached stays non-free.  The seed
                           point is REQUIRED -- inferring it guesses wrong on
                           any scene with enclosed volumes.
  3. queries               line_of_sight (DDA ray march) for concentration
                           sampling; collide_and_slide for filament stepping.

Cell states: FREE=0, OBSTACLE=1, OUTLET=2, OUT_OF_BOUNDS handled implicitly.
Filaments entering OUTLET cells are culled by the transport.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

FREE, OBSTACLE, OUTLET = 0, 1, 2


@dataclass
class OccupancyGrid:
    origin: np.ndarray        # (3,) world position of grid[0,0,0] corner
    cell_size: float
    grid: np.ndarray          # (Nx, Ny, Nz) uint8

    # ------------------------------------------------------------ coordinates
    def world_to_cell(self, p: np.ndarray) -> np.ndarray:
        return np.floor((np.atleast_2d(p) - self.origin) / self.cell_size).astype(np.int64)

    def in_bounds(self, c: np.ndarray) -> np.ndarray:
        d = np.asarray(self.grid.shape)
        return np.all((c >= 0) & (c < d), axis=-1)

    def state_at(self, p: np.ndarray) -> np.ndarray:
        """Cell state at world points; out-of-bounds reports OUTLET (gas that
        leaves the domain is gone, which is exactly what an outlet means)."""
        c = self.world_to_cell(p)
        ok = self.in_bounds(c)
        out = np.full(c.shape[0], OUTLET, dtype=np.uint8)
        if ok.any():
            cc = c[ok]
            out[ok] = self.grid[cc[:, 0], cc[:, 1], cc[:, 2]]
        return out

    # -------------------------------------------------------------- builders
    @classmethod
    def from_triangles(cls, tris: np.ndarray, cell_size: float,
                       empty_point: np.ndarray, padding_cells: int = 2,
                       outlet_boundary: bool = True) -> "OccupancyGrid":
        """tris: (T, 3, 3) world-space triangles (collision geometry)."""
        tris = np.asarray(tris, np.float64)
        lo = tris.reshape(-1, 3).min(0) - padding_cells * cell_size
        hi = tris.reshape(-1, 3).max(0) + padding_cells * cell_size
        dims = np.maximum(np.ceil((hi - lo) / cell_size).astype(int), 1)
        g = cls(origin=lo, cell_size=cell_size, grid=np.zeros(dims, np.uint8))
        for t in tris:
            g._rasterize_triangle(t)
        g._flood_fill(np.asarray(empty_point, np.float64))
        if outlet_boundary:
            g._mark_boundary_outlets()
        return g

    @classmethod
    def from_boxes(cls, domain_min, domain_max, cell_size: float,
                   boxes: list[tuple], empty_point=None,
                   outlet_boundary: bool = True) -> "OccupancyGrid":
        """Axis-aligned box obstacles: [(lo, hi), ...]. The fast path for
        tests and procedurally generated scenes."""
        lo = np.asarray(domain_min, np.float64)
        hi = np.asarray(domain_max, np.float64)
        dims = np.maximum(np.ceil((hi - lo) / cell_size).astype(int), 1)
        grid = np.zeros(dims, np.uint8)
        g = cls(origin=lo, cell_size=cell_size, grid=grid)
        # Cell-center sampling: a cell is OBSTACLE if its center is inside a box.
        idx = np.stack(np.meshgrid(*[np.arange(d) for d in dims], indexing="ij"), -1)
        centers = lo + (idx + 0.5) * cell_size
        for blo, bhi in boxes:
            blo, bhi = np.asarray(blo), np.asarray(bhi)
            inside = np.all((centers >= blo) & (centers <= bhi), axis=-1)
            grid[inside] = OBSTACLE
        if empty_point is not None:
            g._flood_fill(np.asarray(empty_point, np.float64))
        if outlet_boundary:
            g._mark_boundary_outlets()
        return g

    # ---------------------------------------------------------- rasterization
    def _rasterize_triangle(self, tri: np.ndarray) -> None:
        cs = self.cell_size
        clo = np.maximum(np.floor((tri.min(0) - self.origin) / cs).astype(int), 0)
        chi = np.minimum(np.floor((tri.max(0) - self.origin) / cs).astype(int),
                         np.asarray(self.grid.shape) - 1)
        if np.any(chi < clo):
            return
        h = cs / 2.0
        for i in range(clo[0], chi[0] + 1):
            for j in range(clo[1], chi[1] + 1):
                for k in range(clo[2], chi[2] + 1):
                    center = self.origin + (np.array([i, j, k]) + 0.5) * cs
                    if _tri_box_overlap(center, h, tri):
                        self.grid[i, j, k] = OBSTACLE

    def _flood_fill(self, empty_point: np.ndarray) -> None:
        seed = self.world_to_cell(empty_point)[0]
        if not self.in_bounds(seed[None, :])[0]:
            raise ValueError(f"empty_point {empty_point} is outside the grid")
        if self.grid[tuple(seed)] == OBSTACLE:
            raise ValueError(
                f"empty_point {empty_point} lands inside an obstacle cell -- "
                "pick a point you know is open air")
        # Everything starts implicitly unreachable: mark non-obstacle cells as
        # tentative-obstacle, then carve out the connected FREE component.
        reach = np.zeros_like(self.grid, dtype=bool)
        q = deque([tuple(seed)])
        reach[tuple(seed)] = True
        dims = self.grid.shape
        while q:
            x, y, z = q.popleft()
            for dx, dy, dz in ((1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)):
                n = (x+dx, y+dy, z+dz)
                if (0 <= n[0] < dims[0] and 0 <= n[1] < dims[1] and 0 <= n[2] < dims[2]
                        and not reach[n] and self.grid[n] != OBSTACLE):
                    reach[n] = True
                    q.append(n)
        self.grid[(self.grid != OBSTACLE) & ~reach] = OBSTACLE

    def _mark_boundary_outlets(self) -> None:
        g = self.grid
        for sl in ((0, slice(None), slice(None)), (-1, slice(None), slice(None)),
                   (slice(None), 0, slice(None)), (slice(None), -1, slice(None)),
                   (slice(None), slice(None), 0), (slice(None), slice(None), -1)):
            face = g[sl]
            face[face == FREE] = OUTLET

    # ---------------------------------------------------------------- queries
    def line_of_sight(self, a: np.ndarray, b: np.ndarray) -> bool:
        """Ray march at half-cell steps; blocked iff any OBSTACLE cell is hit.
        Half-cell stepping can, in principle, tunnel through an exact corner;
        acceptable for concentration gating, do not reuse for physics."""
        a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
        d = b - a
        dist = float(np.linalg.norm(d))
        if dist < 1e-12:
            return True
        n = max(int(dist / (0.5 * self.cell_size)), 1)
        pts = a[None, :] + (np.arange(1, n)[:, None] / n) * d[None, :]
        if len(pts) == 0:
            return True
        c = self.world_to_cell(pts)
        ok = self.in_bounds(c)
        cc = c[ok]
        return not np.any(self.grid[cc[:, 0], cc[:, 1], cc[:, 2]] == OBSTACLE)

    def line_of_sight_batch(self, origin: np.ndarray, targets: np.ndarray) -> np.ndarray:
        return np.array([self.line_of_sight(origin, t) for t in targets], dtype=bool)

    def collide_and_slide(self, p_old: np.ndarray, p_new: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Axis-decomposed slide response for filaments (vectorised).

        For each mover whose destination cell is OBSTACLE, accept each axis
        component independently from the old position; components that would
        enter an obstacle are cancelled. This slides along walls rather than
        sticking to them. It is an approximation of the recursive
        project-and-slide response (documented in CHEMICAL_MODEL.md); exact
        for axis-aligned geometry, approximate near corners.

        Returns (positions, outlet_mask). OUT-of-domain or OUTLET destinations
        report True in outlet_mask (caller culls those filaments).
        """
        p_old = np.atleast_2d(p_old).copy()
        p_new = np.atleast_2d(p_new).copy()
        st = self.state_at(p_new)
        outlet = st == OUTLET
        hit = st == OBSTACLE
        if hit.any():
            idx = np.flatnonzero(hit)
            for ax in range(3):
                trial = p_old[idx].copy()
                trial[:, ax] = p_new[idx, ax]
                blocked = self.state_at(trial) == OBSTACLE
                # cancel the axis move where blocked
                p_new[idx[blocked], ax] = p_old[idx[blocked], ax]
            # a mover fully boxed in stays put (all axes cancelled) -- fine.
            st2 = self.state_at(p_new[idx])
            still = st2 == OBSTACLE
            p_new[idx[still]] = p_old[idx[still]]
            outlet[idx] |= st2 == OUTLET
        return p_new, outlet


# ----------------------------------------------------------------------------
# Triangle/box overlap -- separating axis test, 13 axes (Akenine-Moller 2001,
# "Fast 3D Triangle-Box Overlap Testing"; implemented from the algorithm as
# published, which is a mathematical method and carries no license).
# ----------------------------------------------------------------------------
def _tri_box_overlap(center: np.ndarray, half: float, tri: np.ndarray) -> bool:
    v = tri - center[None, :]

    # 1) box face normals (3 axes) = AABB overlap
    if np.any(v.min(0) > half) or np.any(v.max(0) < -half):
        return False

    # 2) triangle plane
    e = np.array([v[1] - v[0], v[2] - v[1], v[0] - v[2]])
    n = np.cross(e[0], e[1])
    d = -float(n @ v[0])
    r = half * float(np.abs(n).sum())
    if abs(d) > r:  # plane vs box: |n . c + d| vs projected radius
        return False

    # 3) nine cross-axis tests a_ij = e_i x axis_j
    for i in range(3):
        for j in range(3):
            axis = np.zeros(3)
            axis[j] = 1.0
            a = np.cross(e[i], axis)
            if not np.any(a):
                continue
            p = v @ a
            rad = half * float(np.abs(a).sum())
            if p.min() > rad or p.max() < -rad:
                return False
    return True
