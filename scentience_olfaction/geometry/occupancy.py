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
  3. queries               line_of_sight for concentration sampling;
                           swept voxel collision and slide for filament stepping.

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
        """Check endpoints, then march; blocked if an OBSTACLE cell is hit.
        Half-cell stepping can, in principle, tunnel through an exact corner;
        acceptable for concentration gating, do not reuse for physics."""
        a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
        if np.any(self.state_at(np.stack((a, b))) == OBSTACLE):
            return False
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
        Swept voxel collision with sliding, batched over (N, 3) points.

        Traverse every crossed cell, stop just before the first solid face,
        and sweep the remaining tangential displacement. Simultaneous edge /
        corner crossings conservatively check all touched cells. Each contact
        removes at least one displacement axis, so at most three sweeps suffice.

        Returns (positions, outlet_mask). The FIRST outlet or domain exit on
        the actual swept path reports True, even if the requested endpoint is
        free. A wall before an outlet blocks it. Starts in solid cells are
        invalid; starts in outlets are immediately reported for culling.
        """
        start = np.atleast_2d(np.asarray(p_old, dtype=np.float64))
        target = np.atleast_2d(np.asarray(p_new, dtype=np.float64)).copy()
        if (start.ndim != 2 or start.shape[1] != 3 or start.shape != target.shape or
                not np.isfinite(start).all() or not np.isfinite(target).all()):
            raise ValueError("movement endpoints must be finite arrays with matching shape (N, 3)")
        state = self.state_at(start)
        if np.any(state == OBSTACLE):
            raise ValueError("movement cannot start inside an obstacle cell")
        out = start.copy()
        outlet = state == OUTLET
        active = np.flatnonzero(~outlet)
        for _ in range(3):
            if not active.size:
                break
            position, blocked, escaped = self._sweep(out[active], target[active])
            out[active] = position
            outlet[active] = escaped
            # Keep the original tangential destination, but remove every
            # blocked normal component before sweeping the slide itself.
            target[active] = np.where(blocked, position, target[active])
            active = active[blocked.any(axis=1) & ~escaped]
        return out, outlet

    def _sweep(self, start: np.ndarray, target: np.ndarray):
        """First contact for each segment; vectorized voxel DDA, no point sampling."""
        cell = self.world_to_cell(start)
        end_cell = self.world_to_cell(target)
        position = target.copy()
        blocked = np.zeros(start.shape, dtype=bool)
        outlet = np.zeros(start.shape[0], dtype=bool)
        # A voxel is convex: segments contained in one free voxel need no DDA.
        active = np.flatnonzero(np.any(cell != end_cell, axis=1))
        if not active.size:
            return position, blocked, outlet

        delta = target - start
        direction = np.sign(delta).astype(np.int64)
        boundary = self.origin + (cell + (direction > 0)) * self.cell_size
        next_t = np.full(start.shape, np.inf)
        stride = np.full(start.shape, np.inf)
        np.divide(boundary - start, delta, out=next_t, where=delta != 0)
        np.divide(self.cell_size, np.abs(delta), out=stride, where=delta != 0)
        next_t = np.maximum(next_t, 0.0)
        eps = np.finfo(np.float64).eps

        while active.size:
            t = np.min(next_t[active], axis=1)
            # Include endpoint contacts. A handful of ulps covers roundoff in
            # boundary times and conservatively merges near-exact corner ties.
            tol = 16 * eps * np.maximum(1.0, np.abs(t))
            visiting = t <= 1.0 + tol
            active, t, tol = active[visiting], t[visiting], tol[visiting]
            if not active.size:
                break
            crosses = next_t[active] <= t[:, None] + tol[:, None]
            normals = np.zeros(crosses.shape, dtype=bool)
            exits = np.zeros(active.size, dtype=bool)

            # Supercover of ties: for an XY edge inspect X, Y AND XY; for
            # an XYZ vertex inspect all seven neighbours. Advancing only the
            # diagonal cell permits leaks between edge-touching solid voxels.
            for bits in range(1, 8):
                axes = (np.array([1, 2, 4]) & bits) != 0
                rows = np.flatnonzero(np.all(crosses[:, axes], axis=1))
                if not rows.size:
                    continue
                ids = active[rows]
                candidate = cell[ids] + direction[ids] * axes
                inside = self.in_bounds(candidate)
                state = np.full(rows.size, OUTLET, dtype=np.uint8)
                cc = candidate[inside]
                state[inside] = self.grid[cc[:, 0], cc[:, 1], cc[:, 2]]
                # A blocked X face already prevents entering an XY/XYZ
                # neighbour behind it; do not spuriously cancel Y/Z sliding
                # along a flat wall merely because grid planes coincide.
                new_face = (state == OBSTACLE) & ~np.any(normals[rows] & axes, axis=1)
                normals[rows] |= new_face[:, None] & axes
                exits[rows] |= state == OUTLET

            hit = normals.any(axis=1)
            contact = hit | exits
            if contact.any():
                ids = active[contact]
                at = start[ids] + np.minimum(t[contact], 1.0)[:, None] * delta[ids]
                # Solid faces win simultaneous solid/outlet contacts; a later
                # tangential sweep can still enter an unobstructed outlet.
                outlet[ids] = exits[contact] & ~hit[contact]
                blocked[ids] = normals[contact]
                faces = self.origin + (cell[ids] + (direction[ids] > 0)) * self.cell_size
                # Stay representably on the free side even with translated
                # grids, where nextafter alone can round back into the wall
                # during (position-origin)/cell_size conversion.
                skin = 16 * eps * np.maximum(
                    self.cell_size, np.maximum(np.abs(faces), np.abs(self.origin)))
                safe = faces - direction[ids] * skin
                safe = np.where(direction[ids] > 0, np.maximum(start[ids], safe),
                                np.minimum(start[ids], safe))
                position[ids] = np.where(normals[contact], safe, at)
                # A tangential coordinate can already enter an outlet at the
                # same instant another coordinate hits a wall. Do not start a
                # second sweep inside that outlet and pass through it unseen.
                outlet[ids] |= self.state_at(position[ids]) == OUTLET

            advancing = ~contact
            ids = active[advancing]
            cell[ids] += direction[ids] * crosses[advancing]
            next_t[ids] += np.where(crosses[advancing], stride[ids], 0.0)
            active = ids[np.any(cell[ids] != end_cell[ids], axis=1)]
        return position, blocked, outlet


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
