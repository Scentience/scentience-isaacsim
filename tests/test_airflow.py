import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from scentience_olfaction.airflow.fields import (GridAirflow, UniformAirflow,
                                                 potential_flow)
from scentience_olfaction.geometry.occupancy import OccupancyGrid, OBSTACLE


def test_grid_trilinear_reproduces_linear_field():
    """Trilinear interpolation must be EXACT on a linear field -- the property
    that separates it from nearest-cell lookup."""
    dims = (8, 8, 8)
    origin = np.zeros(3); cs = 0.5
    idx = np.stack(np.meshgrid(*[np.arange(d) for d in dims], indexing="ij"), -1)
    centers = origin + (idx + 0.5) * cs
    u = np.zeros((*dims, 3))
    u[..., 0] = 2.0 * centers[..., 1]          # u_x = 2y, linear
    g = GridAirflow(origin=origin, cell_size=cs, u=u)
    pts = np.array([[1.7, 1.3, 2.0], [2.2, 0.9, 1.1]])
    v = g.velocity(pts)
    assert np.allclose(v[:, 0], 2.0 * pts[:, 1], atol=1e-9)
    assert np.allclose(v[:, 1:], 0.0)


def test_uniform_meander_preserves_speed_and_is_seeded():
    a1 = UniformAirflow(mean=(1.0, 0.0, 0.0), seed=5)
    a2 = UniformAirflow(mean=(1.0, 0.0, 0.0), seed=5)
    for _ in range(200):
        a1.step(0.05); a2.step(0.05)
    v1 = a1.velocity(np.zeros((1, 3)))[0]
    v2 = a2.velocity(np.zeros((1, 3)))[0]
    assert np.allclose(v1, v2), "same seed -> same meander state"
    assert np.isclose(np.linalg.norm(v1), 1.0), "meander rotates, never rescales"


def test_potential_flow_diverts_and_respects_walls():
    occ = OccupancyGrid.from_boxes((-1, -3, 0), (11, 3, 2), 0.25,
                                   boxes=[((5.0, -1.0, 0.0), (5.5, 1.0, 2.0))],
                                   empty_point=(0, 2, 1))
    af = potential_flow(occ, (1.0, 0.0, 0.0), n_iter=500)
    # inside the obstacle: zero velocity
    assert np.allclose(af.velocity(np.array([[5.25, 0.0, 1.0]])), 0.0)
    # beside the obstacle: flow accelerates through the constriction
    v_side = af.velocity(np.array([[5.25, 2.0, 1.0]]))[0]
    v_free = af.velocity(np.array([[1.0, 2.0, 1.0]]))[0]
    assert v_side[0] > 0.5 * v_free[0], "flow should pass around the obstacle"
    # far field is the imposed wind
    assert np.allclose(af.velocity(np.array([[0.0, 0.0, 1.0]]))[0][0], 1.0,
                       atol=0.5)


def test_divergence_small_in_free_space():
    occ = OccupancyGrid.from_boxes((0, 0, 0), (6, 6, 3), 0.5, boxes=[],
                                   empty_point=(3, 3, 1.5), outlet_boundary=False)
    af = potential_flow(occ, (1.0, 0.0, 0.0), n_iter=800)
    u = af.u
    # central-difference divergence over the interior
    div = np.zeros(u.shape[:3])
    h = 2 * 0.5
    for ax in range(3):
        div += (np.roll(u[..., ax], -1, axis=ax) - np.roll(u[..., ax], 1, axis=ax)) / h
    interior = div[2:-2, 2:-2, 1:-1]
    assert np.abs(interior).max() < 0.15, f"max |div u| = {np.abs(interior).max():.3f}"
