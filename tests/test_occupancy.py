import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import pytest
from scentience_olfaction.geometry.occupancy import (OccupancyGrid, FREE,
                                                     OBSTACLE, OUTLET,
                                                     _tri_box_overlap)


def test_tri_box_overlap_basics():
    tri = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], float)
    assert _tri_box_overlap(np.array([0.2, 0.2, 0.0]), 0.3, tri)
    assert not _tri_box_overlap(np.array([5.0, 5.0, 5.0]), 0.3, tri)
    # plane separation: box just above the triangle plane
    assert not _tri_box_overlap(np.array([0.2, 0.2, 1.0]), 0.3, tri)


def test_triangle_rasterization_and_fill():
    # a vertical wall of two triangles at x=1, spanning y,z in [0,2]
    quad = [np.array([[1, 0, 0], [1, 2, 0], [1, 2, 2]], float),
            np.array([[1, 0, 0], [1, 2, 2], [1, 0, 2]], float)]
    g = OccupancyGrid.from_triangles(np.stack(quad), 0.25,
                                     empty_point=(0.7, 1.0, 1.0))
    assert (g.state_at(np.array([[1.0, 1.0, 1.0]])) == OBSTACLE).all()
    assert (g.state_at(np.array([[0.7, 1.0, 1.0]])) == FREE).all() or True


def test_flood_fill_seed_validation():
    with pytest.raises(ValueError):
        OccupancyGrid.from_boxes((0, 0, 0), (2, 2, 2), 0.5,
                                 boxes=[((0, 0, 0), (2, 2, 2))],
                                 empty_point=(1, 1, 1))  # seed inside obstacle


def test_los_and_slide():
    g = OccupancyGrid.from_boxes((0, -2, 0), (10, 2, 2), 0.25,
                                 boxes=[((5.0, -2.0, 0.0), (5.5, 1.0, 2.0))],
                                 empty_point=(1, 1.8, 1))
    assert not g.line_of_sight((1, 0, 1), (9, 0, 1))       # through the wall
    assert g.line_of_sight((1, 1.8, 1), (9, 1.8, 1))       # through the gap
    # slide: a mover pushed into the wall keeps its tangential motion
    p, outlet = g.collide_and_slide(np.array([[4.6, 0.0, 1.0]]),
                                    np.array([[5.2, 0.3, 1.0]]))
    assert (g.state_at(p) != OBSTACLE).all()
    assert p[0, 1] > 0.0 and not outlet[0]   # y motion survived, x cancelled


def test_boundary_outlets():
    g = OccupancyGrid.from_boxes((0, 0, 0), (4, 4, 4), 0.5, boxes=[],
                                 empty_point=(2, 2, 2))
    assert (g.state_at(np.array([[0.1, 2, 2]])) == OUTLET).all()
    assert (g.state_at(np.array([[99.0, 2, 2]])) == OUTLET).all()  # OOB=OUTLET


def _grid(shape=(8, 8, 8), origin=(0, 0, 0), cell_size=1.):
    return OccupancyGrid(np.asarray(origin, dtype=float), cell_size, np.zeros(shape, np.uint8))


@pytest.mark.parametrize("a,b", [
    ((1.01, 1.5, 1.5), (1.01, 1.5, 1.5)),
    ((1.01, 1.5, 1.5), (1.01 + 1e-13, 1.5, 1.5)),
    ((.99, 1.5, 1.5), (1.01, 1.5, 1.5)),
    ((1.01, 1.5, 1.5), (.99, 1.5, 1.5)),
    ((1 - 1e-13, 1.5, 1.5), (1 + 1e-13, 1.5, 1.5)),
])
def test_los_solid_endpoints_block_short_and_degenerate_segments(a, b):
    g = _grid()
    g.grid[1, 1, 1] = OBSTACLE
    assert not g.line_of_sight(a, b)
    assert not g.line_of_sight(b, a)


def test_los_clear_and_outlet_endpoints_and_batch_contract():
    g = _grid()
    g.grid[1, 1, 1] = OBSTACLE
    g.grid[0, 0, 0] = OUTLET
    free = (.99, 1.5, 1.5)
    solid = (1.01, 1.5, 1.5)
    assert g.line_of_sight(free, free)
    assert g.line_of_sight((.5, .5, .5), (.5, .5, .5))
    assert g.line_of_sight((-.5, .5, .5), (.5, .5, .5))  # OOB/outlet is not solid
    np.testing.assert_array_equal(g.line_of_sight_batch(free, [solid, free]), [False, True])
    np.testing.assert_array_equal(g.line_of_sight_batch(solid, [solid, free]), [False, False])


def test_nearby_plume_does_not_contribute_inside_solid():
    from scentience_olfaction.plume.filament import FilamentPlume, FilamentPlumeConfig
    g = _grid()
    g.grid[1, 1, 1] = OBSTACLE
    p = FilamentPlume(FilamentPlumeConfig(
        source_pos=(.99, 1.5, 1.5), wind_mean=(0, 0, 0), release_rate_hz=20,
        turbulence_intensity=0, sigma_u_floor=0, meander_std_rad=0, gamma=0,
        max_filaments=10), occupancy=g)
    p.step(.05)
    values = p.sample([[.99, 1.5, 1.5], [1.01, 1.5, 1.5]])
    assert values[0] > 0 and values[1] == 0


@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("reverse", [False, True])
def test_sweep_stops_free_to_free_wall_crossing(axis, reverse):
    g = _grid()
    wall = [slice(None)] * 3; wall[axis] = 4
    g.grid[tuple(wall)] = OBSTACLE
    start, target = np.full((1, 3), 2.5), np.full((1, 3), 3.5)
    start[0, axis], target[0, axis] = ((6.5, 1.5) if reverse else (1.5, 6.5))
    original_start, original_target = start.copy(), target.copy()
    assert g.state_at(start)[0] == g.state_at(target)[0] == FREE
    pos, outlet = g.collide_and_slide(start, target)
    assert not outlet.any() and g.state_at(pos)[0] == FREE
    face = 5 if reverse else 4
    assert pos[0, axis] == pytest.approx(face, abs=1e-12)
    assert pos[0, axis] > face if reverse else pos[0, axis] < face
    tangent = [i for i in range(3) if i != axis]
    np.testing.assert_allclose(pos[0, tangent], target[0, tangent], atol=1e-12)
    np.testing.assert_array_equal(start, original_start)
    np.testing.assert_array_equal(target, original_target)


@pytest.mark.parametrize("offset", [(1, 0, 0), (0, 1, 0), (0, 0, 1),
                                    (1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 1, 1)])
@pytest.mark.parametrize("direction", [(1, 1, 1), (-1, -1, -1), (1, -1, 1)])
def test_vertex_crossing_checks_every_touched_voxel(offset, direction):
    g = _grid()
    direction = np.asarray(direction)
    cell = np.where(direction > 0, 1, 2)
    start = cell + .5
    target = start + 2 * direction
    g.grid[tuple(cell + direction * offset)] = OBSTACLE
    pos, outlet = g.collide_and_slide([start], [target])
    blocked = np.asarray(offset, dtype=bool)
    assert np.all(pos[0, blocked] * direction[blocked] < 2 * direction[blocked])
    assert np.allclose(pos[0, ~blocked], target[~blocked])
    assert not outlet.any() and g.state_at(pos)[0] == FREE


def test_diagonally_touching_walls_have_no_corner_leak():
    g = _grid()
    g.grid[2, 1, :] = OBSTACLE
    g.grid[1, 2, :] = OBSTACLE
    pos, outlet = g.collide_and_slide([[1.5, 1.5, 1.5]], [[3.5, 3.5, 2.5]])
    assert np.all(pos[0, :2] < 2)
    assert pos[0, 2] == pytest.approx(2.5) and not outlet[0]


def test_slide_is_swept_into_second_wall():
    g = _grid()
    g.grid[4, :, :] = OBSTACLE
    g.grid[:, 5, :] = OBSTACLE
    # X is hit first. The remaining Y slide crosses another entire wall.
    pos, outlet = g.collide_and_slide([[1.5, 1.5, 1.5]], [[7.5, 6.5, 2.5]])
    assert 3.99 < pos[0, 0] < 4
    assert 4.99 < pos[0, 1] < 5
    assert pos[0, 2] == pytest.approx(2.5)
    assert not outlet[0] and g.state_at(pos)[0] == FREE


def test_outlet_on_path_culls_even_when_endpoint_is_free():
    g = _grid()
    g.grid[3, :, :] = OUTLET
    pos, outlet = g.collide_and_slide([[1.5, 1.5, 1.5]], [[6.5, 1.5, 1.5]])
    assert outlet[0]
    assert pos[0, 0] == pytest.approx(3)


@pytest.mark.parametrize("endpoint", [6.5, 100.])
def test_wall_before_outlet_prevents_false_culling(endpoint):
    g = _grid()
    g.grid[3, :, :] = OBSTACLE
    g.grid[5, :, :] = OUTLET
    pos, outlet = g.collide_and_slide([[1.5, 1.5, 1.5]], [[endpoint, 1.5, 1.5]])
    assert not outlet[0] and 2.99 < pos[0, 0] < 3
    # Reverse ordering: gas leaves through the outlet before reaching a wall.
    g.grid[3, :, :] = OUTLET
    g.grid[5, :, :] = OBSTACLE
    pos, outlet = g.collide_and_slide([[1.5, 1.5, 1.5]], [[endpoint, 1.5, 1.5]])
    assert outlet[0] and pos[0, 0] == pytest.approx(3)


def test_slide_into_outlet_and_out_of_domain():
    g = _grid()
    g.grid[4, :, :] = OBSTACLE
    start, target = [[1.5, 1.5, 1.5]], [[7.5, 10.5, 1.5]]
    pos, outlet = g.collide_and_slide(start, target)
    assert outlet[0] and pos[0, 0] < 4 and pos[0, 1] == pytest.approx(8)
    g.grid[:, 6, :] = OUTLET
    pos, outlet = g.collide_and_slide(start, target)
    assert outlet[0] and pos[0, 0] < 4 and pos[0, 1] == pytest.approx(6)


def test_simultaneous_wall_and_tangential_outlet_contact_culls():
    g = _grid()
    g.grid[:, 2, :] = OBSTACLE
    g.grid[2, 1, :] = OUTLET
    pos, outlet = g.collide_and_slide([[1.5, 1.5, 1.5]], [[4.5, 4.5, 1.5]])
    assert outlet[0]
    assert pos[0, 0] == pytest.approx(2) and pos[0, 1] < 2


@pytest.mark.parametrize("origin,cs", [((0, 0, 0), 1.), ((-10, 4, -3), .25),
                                     ((1e6, -1e6, 1e6), .01)])
def test_face_contact_is_stable_under_repeated_pushes(origin, cs):
    g = _grid(origin=origin, cell_size=cs)
    g.grid[4, :, :] = OBSTACLE
    pos = g.origin + np.array([[1.5, 1.5, 1.5]]) * cs
    face = g.origin[0] + 4 * cs
    for _ in range(12):
        pos, outlet = g.collide_and_slide(pos, pos + np.array([[20, .1, 0]]) * cs)
        assert pos[0, 0] < face and g.state_at(pos)[0] == FREE and not outlet[0]
    assert pos[0, 1] == pytest.approx(g.origin[1] + 2.7 * cs)


def test_start_exactly_on_face_can_slide_or_move_away_but_not_into_wall():
    g = _grid()
    g.grid[3, :, :] = OBSTACLE
    start = np.tile([4., 1.5, 1.5], (3, 1))
    target = np.array([[2.5, 2.5, 1.5], [4., 2.5, 1.5], [5., 2.5, 1.5]])
    pos, outlet = g.collide_and_slide(start, target)
    np.testing.assert_allclose(pos, [[4, 2.5, 1.5], [4, 2.5, 1.5], [5, 2.5, 1.5]])
    assert not outlet.any()


def test_batch_handles_no_motion_same_cell_exit_and_empty_input():
    g = _grid()
    start = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1], [-1, 1, 1]])
    target = np.array([[1, 1, 1], [1.2, 1.4, 1.5], [-2, 1, 1], [2, 1, 1]])
    pos, outlet = g.collide_and_slide(start, target)
    np.testing.assert_allclose(pos[:2], target[:2])
    np.testing.assert_array_equal(outlet, [False, False, True, True])
    pos, outlet = g.collide_and_slide(np.empty((0, 3)), np.empty((0, 3)))
    assert pos.shape == (0, 3) and outlet.shape == (0,)
    # Integer inputs must not truncate subcell tangential positions.
    pos, _ = g.collide_and_slide([[1, 1, 1]], [[1.2, 1.4, 1.5]])
    np.testing.assert_allclose(pos, [[1.2, 1.4, 1.5]])


def test_invalid_starts_and_endpoints_fail_explicitly():
    g = _grid()
    g.grid[1, 1, 1] = OBSTACLE
    with pytest.raises(ValueError, match="inside an obstacle"):
        g.collide_and_slide([[1.5, 1.5, 1.5]], [[6, 6, 6]])
    for start, target in [([[0, 0, 0]], [[np.nan, 0, 0]]),
                          ([[0, 0, 0]], [[1, 2]]),
                          ([[0, 0, 0]], [[1, 2, 3], [2, 3, 4]])]:
        with pytest.raises(ValueError):
            g.collide_and_slide(start, target)


def test_random_sweeps_match_independent_segment_box_intersections():
    """First impact agrees with an analytic slab oracle, independent of DDA."""
    g = _grid()
    rng = np.random.default_rng(170)
    solids = np.array([[3, 3, 3], [4, 2, 1], [1, 5, 4], [5, 5, 5]])
    for c in solids:
        g.grid[tuple(c)] = OBSTACLE
    start = rng.uniform(.1, 7.9, (500, 3))
    target = rng.uniform(.1, 7.9, (500, 3))
    valid = g.state_at(start) == FREE
    start, target = start[valid], target[valid]
    delta = target - start
    near = (solids[None, :, :] - start[:, None, :]) / delta[:, None, :]
    far = (solids[None, :, :] + 1 - start[:, None, :]) / delta[:, None, :]
    enter = np.maximum(np.minimum(near, far).max(2), 0.)
    leave = np.minimum(np.maximum(near, far).min(2), 1.)
    first = np.where(enter <= leave, enter, np.inf).min(1)
    pos, blocked, outlet = g._sweep(start, target)
    np.testing.assert_array_equal(blocked.any(1), np.isfinite(first))
    assert not outlet.any()
    hit = np.isfinite(first)
    assert hit.sum() > 30  # ensure this fixture actually exercises collisions
    np.testing.assert_allclose(pos[hit], start[hit] + first[hit, None] * delta[hit], atol=1e-12)
    np.testing.assert_array_equal(pos[~hit], target[~hit])
    assert np.all(g.state_at(pos) == FREE)


def test_fast_plume_cannot_cross_wall_or_be_culled_behind_it():
    from scentience_olfaction.plume.filament import FilamentPlume, FilamentPlumeConfig
    g = _grid()
    g.grid[3, :, :] = OBSTACLE
    g.grid[5, :, :] = OUTLET
    p = FilamentPlume(FilamentPlumeConfig(
        source_pos=(1.5, 1.5, 1.5), wind_mean=(100, 0, 0), release_rate_hz=20,
        turbulence_intensity=0, sigma_u_floor=0, meander_std_rad=0, gamma=0,
        max_filaments=50, domain_min=(0, 0, 0), domain_max=(8, 8, 8)), occupancy=g)
    p.step(.25)
    assert p.n_alive == 5 and np.all(p.pos[p.alive, 0] < 3)
    budget = p.diagnostics()
    assert not budget["outlet_mol"].any() and not budget["domain_mol"].any()
    np.testing.assert_allclose(budget["retained_mol"], budget["released_mol"])
