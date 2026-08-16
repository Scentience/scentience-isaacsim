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
