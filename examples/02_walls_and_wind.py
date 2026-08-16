"""
A wall with a gap: the plume must go around, and so must the robot.
Demonstrates occupancy grids and obstacle-aware potential flow.
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
import numpy as np
from scentience_olfaction import (FilamentPlume, FilamentPlumeConfig,
                                  OccupancyGrid, potential_flow)

occ = OccupancyGrid.from_boxes(
    domain_min=(-2, -5, 0), domain_max=(20, 5, 4), cell_size=0.25,
    boxes=[((6.0, -5.0, 0.0), (6.5, 1.0, 4.0))],   # wall with a gap at y>1
    empty_point=(0, 0, 1))
airflow = potential_flow(occ, mean_wind=(1.0, 0.0, 0.0))

cfg = FilamentPlumeConfig(source_pos=(0, -2, 1), release_rate_hz=60,
                          domain_min=(-2, -5, 0), domain_max=(20, 5, 4),
                          max_filaments=6000)
plume = FilamentPlume(cfg, seed=0, occupancy=occ, airflow=airflow)

front, behind, through_gap = np.zeros(3), 0.0, 0.0
probes = np.array([[5.0, -2.0, 1.0],    # upwind side of the wall
                   [8.0, -2.0, 1.0],    # directly behind the wall
                   [8.0,  2.5, 1.0]])   # downstream of the gap
acc = np.zeros(3)
for i in range(4000):
    plume.step(0.01)
    if i > 1500:
        acc += plume.sample(probes)
acc /= 2500
print(f"upwind of wall : {acc[0]:.3f} ppm")
print(f"behind wall    : {acc[1]:.3f} ppm   (blocked)")
print(f"past the gap   : {acc[2]:.3f} ppm   (leaked around)")
