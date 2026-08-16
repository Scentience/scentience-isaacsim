"""
scentience-olfaction: chemical plume transport and olfactory sensor models
for robotics simulation (NVIDIA Isaac Sim / Isaac Lab, Gymnasium, standalone).

Quick start:

    from scentience_olfaction import OlfactionWorld
    world = OlfactionWorld.simple()
    world.step(0.05)
    print(world.read((5, 0, 1)))
"""
from .api import OlfactionWorld
from .plume.filament import FilamentPlume, FilamentPlumeConfig
from .emitters.emitters import PointEmitter, LineEmitter, BoxEmitter
from .geometry.occupancy import OccupancyGrid
from .airflow.fields import UniformAirflow, GridAirflow, potential_flow
from .chemistry.registry import Species, SpeciesRegistry, DEFAULT_REGISTRY
from .provenance import Evidence, ProvenanceRegistry, coeff

__version__ = "0.1.0"
__all__ = ["OlfactionWorld", "FilamentPlume", "FilamentPlumeConfig",
           "PointEmitter", "LineEmitter", "BoxEmitter", "OccupancyGrid",
           "UniformAirflow", "GridAirflow", "potential_flow",
           "Species", "SpeciesRegistry", "DEFAULT_REGISTRY",
           "Evidence", "ProvenanceRegistry", "coeff"]
