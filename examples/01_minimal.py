"""Smell in five lines. Run: python examples/01_minimal.py"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from scentience_olfaction import OlfactionWorld

world = OlfactionWorld.simple()                 # ethanol source, 1 m/s wind
for _ in range(200):
    world.step(0.05)                            # 10 seconds of plume
reading = world.read((5.0, 0.0, 1.0), dt=0.05)  # virtual Scentience device
print({k: round(v, 4) for k, v in reading.items()})
print("ground truth:", world.truth((5.0, 0.0, 1.0)))
