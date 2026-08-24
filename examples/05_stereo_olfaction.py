"""Stereo olfaction with the two MiCS-6814 dies.
Run: python examples/05_stereo_olfaction.py

The Scentience dev kit carries TWO MiCS-6814 sensors so a robot can
lateralise a plume: with a heading, `chem_left_*` samples the air to the robot's
LEFT and `chem_right_*` to its RIGHT, `stereo_baseline_m` apart. The sign of the
left-right difference says which way to turn -- the first-order stereo cue.

Here a robot stands downwind, off the plume centreline to its right-hand
side, and simply compares its two dies. RED-die Rs/R0 DROPS with gas, so the
smaller number is the side that smells more.
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from scentience_olfaction import OlfactionWorld
from scentience_olfaction.plume.filament import FilamentPlume, FilamentPlumeConfig

# Meander off for a clean demo -- the time-averaged plume then sits exactly on
# y=0 and geometry alone decides which die wins. (With meander on, the cue
# still works but flips whenever the plume snakes across the robot.)
cfg = FilamentPlumeConfig(source_pos=(0.0, 0.0, 1.0), wind_mean=(1.0, 0.0, 0.0),
                          ppm_center_initial=300.0, release_rate_hz=40.0,
                          meander_std_rad=0.0, turbulence_intensity=0.10)
world = OlfactionWorld(FilamentPlume(cfg, seed=0),
                       sensor_profile="fast_modulated", seed=0,
                       stereo_baseline_m=0.5)   # exaggerated for a clear demo

for _ in range(100):                            # plume spin-up
    world.step(0.05)

# Robot at y=-0.25 facing +x (downwind): plume centreline is on its LEFT.
left = right = 0.0
n = 400
for _ in range(n):
    world.step(0.05)
    r = world.read((4.0, -0.25, 1.0), dt=0.05, heading=0.0)
    left += r["chem_left_red"] / n
    right += r["chem_right_red"] / n

print(f"left  die (chem_left_red)  mean Rs/R0 = {left:.3f}")
print(f"right die (chem_right_red)  mean Rs/R0 = {right:.3f}")
side = "LEFT" if left < right else "RIGHT"
print(f"lower ratio = more gas -> plume is to the robot's {side}; turn that way.")
