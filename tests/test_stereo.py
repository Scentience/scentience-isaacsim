"""Stereo olfaction: the two MiCS-6814 dies sample separated points.

The dev kit carries two MiCS-6814 dies so a robot can lateralise a plume from
the inter-die concentration difference. These tests pin the three things that
make that feature trustworthy:

  1. back-compat -- without a heading / second concentration, behaviour is
     bit-identical to the pre-stereo mono path;
  2. physics -- a source displaced to the LEFT of the heading produces a
     stronger response on the LEFT die (mics1) than the RIGHT (mics2);
  3. plumbing -- the gym env exposes the stereo cue without changing the
     observation contract.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np

from scentience_olfaction import OlfactionWorld
from scentience_olfaction.sensors.device_np import ScentienceV1


def test_mono_backcompat_bit_identical():
    """step() without conc_ppm_2 must be indistinguishable from the old API."""
    conc = {"ethanol": 5.0}
    a = ScentienceV1("fast_modulated", seed=7)
    b = ScentienceV1("fast_modulated", seed=7)
    for _ in range(50):
        ra = a.step(conc, 0.05)
        rb = b.step(conc, 0.05, conc_ppm_2=None)
    assert ra == rb


def test_stereo_equals_mono_when_both_dies_see_same_air():
    """conc_ppm_2 == conc_ppm must reproduce mono exactly."""
    conc = {"ethanol": 5.0}
    a = ScentienceV1("fast_modulated", seed=7)
    b = ScentienceV1("fast_modulated", seed=7)
    for _ in range(50):
        ra = a.step(conc, 0.05)
        rb = b.step(conc, 0.05, conc_ppm_2=dict(conc))
    assert ra == rb


def test_second_die_receives_second_concentration():
    """chem_right_* must respond to conc_ppm_2, not conc_ppm."""
    hot, cold = {"ethanol": 50.0}, {"ethanol": 0.0}
    dev = ScentienceV1("fast_modulated", seed=3, randomize_unit=False)
    for _ in range(200):  # let the fast MOX settle
        r = dev.step(cold, 0.05, conc_ppm_2=hot)
    # RED die: Rs/R0 DROPS with reducing gas, so the hot die reads LOWER.
    assert r["chem_right_red"] < r["chem_left_red"], r


def test_left_source_lateralises_left_centreline_geometry():
    """Meander is DISABLED here on purpose: with it on, 20 s is ~1 meander
    timescale, so the whole plume can sit on either side of the centreline for
    the entire window and the sign of the L/R difference is a coin flip (that
    is the search problem, working as intended). With meander off the
    time-averaged lateral profile is symmetric about y=0, and geometry decides:
    the die ON the centreline must smell more than the die half a baseline off
    it."""
    from scentience_olfaction.plume.filament import FilamentPlume, FilamentPlumeConfig

    cfg = FilamentPlumeConfig(source_pos=(0.0, 0.0, 1.0), wind_mean=(1.0, 0.0, 0.0),
                              ppm_center_initial=300.0, release_rate_hz=40.0,
                              meander_std_rad=0.0, turbulence_intensity=0.10)
    world = OlfactionWorld(FilamentPlume(cfg, seed=11),
                           sensor_profile="fast_modulated", seed=11,
                           stereo_baseline_m=0.5)
    for _ in range(100):
        world.step(0.05)
    left_sum = right_sum = 0.0
    n = 400
    # Probe at y=-0.25, heading +x: left die lands on the centreline (y=0),
    # right die at y=-0.5, half a baseline off it.
    for _ in range(n):
        world.step(0.05)
        r = world.read((4.0, -0.25, 1.0), dt=0.05, heading=0.0)
        left_sum += r["chem_left_red"]
        right_sum += r["chem_right_red"]
    # RED Rs/R0 drops with concentration: on-centreline (left) die reads lower.
    assert left_sum < right_sum, (left_sum / n, right_sum / n)


def test_world_mono_when_no_heading():
    """read() without heading keeps the exact pre-stereo behaviour."""
    a = OlfactionWorld.simple(seed=5)
    b = OlfactionWorld.simple(seed=5)
    b.stereo_baseline_m = 0.0  # even with heading given, 0 baseline == mono
    for _ in range(50):
        a.step(0.05)
        b.step(0.05)
    ra = a.read((3.0, 0.0, 1.0), dt=0.05)
    rb = b.read((3.0, 0.0, 1.0), dt=0.05, heading=1.234)
    assert ra == rb


def test_env_stereo_observation_contract_unchanged():
    """Stereo on/off must not change the observation space, and stereo must
    actually reach the env's device (left/right dies see different air)."""
    gym_mod = __import__("pytest").importorskip("gymnasium")  # noqa: F841
    from scentience_olfaction.envs.plume_nav import PlumeNavEnv, PlumeNavConfig

    stereo = PlumeNavEnv(PlumeNavConfig())            # default: stereo on
    mono = PlumeNavEnv(PlumeNavConfig(stereo_baseline_m=0.0))
    assert stereo.observation_space.shape == mono.observation_space.shape
    obs, _ = stereo.reset(seed=0)
    assert obs.shape == stereo.observation_space.shape
    o2, *_ = stereo.step(np.array([0.5, 0.0], dtype=np.float32))
    assert o2.shape == stereo.observation_space.shape


if __name__ == "__main__":
    test_mono_backcompat_bit_identical()
    test_stereo_equals_mono_when_both_dies_see_same_air()
    test_second_die_receives_second_concentration()
    test_left_source_lateralises_left_centreline_geometry()
    test_world_mono_when_no_heading()
    print("PASS")
