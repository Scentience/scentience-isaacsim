# scentience-olfaction

**Olfactory sensing for robotics simulation.** Chemical plume transport and
virtual Scentience olfactory sensors for NVIDIA Isaac Sim / Isaac Lab,
Gymnasium, and standalone Python. To our knowledge, the first olfactory and
chemical sensing package for Isaac Sim.

By [Scentience](https://scentience.ai) -- sensors and AI for machine
olfaction.

## Five lines to smell

```python
from scentience_olfaction import OlfactionWorld

world = OlfactionWorld.simple()               # ethanol source, 1 m/s wind
world.step(0.05)
reading = world.read((5.0, 0.0, 1.0))         # virtual Scentience device
truth   = world.truth((5.0, 0.0, 1.0))        # ground truth, for debugging
```

`pip install -e .` -- core needs only NumPy. `pip install -e ".[dev]"` for
everything (Warp, torch, gymnasium, pytest). `pytest -m "not isaac"` runs the
full physics validation on CPU, no Isaac, no GPU.

## What is in the box

| | |
|---|---|
| **Plume transport** | Filament model (Farrell 2002): multi-species, multiple emitters, walls (occupancy + line-of-sight + slide), two-scale turbulence. NumPy reference + Warp GPU twin, parity-tested. |
| **Sensor suite** | Two chemical (metal-oxide) sensors, `chem_left` / `chem_right`, for stereo olfaction -- each samples the plume at its own position across a configurable baseline, with the full signal chain per sensor (power law -> asymmetric lag -> drift/1-f -> ADC divider). Plus a CO2 channel (photoacoustic, ASC), electrochemical cells (linear + Cottrell), and PID. Full Scentience V1 device in the hardware BLE channel schema. |
| **Realism gate** | CI-enforced plume statistics vs published turbulence theory. A plume that gets too easy FAILS THE BUILD. |
| **OIO** | Olfactory Inertial Odometry reference implementation (France et al., arXiv:2506.04539; Chasing Ghosts bout detection) with UAV / quadruped / biped / arm presets. |
| **RL** | Gymnasium `PlumeNavEnv` (hardware-shaped observations, stereo cue included), cast-and-surge + stereo (onset-lag steering, sensor-only source declaration; Chasing Ghosts, arXiv:2602.19577) + random baselines, episode recorder. Isaac Lab `SensorBase` integration. |
| **Provenance** | Every physical constant carries an evidence level (MEASURED/DATASHEET/DIGITIZED/SYNTHESIZED/ASSUMED); `claim_check()` refuses claims the evidence cannot support. |

## Examples

```bash
python examples/01_minimal.py                                # smell in 5 lines
python examples/02_walls_and_wind.py                         # plume vs a wall
python examples/03_olfactory_inertial_odometry.py --platform quadruped   # or uav|biped|arm
python examples/04_gym_baseline.py                           # the benchmark loop
python examples/05_stereo_olfaction.py                       # two sensors, one plume: lateralisation
```

Visual verification (`pip install "scentience-olfaction[viz]"`):
`python scripts/plot_verification.py` renders ground truth vs the slow and
fast device responses, and the stereo left/right cue, as PNGs.
Something not working? See `docs/TROUBLESHOOTING.md`.

## The two numbers to know before using it

**1. Large-scale meander is not optional.** Blank-duration CV 1.7 +/- 0.4
with it (range 1.4-2.4 over 5 seeds), 0.95 +/- 0.02 without (600 s @ 100 Hz,
8 m downwind). CV < 1 means exponential blanks: no search problem, and
policies learn gradient ascent that fails on hardware. Tail exponents bracket
the -3/2 of Celani et al. (PRX 4:041015). Seed and threshold conventions in
`docs/CHEMICAL_MODEL.md`.

**2. Sensor bandwidth gates what a policy can see.** On an identical plume,
a packaged MOX (tau_fall 12 s) retains **19%** of whiff events; a fast sensor
(46 ms, time constants per Dennler et al., Sci. Adv. 2024) retains **97%**.
State your `sensor_profile` in every result.

## Isaac Sim / Isaac Lab status

The Isaac Lab sensor (`scentience_isaaclab/`) targets Isaac Lab 2.3.x /
Isaac Sim 5.1. It is **API-contract validated** against the real
`isaaclab==2.3.2` wheel -- 22 static checks plus our classes executed by
genuine isaaclab code with the kit runtime stubbed
(`scripts/check_isaaclab_contract.py`, `scripts/check_isaaclab_binding.py`;
record in `docs/ISAAC_COMPATIBILITY.md`) -- but has NOT yet been executed in
a live Isaac Sim install. Run `scripts/validate_install.py` inside Isaac and
paste its output into `docs/ISAAC_COMPATIBILITY.md` before relying on it.
Until then, the supported paths are standalone Python and Gymnasium.

## Honesty policy

Sensitivity coefficients ship as DIGITIZED/SYNTHESIZED evidence (datasheets
publish graphs, not tables) and the package says so at runtime. Buoyancy is
off rather than wrong. GADEN (LGPL) is cited, never transcribed -- the plume
is implemented from Farrell's published equations. See
`docs/LICENSES_AND_PROVENANCE.md`.

## Cite

See `CITATION.cff`. Related Scentience research: olfaction standardization
(arXiv:2506.00398), olfactory inertial odometry (arXiv:2506.04539),
accelerated chronoamperometry (arXiv:2506.04540), Chasing Ghosts
(arXiv:2602.19577).

Apache-2.0.
