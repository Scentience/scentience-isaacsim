# The chemical model

What is modelled, how, and -- more usefully -- what is **not**.

Five places in the source point here. Each is answered below:

| Source | Question it defers here |
|---|---|
| `plume/filament.py:8` | design notes and deviations from Farrell |
| `plume/filament.py:83,188` | the buoyancy slip model |
| `chemistry/registry.py:32` | why `specific_gravity` is usually inert |
| `airflow/fields.py:122` | what potential flow does and does not give you |
| `geometry/occupancy.py:176` | the project-and-slide wall response |

---

## 1. Transport: filaments, not a grid

The plume is a Lagrangian **filament** model after Farrell et al. (2002),
*Environmental Fluid Mechanics* 2:143-169. Each filament is a puff of released
material with a centre position and a Gaussian radius `sigma`. Concentration at
a probe is the superposition of every filament's 3-D Gaussian.

This is chosen over an Eulerian grid because the quantity that matters for
search -- **intermittency**, the alternation of whiffs and blanks -- is
destroyed by grid diffusion. A grid solver gives you a smooth time-averaged
cone that a gradient-ascent policy solves trivially and that does not exist in
any real turbulent flow.

`plume/filament.py` (NumPy) is the **specification**.
`transport/filament_warp.py` (Warp/GPU) is the fast path.
`tests/test_warp_parity.py` binds them: they must agree on physics
(growth law, mass conservation, OU stationary variance, aggregate field
statistics), not on RNG draws. Asserting bitwise equality would test the
random number generator, not the model.

### Two deliberate deviations from common practice

Both are corrections, and both are load-bearing.

**1. Exact Ornstein-Uhlenbeck integration.**
Small-scale turbulent velocity is an OU process with Lagrangian timescale
`T_L`, advanced with the exact discrete update

```
a  = exp(-dt / T_L)
u' = a*u' + sigma_u * sqrt(1 - a^2) * xi        xi ~ N(0, 1)
```

The common shortcut -- a memoryless kick scaled by `dt` -- makes the resulting
turbulent diffusivity a function of your timestep. Your plume then changes
character when you change `dt`, which is silent and ruinous. `test_warp_parity.py`
asserts stationary variance holds across `dt` in {0.005, 0.01, 0.05}.

**2. A shared large-scale bearing meander.**
One OU process on the mean wind bearing, advanced **once per step** and shared
by every filament. This is what produces heavy-tailed blank durations.

This is not a detail. Without it, blank durations are near-exponential, there
is no search problem, and a policy learns gradient ascent that fails on
hardware. The realism gate (`tests/test_plume_gate.py`) fails the build if
meander is ablated -- that is its entire purpose.

**On the published figure**: `filament.py`'s docstring and the README quote
"CV 2.31 -> 0.96" for the meander ablation. That comparison is correct in
direction and mechanism, but the full-plume value is seed-dependent. Measured
across 5 seeds at 600 s / 100 Hz / 8 m downwind, using the shared threshold
convention in `scripts/validate_physics.py`:

| series | mean | sd | range |
|---|---|---|---|
| full plume | 1.72 | 0.41 | 1.40 - 2.40 |
| meander ablated | 0.950 | 0.020 | 0.923 - 0.968 |

2.31 is a high draw, not a typical one; expect ~1.7. The ablated figure is
robust. When you report a CV, state your seed and threshold.

### Growth and decay

Filaments grow by Farrell "Model 2", `dsigma/dt = gamma / (2*sigma)`, floored
by the species' molecular diffusivity. `gamma` ships at `2e-3 m^2/s`, tuned up
from Farrell's outdoor `1e-3` for indoor scales. Its evidence level is
**ASSUMED** and `claim_check()` knows it.

Optional first-order decay per species (`decay_rate_per_s`) is available and
defaults to zero.

---

## 2. Airflow

Three fields, in increasing order of fidelity and cost.

**`UniformAirflow`** -- constant mean wind plus the OU bearing meander above.
The default, and sufficient for most benchmark work.

**`GridAirflow`** -- a precomputed velocity field on a regular grid, trilinear
interpolation. This is the import path for a real CFD/RANS solution. If you
care about wakes or recirculation, compute them properly in a CFD tool and
import the result here.

**`potential_flow()`** -- inviscid, incompressible flow around occupancy
obstacles.

> What you get: smooth, curl-free, divergence-free flow that goes **around**
> obstacles.
> What you do **not** get: wakes, separation, recirculation -- the interesting
> parts of real indoor flow.

Potential flow has no viscosity, so it cannot separate from a surface. Behind a
bluff body it reattaches cleanly and predicts *no* recirculation zone. Real
indoor plumes pool and swirl in exactly those regions, and a search policy
trained without them will be over-confident downwind of furniture. Use
`GridAirflow` with a RANS import when that matters.

---

## 3. Obstacles: occupancy, line-of-sight, slide

Geometry is voxelized to an `OccupancyGrid`. Three things use it:

1. **Occupancy** -- a filament cannot occupy an obstacle voxel.
2. **Line-of-sight** -- a filament does not contribute concentration to a probe
   it cannot see. Without this, material bleeds through walls, because a
   Gaussian has infinite support.
3. **Slide** -- the wall response.

### The project-and-slide response

The exact response to a wall is recursive: project the motion onto the surface,
then re-test the projected motion, repeatedly, until it neither penetrates nor
changes. Corners can require several passes.

What is implemented is a **single-pass axis-decomposed approximation**: each
component of the step is tested independently from the old position, and
components that would enter an obstacle are cancelled. This slides along walls
instead of sticking to them.

It is **exact for axis-aligned geometry** and **approximate near corners**,
where a true recursive solve would redistribute the cancelled component into
the remaining free direction and this does not. The error is bounded by one
step length, and it is conservative -- filaments lose a little motion at
corners rather than tunnelling.

Destinations that leave the domain, or reach an `OUTLET` voxel, retire the
filament.

---

## 4. What is deliberately not modelled

Each of these is **off, not wrong**. Shipping an undefended model silently
enabled is worse than shipping none.

### Buoyancy -- off by default

`buoyancy_model` defaults to `"none"`. The alternative, `"slip"`, adds a
vertical slip velocity scaled by dilution:

```
v_z += slip_speed_scale * (1 - specific_gravity) * (sigma0 / sigma)^3
```

The `(sigma0/sigma)^3` term makes a grown, dilute filament stop separating from
the carrier air, which is qualitatively right. But the model is
**phenomenological, not derived physics** -- `slip_speed_scale` is not a
measured quantity and the functional form is a modelling choice, not a
buoyancy equation.

It exists so that dense plumes (CO2) and light ones (H2, CH4) are not silently
pretending to be neutrally buoyant. It is off by default precisely because it
is undefended. Turn it on if you need the qualitative behaviour; do not report
quantitative results from it.

This is why `Species.specific_gravity` is inert in a default run: nothing reads
it unless a buoyancy model is explicitly enabled.

### Also absent

- **Thermal effects.** No stratification, no buoyant rise from a warm source.
- **Wall adsorption/desorption.** Real surfaces store and re-release VOCs; a
  room does not have a memoryless boundary. Not modelled.
- **Chemical reaction between species.** Species are transported independently
  and superposed. No ozonolysis, no secondary aerosol.
- **Humidity effects on transport.** Humidity affects the *sensor* models
  (a real, documented effect on MOX) but not the plume.
- **Wakes, separation, recirculation** in `potential_flow()` -- see above.
- **Sensor back-action.** The probe does not perturb the flow.

---

## 5. Units and conventions

SI throughout: metres, seconds, kilograms. Concentrations are reported in
**ppm by volume**. Species properties live in `chemistry/registry.py`, and the
ideal-gas conversion uses `R = 82.057338e-6 m^3 atm / (mol K)` with
configurable `temperature_k` and `pressure_atm`.

Ground truth (`world.truth()`) is available for debugging and **must never
enter a policy observation** -- that is invariant 2 in `ARCHITECTURE.md`, and
`tests/test_env_properties.py` enforces it.

---

## References

Farrell, J.A. et al. (2002). Filament-based atmospheric dispersion model to
achieve short time-scale structure of odor plumes. *Environmental Fluid
Mechanics* 2:143-169.

Celani, A., Villermaux, E., Vergassola, M. (2014). Odor landscapes in turbulent
environments. *Physical Review X* 4:041015.

Monroy, J. et al. (2017). GADEN: A 3D gas dispersion simulator for mobile robot
olfaction. *Sensors* 17(7):1479. *(behavioural cross-check only -- LGPL-3.0,
never transcribed; see LICENSES_AND_PROVENANCE.md)*

Dennler, N. et al. (2024). Limits of rapid gas sensing. *Science Advances*
10:eadp1764.
