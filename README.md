# scentience-olfaction

Chemical plume transport and olfactory sensor models for **NVIDIA Isaac Sim /
Isaac Lab**. To our knowledge this is the first olfactory and chemical sensing
package for Isaac Sim — Isaac's sensor set covers camera, lidar, radar, IMU,
contact, effort, and raycast, and nothing chemical.

**Status: v0.1.0.dev0.** The physics core is validated. The Isaac Lab
integration is written against the API but **has not yet run inside Isaac Sim**
— see `docs/ISAAC_COMPATIBILITY.md`. Do not cite it as working until
`scripts/validate_install.py` passes.

## What it does

- **Filament plume transport** (Farrell et al. 2002), NumPy reference + Warp
  GPU path, batched over parallel environments.
- **Turbulence at two scales** — per-filament Ornstein-Uhlenbeck plus a shared
  large-scale bearing meander. Both are required; see below.
- **Olfactory sensor models** — MOX power law with asymmetric response
  dynamics, drift, 1/f noise, and ADC quantisation through the voltage divider.
- **A plume realism gate** that runs in CI and fails builds whose plume
  statistics stop matching published turbulence theory.
- **Evidence provenance** on every physical coefficient.

## The two results worth knowing before you use it

**1. Large-scale meander is not optional.** 600 s at 100 Hz, probe 8 m
downwind:

| configuration | blank-duration CV | gate |
|---|---|---|
| small + large scale turbulence | **2.31** | PASS |
| meander ablated | **0.96** | FAIL |
| Gaussian plume + slow meander | — (always on) | FAIL |

A blank-duration CV below 1 means exponentially distributed blanks: no long
absences, no search problem, and a policy that learns gradient ascent and fails
on hardware. Blank/whiff tail exponents here bracket the -3/2 first-return
exponent predicted by Celani, Villermaux & Vergassola (PRX 4:041015).

**2. Sensor bandwidth, not plume physics, gates how much a policy can see.**
Identical plume, two sensor profiles:

| profile | tau_fall | whiff events retained |
|---|---|---|
| `packaged_slow` | 12 s | **19 %** |
| `fast_modulated` | 46 ms | **97 %** |

Those train different POMDPs. State which profile you used in every result.

## Install

```bash
pip install -e ".[dev]"      # core + Warp + torch + pytest
pytest -m "not isaac"        # physics validation, CPU only, no Isaac needed
```

The core imports with **no Isaac and no GPU**. That is a hard requirement, not
a convenience: it is what lets the physics run in CI.

## Honesty about coefficients

Every physical constant carries an evidence level — `MEASURED`, `DATASHEET`,
`DIGITIZED`, `SYNTHESIZED`, `ASSUMED` — with its source and conditions. Ask any
model for `.provenance.report()`, and use `.claim_check()` before writing a
quantitative statement. Sensor sensitivity coefficients are currently
`DIGITIZED` or `SYNTHESIZED`: they are inversions of open-source driver fits
digitised from datasheet graphs, because the MiCS-6814 datasheet publishes no
tabulated coefficients. They are hypotheses about the sensor, not measurements
of it, and the package says so at runtime.

## License

Apache-2.0. Every equation is implemented from the published literature
(Farrell et al. 2002; Celani et al. 2014); no GADEN source (LGPL-3.0) is
transcribed here.
