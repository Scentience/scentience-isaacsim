# Configuration

Use dataclasses in Python or strict JSON for repeatable experiments. Public
parameters use SI units except explicit `ppm`, `nA`, Celsius and percent-RH
fields. Coordinates are metres in a right-handed, Z-up world; a sensor's local
X points forward and Y points left. The standalone world API explicitly names its scalar-first argument
`orientation_wxyz`; the Isaac Lab 3 adapter uses scalar-last **xyzw**, matching
Lab 3. Convert at this boundary when sharing mount configurations. Geometry
importers must convert their scene units to metres.

## World and transport

`load_world(path)` accepts `plume`, `device_profile`, `sensor_profile`,
`device_config`, `stereo_baseline_m`, and `seed`. Plume fields match
`FilamentPlumeConfig`. Emitters carry a `type` of `point`, `line` or `box` and
the corresponding emitter dataclass fields. Unknown keys are errors.

```python
from scentience_olfaction import FilamentPlume, FilamentPlumeConfig, OlfactionWorld

plume = FilamentPlume(FilamentPlumeConfig(
    molar_rate_mol_s=1e-6,       # mol/s; independent of numerical filament size
    release_rate_hz=40,
    wind_mean=(1.0, 0.0, 0.0),
    max_step_s=0.02,
), seed=7)
world = OlfactionWorld(plume, sensor_profile="packaged_slow")
```

`ppm_center_initial`, `sigma0` and release rate jointly set source flux in the
legacy formulation. Prefer `molar_rate_mol_s` when comparing discretizations.
See [transport](TRANSPORT.md) for growth, cutoff, wall and timestep semantics.
Python constructors accept airflow/occupancy collaborators; these objects are
not serialized by `load_world`. Use `GridAirflow` for externally computed wind.

Concentration is ppm by volume. With no `background_ppm`, CO₂ plume values
represent excess above a sensor's ambient baseline. An explicit
`background_ppm={"carbon_dioxide": 420}` makes CO₂ truth absolute. World and
Isaac defaults then configure absolute CO₂ input; a conflicting excess-mode
configuration is rejected. Other species backgrounds are additive too.

## Device calibration

```python
from dataclasses import replace
from scentience_olfaction.sensors.device_np import DeviceConfig
from scentience_olfaction.sensors.mox import MOX_RED, MOX_NH3, MOX_OX

# Demonstration override, not a measured calibration.
red = replace(MOX_RED, tau_rise_s=2.0, dead_volume_delay_s=0.1)
calibration = DeviceConfig(mox_channels=(red, MOX_NH3, MOX_OX) * 2)
world = OlfactionWorld(plume, device_config=calibration, stereo_baseline_m=0.04)
```

Prefer explicit per-unit sensitivity and uncertainty data from your instrument.
Do not infer concentration accuracy from a plausible response trace. The
`fast_modulated` profile transfers response times from a published experimental
regime; it does not turn an arbitrary packaged MOX sensor into that hardware.
The NumPy/Torch configuration and observable drift choices are documented in
[SCENTIENCE_MODELS.md](SCENTIENCE_MODELS.md).

[scd41.json](../configs/scd41.json) demonstrates a CO₂ instrument with an
explicit atmosphere. Change the profile to choose SCD30 or SCD40.
[emstat_dual.json](../configs/emstat_dual.json) demonstrates two EC circuits;
its coefficients are marked illustrative. See [sensor profiles](SENSOR_PROFILES.md)
for calibration, electrode wiring, current ranges and electronics limits.

## Timing and reproducibility

Advance transport once with `world.step(dt)`. Call `world.read` once per named
sensor tick; reads advance instrument state and are not idempotent. An omitted
read period uses the most recent nonzero world step. `truth` and plotting are
read-only. Pass a name for each independent device; its RNG seed is stable
regardless of the order in which devices are first read.

Pass `heading` for planar stereo or `orientation_wxyz` for a 3D mount. Both
cannot be supplied together. `world.reset(seed=N)` replays transport and
instrument streams; `reset()` starts another episode without requesting replay.
Backend RNGs differ, so CPU/GPU stochastic trajectories are compared
statistically rather than bit-for-bit.

`load_navigation(path)` accepts `PlumeNavConfig`, including optional per-episode
`wind_speed_range` and `source_strength_range`. Partial JSON objects overlay
defaults. An explicit `plume` object replaces the default navigation plume
configuration with `FilamentPlumeConfig` defaults plus its supplied fields.
`config_dict(cfg)` exports dataclass settings; `world.configuration()` records
the resolved plume/device settings for run records. `world.sensor_diagnostics()`
returns the last instrument diagnostics without another sensor tick.
