# Isaac Sim 6 / Isaac Lab 3 integration

The maintainer verified the previous integration on Sim 6 / Lab 3 in September
2026. **This refactor still requires live Isaac validation.** Its verification
consists of CPU contract tests and official tagged-source inspection. See
[compatibility and evidence](ISAAC_COMPATIBILITY.md) for the live test matrix.

## Upstream contract and scope

The integration follows the [current upstream project generator](https://isaac-sim.github.io/IsaacLab/v3.0.0-beta2/source/overview/own-project/template.html),
launched with `./isaaclab.sh --new`. The old
[IsaacLabExtensionTemplate](https://github.com/isaac-sim/IsaacLabExtensionTemplate)
was archived in September 2025. It is not the current template.

Source inspection used official `v3.0.0-beta2`. Later Lab 3 revisions can change
backend APIs, so run the contract check against the exact installed revision.
The [Lab 3 migration guide](https://isaac-sim.github.io/IsaacLab/develop/source/migration/migrating_to_isaaclab_3-0.html)
explains the backend split. This olfactory sensor currently supports **PhysX**,
with torch observations; it does not claim Newton or OvPhysX support.

- `SensorBase._initialize_impl()` establishes clone count and Warp timestamps.
  The subclass calls it before allocating views or per-environment models.
- Physics views are acquired at PHYSICS_READY through
  `isaaclab_physx.physics.PhysxManager`. A rigid object exposes `root_view`;
  `root_physx_view` is a deprecated compatibility alias and is not used here.
- `_update_buffers_impl(env_mask)` receives a Warp boolean mask. Masks take
  priority over indices in `reset(env_ids=None, env_mask=None)`. Empty selections
  do nothing; unselected device/plume state and measurements are preserved.
- Standard Lab 3 sensor arrays may be `ProxyArray` objects with `.torch` views.
  The boundary helpers also accept actual torch tensors and Warp arrays.
- **All olfactory/Imu mount rotations and returned olfactory poses use xyzw.**
  Identity is `(0, 0, 0, 1)`. The old integration's wxyz identity and quaternion
  rolling must be changed. Do not pass the standalone world's explicitly named
  `orientation_wxyz` argument into an Isaac offset without reordering it.

The corresponding official sources are
[SensorBase](https://github.com/isaac-sim/IsaacLab/blob/v3.0.0-beta2/source/isaaclab/isaaclab/sensors/sensor_base.py),
[PhysX Imu](https://github.com/isaac-sim/IsaacLab/blob/v3.0.0-beta2/source/isaaclab_physx/isaaclab_physx/sensors/imu/imu.py),
and [RigidObject](https://github.com/isaac-sim/IsaacLab/blob/v3.0.0-beta2/source/isaaclab_physx/isaaclab_physx/assets/rigid_object/rigid_object.py).

## Attach to a robot

Launch `AppLauncher` before importing sensors, assets, scenes or PhysX APIs.
Place these entries in your existing `InteractiveSceneCfg` alongside the robot:

```python
from isaaclab.sensors import ImuCfg
from scentience_isaaclab.olfactory_sensor import OlfactorySensorCfg
from scentience_olfaction.plume.filament import FilamentPlumeConfig

nose = OlfactorySensorCfg(
    prim_path="{ENV_REGEX_NS}/Robot/base",  # choose your rigid link or fixed child
    update_period=0.0,  # integrate exposure/device physics every physics tick
    offset=OlfactorySensorCfg.OffsetCfg(
        pos=(0.15, 0.0, 0.08), rot=(0.0, 0.0, 0.0, 1.0)),
    left_probe_pos=(0.0, 0.02, 0.0),
    right_probe_pos=(0.0, -0.02, 0.0),
    species=("ethanol",),
    plume=FilamentPlumeConfig(species="ethanol", source_pos=(0.0, 0.0, 1.0)),
    device_backend="torch",
    transport_backend="warp",
)
imu = ImuCfg(
    prim_path="{ENV_REGEX_NS}/Robot/base", update_period=0.0,
    offset=ImuCfg.OffsetCfg(pos=(0.15, 0.0, 0.08), rot=(0.0, 0.0, 0.0, 1.0)),
)
```

Here the 4 cm stereo separation is an **example**, not a calibrated board
measurement. Probes default to coincident positions. Positions are in meters;
probe coordinates are relative to the mounted sensor frame. Both the ancestor
link pose and configured offset rotation affect the probes in full 3D.

A fixed child prim is resolved to its nearest rigid ancestor using Lab's clone
plan helper. The fixed child transform, configured offset, and parent world
pose are composed. The view must contain exactly one body per environment.
The child must be fixed relative to that body, with identical mount geometry
across clones; an independently animated child requires a different binding.

After constructing the scene, before its first measurement:

```python
scene["nose"].set_env_origins(scene.env_origins)
```

Each plume lives in its own environment-local domain. Its source/background and
sampling coordinates are not global stage coordinates. The origin mapping is a
translation; differently rotated/scaled clone frames are unsupported. A single
environment defaults to world origin, but should also bind its scene origin if
translated. Batched sampling without origins raises an error.

## Clocks, lazy sampling and reset

The normal physics loop is sufficient:

```python
scene.write_data_to_sim()
sim.step()
scene.update(sim.get_physics_dt())
reading = scene["nose"].data
```

`OlfactorySensor.update(dt)` advances transport once per physics update. Do not
call the removed `_step_plume()` or add a plume step in `_pre_physics_step`.
In normal DirectRLEnv and manager environments, scene updates already occur
inside the physics loop; adding a second update would double elapsed time.

SensorBase controls capture cadence. `data` calls `_update_outdated_buffers()`;
repeated reads at one timestamp cannot advance the device or draw fresh noise.
Actual device dt is current timestamp minus last capture timestamp, **not** the
configured period. A 50 ms sensor on a 30 ms step can therefore capture at
60 ms. If a lazy consumer skips reads, its next measurement integrates the
current input held over that entire elapsed interval. This aliases unobserved
whiffs and approximates the entire exposure history by its final concentration.
**For physical device dynamics, use `update_period=0.0` and construct the scene
with `lazy_sensor_update=False`.** Then integrate exposure at physics ticks and
let the device's internal sampling cadence hold its published values. Do not
set Lab update_period to five seconds to imitate a CO2 instrument: that skips
exposure history even though the manufacturer model already has a sample clock.
A nonzero Lab period is an explicit speed/accuracy approximation. Setting only
`lazy_sensor_update=False` cannot recover input between nonzero-period captures.

At time zero, pose is available but channel buffers are zero and `valid=False`.
An initial/reset read does not draw noise or advance device time. A positive
elapsed capture sets `valid=True`, `timestamp`, and `sample_dt`. Timestamps are
simulation **episode seconds per environment**, not wall time or one monotonic
clock shared by all episodes. External recorders should retain episode IDs.

Reset through the scene when resetting an environment. Direct sensor reset
only resets sensing/transport; it does not teleport the robot:

```python
nose.reset(env_ids=[0, 2])
nose.reset(env_mask=warp_boolean_mask)  # length num_envs; mask has priority
```

Selected transport and device state, cached truth/channels/wind, validity and
clocks are cleared. Unselected measurements and state remain untouched. Unit
variation follows `randomize_per_episode`; False preserves the current unit's
R0 calibration while clearing dynamics (the adapters retain R0 across the core
model's nominal-R0 reset). Other custom-factory calibration follows its reset
contract. Reset is not a promise to rewind an
RNG to the same episode. Test reproducibility using fresh instances and seeds.

## Species, CO2 basis and device backends

`species` declares the exact ordered input columns. Aliases are canonicalized;
empty lists, duplicate aliases and unknown species fail. All emitted and
background species must appear in this tuple. Requested species absent from
the plume are zero. Config objects are copied before canonicalization.

When `plume.background_ppm` includes CO2, truth includes that reservoir and
**device input is absolute ppm**. An unspecified CO2 device configuration is
resolved to absolute input with its initial ambient baseline set to the
reservoir. Explicit excess-mode configs are rejected to prevent double counting.
No background is subtracted. Without explicit CO2 background, the device's
configured input mode applies (legacy defaults use excess-above-ambient ppm).
For an absolute-input device, include the intended atmosphere in plume background.

| Selection | Implemented scope |
|---|---|
| `transport_backend="warp"` | One passive species and legacy point source per environment; uniform flow; supported same-species background. Mixtures, explicit emitter lists, buoyancy and decay fail. |
| `transport_backend="numpy"` | CPU reference plumes per environment; configured emitter mixtures, species properties and backgrounds. Copies observations to the simulation device. |
| `device_backend="torch"` | Batched configurable Scentience model. Other profiles require an explicitly supplied torch implementation. |
| `device_backend="numpy"` | Routes directly through `sensors.factory.create_sensor`; Scentience, SCD30/40/41 and configured EmStat Pico profiles. CPU loop with device transfers. |

The adapter does not automatically derive occupancy or CFD flow from USD.
Transport/backend choice does not establish manufacturer calibration, runtime
throughput, or sim-to-real validity. Review the repository's actual licenses
and model provenance when selecting a profile.

For example, a directly selectable NumPy CO2 sensor is:

```python
nose = OlfactorySensorCfg(
    prim_path="{ENV_REGEX_NS}/Robot/base",
    device_backend="numpy", device_profile="scd41", transport_backend="numpy",
    species=("carbon_dioxide",),
    plume=FilamentPlumeConfig(species="carbon_dioxide",
                              background_ppm={"carbon_dioxide": 420.0}),
)
```

EmStat Pico requires explicit calibrated analyte/cell configuration via
`device_config`. An arbitrary potentiostat does not identify chemical species
without a sensor cell and calibration.

The torch factory contract is:

```python
build_device(device_profile, sensor_profile, n_envs, device, randomize=True,
             species_names=("ethanol",), seed=0, config=None)
model.step(conc_ppm, dt, conc_ppm_2=None, env_ids=None)  # selected (K,S) -> (K,C)
model.reset(env_ids=None, randomize=True)
```

`env_ids` map selected input rows into device state; output preserves that order.
`dt` may be a positive scalar: the adapter groups rows with equal elapsed times.
Stereo arrays are independently sampled and the Warp adapter copies each result
before the next probe overwrites its shared output buffer. Device dtype is
respected. Metadata comes from `channel_names` (the known Scentience schema is
also supported); `cfg.channel_names` is an optional strict schema assertion.
No CUDA-only or zero-host-synchronization claim is made by these Python adapters.

`torch_device_factory` and `numpy_device_factory` accept a callable or
`package.module:function`. Custom NumPy factories use the core scalar factory
kwargs (`device_profile`, `sensor_profile`, `seed`, `config`), return
`channel_names`, `step(dict, dt, conc_ppm_2=dict)->dict`, and `reset()`.
Randomization for custom scalar profiles is factory/config-defined; selected
randomized resets reconstruct those units with a new seed.

## Standard Imu and policy observations

Use Lab 3's standard `ImuCfg`; the adapter does not instantiate an Isaac Sim
sensor extension. Lab 3 Imu exposes `ang_vel_b` (rad/s) and `lin_acc_b` (proper
acceleration, m/s²) in the **IMU frame**. Proper acceleration is zero in freefall
and +g upward at rest. Pose, linear velocity and projected gravity are not Imu
measurements. Use a separate Pva/estimator only where that information is justified.
See [official Imu data contract](https://github.com/isaac-sim/IsaacLab/blob/v3.0.0-beta2/source/isaaclab/isaaclab/sensors/imu/base_imu_data.py).

```python
from scentience_isaaclab.imu import read_imu
sample = read_imu(scene["imu"])
# Optional: orientation must correspond to this capture, in xyzw order.
corrected = read_imu(scene["imu"], gravity_w=(0.0, 0.0, -9.81),
                     orientation_w=imu_orientation_at_capture)
```

The second call returns `linear_acceleration = specific_force + R^-1 * gravity`.
The caller supplies actual simulation gravity and synchronized attitude. Neither
is fabricated from the lightweight Imu. Rotating output axes requires
`rotation_target_from_imu` and a target frame name; it does not relocate the
measurement origin or apply lever-arm compensation.

ImuData has no public capture timestamp in the inspected release. `read_imu`
reads SensorBase's `_timestamp_last_update` after forcing its lazy data access,
or accepts an explicitly supplied capture timestamp. Missing clocks fail;
wall-clock time is never substituted. Its `valid` flag means positive capture
time, not a claim that a reset derivative transient has settled.

**Beta2 IMU cadence:** its PhysX implementation differentiates velocity using
its last `update(dt)` interval. Use `ImuCfg(update_period=0.0)` and either read
Imu every physics step or set `lazy_sensor_update=False`. Sparse lazy reads
with that implementation can distort acceleration. Initial velocity changes
and resets can still produce derivative transients; discard warmup samples for
quantitative checks. Newer Lab 3 implementations may use solver acceleration;
check the installed source before changing cadence assumptions.

MDP terms include `gas_channels` (optional named subset), `gas_sample_age`,
`gas_sample_valid`, `imu_specific_force`, `imu_angular_velocity`, and IMU
age/validity. `wind_body` now actually rotates world wind into the **nose frame**;
`wind_world` preserves world axes. Wind requires a modeled physical anemometer
for deployment and is not automatically part of chemical/IMU hardware.
It represents world flow expressed in local axes, not robot-relative flow.

Ground truth is unallocated by default. `gas_ground_truth` requires both
`expose_ground_truth=True` and `privileged=True`. This explicit opt-in does not
inspect the actor graph; the task author must keep it out of actor inputs.
Use Lab's observation-manager history rather than advancing a history buffer
on every property read. See [training integration](TRAINING.md).

## Native Isaac Sim 6 Imu and OIO

For a Lab scene, the supplied `ImuCfg`/`read_imu` path is recommended. In a bare
Sim application the documented native runtime is
[`isaacsim.sensors.experimental.physics.IMUSensor`](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/py/source/extensions/isaacsim.sensors.experimental.physics/docs/index.html).
Its `get_data(read_gravity=...)` schema and the following caveat were checked
against [tagged Sim v6.0.0 source](https://github.com/isaac-sim/IsaacSim/blob/v6.0.0/source/extensions/isaacsim.sensors.experimental.physics/python/impl/imu_sensor.py):
the dictionary has no validity field and retains its previous contents if the
raw reading is invalid. Do not infer freshness/validity from finite numbers.

`read_sim_imu_frame` is a pure converter; it imports no Isaac Sim API and does
not call deprecated `get_current_frame`. It requires the caller's exact
`read_gravity` setting and corresponding raw validity. Its output has one batch
row. Inputs must use SI acceleration/angular velocity (use a meter-scale scene).
Native simulation timestamps are preserved; the adapter does not reset them
when your RL environment resets.

| Boundary | Quaternion ordering |
|---|---|
| This Lab 3 olfactory offset and `ImuCfg.OffsetCfg` | xyzw |
| `read_imu(..., orientation_w=...)` | xyzw |
| Native Sim 6 `IMU` authoring and `get_data()['orientation']` | wxyz |
| `read_sim_imu_frame(...).orientation_w` | xyzw (explicitly converted) |
| Standalone world's `orientation_wxyz` | wxyz |

After SimulationApp startup, author the native sensor under your rigid link,
then read inside a physics callback with no simulation step between these reads:

```python
from isaacsim.sensors.experimental.physics import IMU, IMUSensor
from scentience_isaaclab.imu import read_sim_imu_frame

native = IMUSensor(IMU.create(
    "/World/Robot/base/imu",
    translations=[[0.0, 0.0, 0.0]],
    orientations=[[1.0, 0.0, 0.0, 0.0]],  # native wxyz
))
# During physics updates, after PLAY:
raw = native.get_sensor_reading(read_gravity=True)
frame = native.get_data(read_gravity=True)
sample = read_sim_imu_frame(
    frame, read_gravity=True,
    is_valid=bool(raw.is_valid) and frame["time"] == raw.time,
    gravity_w=(0.0, 0.0, -9.81),  # use actual scene gravity
)
```

With `read_gravity=True`, `specific_force` is populated. Supplying `gravity_w`
additionally computes gravity-free `linear_acceleration` using native attitude.
With `read_gravity=False`, the input is already gravity-free: the converter does
not remove gravity twice. It can reconstruct proper acceleration only when
world gravity is supplied. Native orientation is simulator state and must not
silently become a deployable actor observation.

To replace the synthetic IMU in
[the OIO example](../examples/03_olfactory_inertial_odometry.py), use a valid
sample from either adapter with gravity-free acceleration. For a **level planar
robot** whose Imu and nose axes agree, the estimator call is:

```python
# oio is an OlfactoryInertialOdometry instance; baseline is a learned/calibrated
# clean-air chem_left_red ratio. Use one estimator and clock per environment.
now = float(sample.timestamp[0])
if not bool(sample.valid[0]) or (last_time is not None and now < last_time):
    oio.reset()
    last_time = None
elif last_time is not None and now > last_time:
    if sample.linear_acceleration is None:
        raise ValueError("OIO requires gravity-free acceleration")
    index = nose.data.channel_names.index("chem_left_red")
    result = oio.step(
        sample.linear_acceleration[0, :2].cpu().numpy(),
        float(sample.angular_velocity[0, 2]),
        nose.data.wind_b[0, :2].cpu().numpy(),
        max(baseline - float(nose.data.channels[0, index]), 0.0),
        now - last_time,
    )
if bool(sample.valid[0]):
    last_time = now
```

Initialize `last_time=None`, reset it and `oio` on each episode/reset, skip invalid
or repeated samples, and time-align chemical/wind measurements with the IMU.
Nonplanar locomotion needs attitude/frame and lever-arm handling beyond this 2D
estimator. A native orientation or simulator wind field supplies privileged
information unless a corresponding real estimator/anemometer exists. The native
adapter and OIO wiring are CPU-tested/documented paths, not a live native-sensor
or OIO accuracy claim.
