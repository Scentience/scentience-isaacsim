# Isaac compatibility

## Verified by reading tagged source

| Component | Version | How verified |
|---|---|---|
| Isaac Sim | 5.1 target, 6.0.1 current | cloned `isaac-sim/IsaacSim`, read `VERSION` and per-tag extension layouts |
| Isaac Lab | 2.3.x target | read `isaaclab/sensors/{sensor_base,imu}.py` at tag v2.3.2 |
| Warp | bundled `omni.warp.core` >= 1.13 | listed in `isaacsim.exp.base.kit` |

## Breaking changes routed around

- `isaacsim.sensors.physx` REMOVED in 6.0. Not used.
- `isaacsim.sensors.physics` deprecated in 6.0 in favour of
  `isaacsim.sensors.experimental.physics`. Not used -- we subclass Isaac Lab
  `SensorBase` instead, which is stable across both.
- Isaac Lab 2.x -> 3.0: `_update_buffers_impl(env_ids)` becomes
  `(env_mask: wp.array)`, and `data.field` becomes `data.field.torch`.
  **We target 2.3.x.** Porting is mechanical but has not been done.

## NOT YET VALIDATED -- read this before citing anything

`scentience_isaaclab/olfactory_sensor.py` has **never been executed inside
Isaac Sim**. It was written against the API by reading tagged source in an
environment with no Isaac installation. Until `scripts/validate_install.py`
passes on a real install, this file is a first draft.

Do not describe Isaac integration as working, tested, or validated until that
script passes and its output is pasted into this document with a date.

### What is validated, without Isaac

- Warp <-> NumPy physics parity (`tests/test_warp_parity.py`)
- OU stationary variance invariant across timestep -- the property that a
  dt-scaled (rather than sqrt(dt)-scaled) turbulence kick silently breaks
- Plume realism gate and the meander ablation (`tests/test_plume_gate.py`)
- Deterministic replay under seed
- Multi-environment stepping and partial reset

## v0.1 GPU-path scope

`transport/filament_warp.py` implements the fast common subset: single
species, point sources, no occupancy, no decay. The NumPy reference
(`plume/filament.py`) has all features and is the specification. GPU
multi-species + occupancy are roadmap v0.2. The parity test covers the
common subset only, deliberately.
