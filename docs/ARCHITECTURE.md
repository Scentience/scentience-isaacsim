# Architecture

```
                      standalone Python / Gymnasium            Isaac Lab RL
                                   |                                |
   +----------------+     +---------------+                +----------------+
   | OlfactionWorld |     |  PlumeNavEnv  |                | OlfactorySensor|
   |  (5-line API)  |     |  (gymnasium)  |                |  (SensorBase)  |
   +-------+--------+     +-------+-------+                +-------+--------+
           |                      |                                |
           v                      v                                v
   +--------------------------------------------------+   +----------------+
   |            FilamentPlume  (NumPy = spec)          |   | WarpFilament-  |
   |  emitters -> release -> OU turbulence + meander   |   | Plume (GPU,    |
   |  -> advect -> slide on OccupancyGrid -> grow/decay|   | parity-tested) |
   +---------------------+----------------------------+   +----------------+
                         |  C(x, t) per species
                         v
   +--------------------------------------------------+
   |   Virtual Scentience V1 device (device_np)        |
   |   2x MiCS-6814 MOX | SCD4x CO2 | 2x EC | T/RH     |
   |   power law -> lag -> drift -> noise -> ADC       |
   +---------------------+----------------------------+
                         |  hardware-shaped channels
          +--------------+--------------+
          v              v              v
     BLE-schema      EpisodeRecorder    OIO (bout detect +
     bridge          (npz + json)       drift correction)
```

Module map: `chemistry/` species registry . `emitters/` point/line/box .
`plume/` NumPy reference transport . `transport/` Warp GPU twin . `geometry/`
occupancy voxelization/LoS/slide . `airflow/` uniform+meander, grid, potential
flow . `sensors/` MOX, EC, SCD4x, PID, device . `oio/` olfactory inertial
odometry . `envs/` + `agents/` Gymnasium benchmark . `bridge/` Scentience
client schema . `validation/` the realism gate . `provenance.py` evidence
levels on every constant.

Load-bearing invariants:
1. NumPy transport is the SPECIFICATION; Warp is the fast path;
   `tests/test_warp_parity.py` binds them.
2. Ground truth never enters a policy observation.
3. The realism gate (`tests/test_plume_gate.py`) runs in CI; a plume
   regression fails the build.
4. Core package imports with no Isaac, no GPU, no torch.
