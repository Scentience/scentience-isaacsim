# Isaac compatibility and validation evidence

Target: **Isaac Sim 6.x / Isaac Lab 3.x, PhysX backend**.

The maintainer verified the baseline integration on Sim 6 / Lab 3.
The September 9, 2026 refactor has **not yet been validated in a live Isaac
runtime**; CPU tests and source inspection do not extend the baseline claim
to these changes.
Exact baseline patch versions and logs have not been captured in this document.

## Evidence for this refactor

| Check | Result / limit |
|---|---|
| Official Lab source | `v3.0.0-beta2` inspected, September 9, 2026. SensorBase, PhysX views, standard Imu, configclass/scene contracts and task registry reviewed. |
| Source checker | 4/4 checks pass against the official downloaded tag. No Isaac code executed. |
| Focused CPU tests | 32 tests pass: actual adapter/model math with explicit SensorBase/PhysX doubles, plus real CPU Warp and native Sim dictionary conversion. Not a Kit/PhysX validation. |
| Real binding checker | Pending in target runtime. Former Lab 2 Kit stubs retired. |
| Live cuboid sensor/Imu | Pending for this refactor. |
| Go2 / H1 / Crazyflie | Live mount/finite-data/subset-reset smoke targets implemented; each run pending. |
| Native Sim Imu | Pure dictionary converter checked against official Sim 6.0.0 docs/source; native runtime and OIO accuracy pending. |
| Isaac RL training | No trained or ready-to-train generic Isaac robot task claimed. Use completed robot tasks and the explicit registry helper. |

Inspected SensorBase SHA256:
`618ce99c742ac35c2fcbdbe29649930fb228bf16d460277636cd7d02daf19b18`.

## Reproduce and record

Without Isaac, using an official source checkout at your target revision:

```bash
python scripts/check_isaaclab_contract.py /path/to/IsaacLab
WARP_CACHE_PATH=/tmp/scentience-warp-cache python -m pytest -q tests/test_isaac_contract.py
```

The source checker also accepts Lab/PhysX wheel paths. It requires the Lab 3
source layout and deliberately rejects the old Lab 2 `env_ids` update API.
Do not substitute a passing stub harness for a live binding run.
The obsolete Lab 2 `setup_isaaclab_local.py`, `demo_isaaclab_sensor_local.py`
and `isaaclab_kit_stubs.py` were removed; their changelog mentions are historical.

On a supported Isaac host, install this project into the interpreter used by
`isaaclab.sh`, then run from the Isaac Lab checkout (replace `/repo`):

```bash
./isaaclab.sh -p /repo/scripts/validate_install.py --headless
./isaaclab.sh -p /repo/scripts/verify_in_isaac.py --headless --robot cuboid --out /repo/runs/cuboid
./isaaclab.sh -p /repo/scripts/verify_in_isaac.py --headless --robot go2 --out /repo/runs/go2
./isaaclab.sh -p /repo/scripts/verify_in_isaac.py --headless --robot h1 --out /repo/runs/h1
./isaaclab.sh -p /repo/scripts/verify_in_isaac.py --headless --robot crazyflie --out /repo/runs/crazyflie
```

`check_isaaclab_binding.py` is an alternate entry to the same real binding check.
It launches AppLauncher and uses no Kit stubs. Run the scene smoke separately.
The robot assets require NVIDIA asset access/cache. Use `--body-path` to adapt
modified USD layouts; an incorrect prim still fails at initialization.

The smoke defaults to three environments and 360 steps. Nose and standard Imu
share a rigid parent and nonzero translated/rotated offset. Robots use a fixed
root fixture and Crazyflie rotors start at zero speed: **this is not locomotion
or flight control validation**. It checks finite outputs, preserved stereo
spacing, idempotent reads, and isolation under index and Warp-mask resets.

Outputs are `.npz` records and `.json` metadata next to `--out`. Records include
physics time, per-environment capture time, validity, pose/probe positions,
channels, both true concentrations and Imu data. A zero gas maximum is reported
as a binding-only result; it is not treated as evidence of chemical response.
The old automatic plot has been removed because arbitrary manufacturer channel
schemas cannot safely assume a RED-channel deflection plot.

For each empirical run, record date, git commit, Sim/Lab versions, OS/GPU,
command, console output and output-file paths. Update individual rows only for
the configurations actually executed. A successful run does not establish
calibration, all robot compatibility, RL convergence or hardware transfer.

For mount/config changes and backend limitations see
[ISAAC_INTEGRATION.md](ISAAC_INTEGRATION.md). For task composition see
[TRAINING.md](TRAINING.md).
