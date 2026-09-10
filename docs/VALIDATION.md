# Validation record

Local validation of the September 9, 2026 changes used macOS arm64 and Python
3.11.6. No Isaac runtime or CUDA device was available on this host.

| Check | Observed result |
|---|---|
| Full non-Isaac suite, including slow statistical checks | 328 passed, 14 CUDA-dependent tests skipped; 83.95 seconds |
| Final focused checks after the collision endpoint correction | 131 passed: occupancy, research workflow, optional sensors and Isaac adapter contracts |
| Ruff and whitespace checks | Passed |
| Wheel and source distribution | Built successfully |
| Installed-wheel import | Core world and 11-channel reading worked outside the repository with optional imports blocked |
| Training commands | PPO, SAC and TD3 completed short CPU runs; SAC/TD3 ran beyond the learning-start threshold |
| Checkpoint evaluation | Saved PPO model and observation normalization reloaded and evaluated |
| Baseline benchmark | All three policies completed seeded episodes and wrote metrics/transition records |
| Sensor configuration examples | SCD41 and dual-cell EmStat configurations executed |
| Visual outputs | Plume/response figures inspected; Gym RGB rendering returned an image |
| Existing legal files | `LICENSE`, `LICENSE_MODEL`, `NOTICE` and `ACKNOWLEDGEMENTS` unchanged byte-for-byte |

The three Gymnasium checker warnings concern the physical-unit action space and
unbounded observation bounds. The SB3 training command normalizes actions and
observations. The short training runs establish executable learning/update and
save/reload paths; they do not measure policy quality.

Dependency versions used: NumPy 2.4.6, Torch 2.12.0, Warp 1.17.0, Gymnasium
1.3.0, Stable-Baselines3 2.9.0, Matplotlib 3.11.1, pytest 9.0.3 and Ruff 0.15.13.
These are an observed environment, not new minimum requirements or a lockfile.
The CI configuration also targets Python 3.10 and 3.12; those jobs have not been
run locally as part of this record.

### Scentience SDK integration follow-up

The optional `bridge` extra installed successfully with Scentience 2.2.2 and
Bleak 3.0.2. The fast non-Isaac regression run passed **342 tests**, with
14 CUDA-dependent skips and five slow tests excluded. The 14 focused BLE/SDK
tests passed, including the actual SDK's public sampling/JSON/logging paths
with a fake GATT transport and its local OVL mapper. No live hardware or
hosted inference was used. The offline SDK example, lint and package builds
also passed. See [SDK integration](SDK_INTEGRATION.md).

## Reproduce

```bash
python -m pip install -e ".[dev]"
python -m ruff check .
python -m pytest -q -m "not isaac"
python -m build
```

Use the commands in [the research workflow](RESEARCH_WORKFLOW.md) and
[training guide](TRAINING.md) to reproduce plotting, benchmarking and learning
smokes in new output directories. Test counts may change as the suite evolves.

## Outstanding validation

[Isaac compatibility](ISAAC_COMPATIBILITY.md) records the source checks and
ready-to-run live checks for cuboid, Go2, H1 and Crazyflie mounts with an IMU.
The revised integration still needs those checks on an Isaac Sim 6.x / Isaac
Lab 3.x host. CUDA transport/device parity and throughput also need a GPU run.
The Torch model currently prioritizes independent sensor state and fidelity;
no fused-kernel throughput claim is made.

Hardware calibration, controlled plume measurements, held-out navigation
evaluation and olfactory inertial odometry accuracy remain experimental work.
Manufacturer specifications and published scientific models guide the
simulation; passing software tests does not establish those empirical claims.
