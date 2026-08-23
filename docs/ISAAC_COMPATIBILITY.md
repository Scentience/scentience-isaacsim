# Isaac compatibility status

**STATUS: API-contract validated against isaaclab 2.3.2; NOT yet executed in a
live Isaac Sim install (blocked by GPU hardware).**
Last checked: 2026-08-23.

## Validation tiers

| Tier | What | Status |
|---|---|---|
| 1. Static contract | Every symbol/signature/field this code assumes -- including the full API surface of `scripts/verify_in_isaac.py` -- checked against the real `isaaclab==2.3.2` wheel from PyPI. `scripts/check_isaaclab_contract.py`. | **34/34 PASS** (2026-08-23) |
| 2. Executed binding | The WHOLE wrapper executed by GENUINE isaaclab 2.3.2 code: sensor + Cfg, mdp observation terms, the DirectRLEnv task cfg (constructs under the real `DirectRLEnvCfg`, 25 fields), and gym registration with resolving entry points; only the Omniverse kit runtime is stubbed. `scripts/check_isaaclab_binding.py`. | **10/10 PASS** (2026-08-23) |
| 3. Live install | `scripts/validate_install.py` then `scripts/verify_in_isaac.py` inside a running Isaac Sim 5.1 + Isaac Lab 2.3.x (see `ISAAC_USAGE.md`). | **NOT RUN** -- needs RTX hardware |

Tier 2 caught a real bug before any live install existed: `OlfactorySensorCfg`
bound `class_type` by post-hoc class-attribute assignment, but `@configclass`
had already baked the `None` default into `__init__`, so every INSTANCE had
`cfg.class_type = None` and sensor construction would have failed on first use.
Fixed by binding at definition (upstream Isaac Lab convention). This is
exactly the failure `validate_install.py` check 4 exists to catch -- it now
passes at tier 2.

Notable tier-1 facts: `SensorBase._update_buffers_impl` takes `env_ids`
(the 2.x API this code targets, not 3.0's `env_mask`); `imu.py` lines 199-200
in the real wheel are line-for-line the same `get_transforms()` + `roll(1)`
xyzw->wxyz handling our sensor uses; isaaclab itself imports
`SimulationManager` from the same path we do (14 files). The PyPI `isaaclab`
distribution is 2.3.2; the inner package reports `__version__` 0.54.2.

What tier 2 deliberately does NOT prove: PhysX views, prim binding, timeline
callbacks, rendering. Until tier 3 runs, treat runtime behaviour inside Isaac
as unverified.

---

Historical record of the tier-3 blocker follows (2026-08-21).

Per `README.md` and `CONTRIBUTING.md`, this file may only claim the Isaac
integration is validated once `scripts/validate_install.py` passes inside a
live install. It does not pass. Output is pasted verbatim below.

## Two separate blockers

### 1. The GPU cannot run Isaac Sim at all

NVIDIA's own checker (`C:\isaacsim\isaac-sim.compatibility_check.bat`) on this
machine, 2026-08-21:

```
[Warning] [omni.rtx] Skipping unsupported non-RTX GPU: NVIDIA GeForce GTX 1650 with Max-Q Design
[Warning] [omni.rtx] Skipping unsupported non-NVIDIA GPU: Intel(R) Iris(R) Plus Graphics
[Error]   [omni.rtx] No device could be created.
          - Your GPUs do not support RayTracing: DXR or Vulkan ray_tracing,
            or hardware is excluded due to performance.
[Warning] [omni.gpu_foundation_factory.plugin] RT-capable GPU not found, switching to compatibility mode
```

Against the Isaac Sim 5.1 minimum spec:

| Requirement | Minimum | This machine | |
|---|---|---|---|
| GPU | GeForce RTX 4080 | GeForce GTX 1650 Max-Q | FAIL -- no RT cores |
| VRAM | 16 GB | 3951 MB | FAIL |
| Driver (Windows) | 580.88 | 538.92 | FAIL |
| RAM | 32 GB | 32354 MB | pass |
| CPU cores | 4 | 4C / 8T i7-1065G7 | pass |
| Storage | 50 GB SSD | 614 GB free | pass |

The GPU row is not a driver-update problem. TU117 has no RT cores, so the
Omniverse RTX renderer cannot create a device on it at any driver version.

### 2. The installed Isaac Sim is the wrong major version for this code

`C:\isaacsim\VERSION` reports `6.0.1-rc.7+release.42383.32955d8d.gl`
(bundled Python 3.12.13, no Isaac Lab installed).

This repo targets **Isaac Lab 2.3.x / Isaac Sim 5.1**. Isaac Sim 6.0 pairs with
Isaac Lab 3.0, which changes `SensorBase._update_buffers_impl(env_ids)` to
`(env_mask: wp.array)` -- the exact break `scentience_isaaclab/olfactory_sensor.py:25`
documents and `validate_install.py` check 2 refuses. So the most recent
*compatible* Isaac Sim is **5.1**, not the 6.0.1-rc already on disk.

## validate_install.py output (2026-08-21)

Run as `C:\isaacsim\python.bat scripts\validate_install.py` with
`PYTHONPATH` set to the repo root:

```
[FAIL] isaacsim + isaaclab import, versions reported
       ModuleNotFoundError: No module named 'isaaclab'
[FAIL] SensorBase API shape matches what we subclass (2.3.x vs 3.0)
       ModuleNotFoundError: No module named 'isaaclab'
[FAIL] warp available and reports a CUDA device
       ModuleNotFoundError: No module named 'warp'
[FAIL] our sensor cfg constructs and binds class_type
       ModuleNotFoundError: No module named 'torch'
[FAIL] warp/numpy physics parity
       ModuleNotFoundError: No module named 'warp'

0/5 checks passed
Isaac integration is NOT validated. Do not update ISAAC_COMPATIBILITY.md.
```

Checks 3-5 fail only because Isaac Sim's bundled Python 3.12 has no
warp/torch; they pass in the project venv (see below). Checks 1-2 are the
substantive ones and cannot be run until an Isaac Lab 2.3.x install exists on
supported hardware.

## What DOES work on this machine

The core physics is unaffected -- it was designed to run with no Isaac and no
RTX, and it does:

```
pytest -m "not isaac"                       62 passed
warp.get_cuda_device_count()                1  ("cuda:0" GTX 1650, sm_75, mempool enabled)
```

Warp compute (CUDA sm_75) works fine on this card -- it is only the *RTX
renderer* that requires RT cores. So the GPU plume transport twin
(`scentience_olfaction/transport/filament_warp.py`) runs here even though
Isaac Sim will not.

## To unblock

1. Move to a machine with an RTX 4080-class GPU (16 GB VRAM) and driver >= 580.88.
2. Install **Isaac Sim 5.1** + **Isaac Lab 2.3.x** (not 6.0 / 3.0).
3. Re-run `scripts/validate_install.py` and replace this file's output.

Note: `.gitignore` contains `*_*.md`, so this filename is ignored by git.
Force-add it (`git add -f docs/ISAAC_COMPATIBILITY.md`) if it should be tracked.
