# Using the olfactory sensor in Isaac Lab

Step-by-step, in the style of vendor sensor extensions. Read
`ISAAC_COMPATIBILITY.md` first: the wrapper is **API-contract validated
(34 static checks) and executes 10/10 checks under genuine isaaclab 2.3.2
code, but has never run in a live Isaac Sim install**. This page tells you
how to be the first, safely.

## Requirements

| Component | Version |
|---|---|
| Isaac Sim | 5.1 (RTX GPU required -- see `ISAAC_COMPATIBILITY.md`) |
| Isaac Lab | 2.3.x (`_update_buffers_impl(env_ids)` API; 3.0 is refused on purpose) |
| Python | Isaac Sim's bundled interpreter |
| This package | `pip install -e .` into Isaac's Python, or add the repo root to `PYTHONPATH` |

## 1. Install

```bash
# from the repo root, using Isaac Lab's launcher python
./isaaclab.sh -p -m pip install -e /path/to/scentience-olfaction
```

## 2. Validate the install BEFORE using it

```bash
./isaaclab.sh -p /path/to/scripts/validate_install.py
```

All five checks must pass. Check 2 failing with `env_mask` means you are on
Isaac Lab 3.0 -- stop; see `BRANCHING.md`. Until this passes on your machine,
treat everything below as unverified.

## 3. Attach the sensor to a robot

The sensor is a standard `SensorBase`: give it the prim to ride on, and read
`data` after each step. Channel order is the Scentience device schema
(`chem_left_red ... ec2`).

```python
from isaaclab.utils import configclass
from isaaclab.scene import InteractiveSceneCfg
from scentience_isaaclab.olfactory_sensor import OlfactorySensorCfg

@configclass
class MySceneCfg(InteractiveSceneCfg):
    # ... your robot, terrain, lights ...
    nose = OlfactorySensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",   # the link that carries it
        update_period=0.05,                       # 20 Hz, decoupled from physics
        sensor_profile="fast_modulated",          # or "packaged_slow" -- state it!
        species=("ethanol",),
    )
```

Reading it (e.g. in an observation term or a debug loop):

```python
sensor = scene["nose"]
channels = sensor.data.channels          # (num_envs, 11) -- THE observation
wind     = sensor.data.wind_w            # (num_envs, 3) simulated anemometer
```

Ground truth (`data.concentration_gt`) exists for rewards/eval only; the
`mdp.gas_ground_truth` term warns loudly if you wire it into an actor.

## 4. Observation terms (manager-based envs)

```python
from scentience_isaaclab import mdp
# in your ObservationsCfg group:
#   gas   = ObsTerm(func=mdp.gas_channels,  params={"asset_cfg": SceneEntityCfg("nose")})
#   wind  = ObsTerm(func=mdp.wind_body,     params={"asset_cfg": SceneEntityCfg("nose")})
```

## 5. The RL task

Importing `scentience_isaaclab.tasks` registers
`Isaac-PlumeNav-Scentience-v0` (DirectRLEnv). Its observation mirrors the
standalone Gymnasium env so policies are comparable across both.

## 6. Runtime verification (be the first, and tell us)

```bash
./isaaclab.sh -p /path/to/scripts/verify_in_isaac.py --steps 2000
```

This stands up a minimal scene, logs device channels against ground truth,
writes `verify_in_isaac.npz`/`.png`, and prints a summary. If it runs, please
paste the console output into `ISAAC_COMPATIBILITY.md` (and open a PR or
issue) -- that single paste moves the integration from "contract-validated"
to "live-validated" for everyone.

## What is validated today, without a live install

| Layer | Evidence |
|---|---|
| Every symbol/signature this wrapper assumes | 34 static checks vs the real `isaaclab==2.3.2` wheel (`scripts/check_isaaclab_contract.py`) |
| Sensor + Cfg + mdp + task cfg + gym registration EXECUTE under genuine isaaclab code | 10/10 (`scripts/check_isaaclab_binding.py`, kit runtime stubbed) |
| The physics and device models underneath | 78-test CPU suite incl. the realism gate |
| PhysX views, prim binding, rendering, timeline | **nothing** -- needs the live run above |

## Troubleshooting

Isaac-specific entries 10-11 in `TROUBLESHOOTING.md` (wrong Isaac Lab
version; non-RTX GPU). Everything else there applies inside Isaac unchanged.
