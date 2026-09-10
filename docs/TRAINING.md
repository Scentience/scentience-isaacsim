# Training with chemical and inertial observations

## Runnable standalone reference

Use [scripts/train_sb3.py](../scripts/train_sb3.py) with
[configs/navigation.json](../configs/navigation.json) for the implemented CPU
navigation environment. It supports PPO, SAC and TD3 with saved observation
history and normalization. This trains planar navigation, not Isaac robot
actuation or a locomotion/flight controller.

```bash
python scripts/train_sb3.py --algorithm ppo --config configs/navigation.json --steps 100000 --out runs/ppo_trial
python scripts/train_sb3.py --evaluate runs/ppo_trial --episodes 20 --seed 100
```

Use a new output directory for each experiment. Keep the environment config,
seed, history length, algorithm, normalization and checkpoint together. For
SAC/TD3 select `--algorithm sac` or `--algorithm td3`. Inspect `--help` for the
current standalone command interface and [SETUP.md](../SETUP.md) for dependencies.

## Isaac task status

The former `Isaac-PlumeNav-Scentience-v0` registration has been removed. Its
class did not spawn a robot/sensor or apply actions, declared 11 observations
but produced 8, used world wind as body wind, and had no runner configuration.
Registering it made an incomplete sketch appear trainable.

The integration now provides a robot-agnostic sensor, standard Imu adapters,
MDP terms, and a **usable explicit registry helper** for completed robot tasks.
It does not register a replacement generic robot/controller or ship arbitrary
runner hyperparameters as a tested robot policy. The fixed-root live smoke is
sensor verification, not an RL task.

Generate an external task using the [official Lab 3 generator](https://isaac-sim.github.io/IsaacLab/v3.0.0-beta2/source/overview/own-project/template.html):

```bash
./isaaclab.sh --new
```

Select a direct or manager-based workflow and the runner libraries you need.
Keep the generated task-specific runner configs and scripts. For each robot,
implement and verify these components before starting training:

1. Its asset, rigid mounting link, articulation actuators and action mapping.
   A navigation policy for a quadruped/humanoid commonly commands an existing
   locomotion controller; a drone needs a flight controller. Chemical sensing
   does not supply either controller.
2. Scene origins, local source/emitter placement, boundaries and observations.
   Bind `scene["nose"].set_env_origins(scene.env_origins)` before first capture.
3. Rewards, success and failure conditions, time-limit truncation, action scaling,
   and selective reset of robot, sensor, transport and observation history.
4. Runner observation groups, declared dimensions, normalization, rollout horizon,
   and checkpoint/evaluation settings appropriate to the actual sensor schema.

See [mounting and physics-loop integration](ISAAC_INTEGRATION.md). The sensor
advances transport in `scene.update(physics_dt)`; no extra pre-action plume step
is needed. For source-distance rewards, compare positions in the same local or
world frame, including each environment origin.

## Common runner registry

`task_kwargs` and `register_task` use the same loader keys as the
[official Lab 3 task registry](https://github.com/isaac-sim/IsaacLab/blob/v3.0.0-beta2/source/isaaclab_tasks/isaaclab_tasks/direct/cartpole/__init__.py).
Imports do not register any unfinished task; duplicate IDs are rejected.

| Helper runner name | Isaac registry key | Config supplied by your completed task |
|---|---|---|
| `rsl_rl` | `rsl_rl_cfg_entry_point` | Runner config class |
| `rl_games` | `rl_games_cfg_entry_point` | Runner YAML or supported config entry |
| `skrl` | `skrl_cfg_entry_point` | Algorithm-specific config |
| `sb3` | `sb3_cfg_entry_point` | SB3 config |

This example belongs in your external project's task package, after replacing
the module/class names with those generated and implemented in that project:

```python
from scentience_isaaclab.tasks import register_task

register_task(
    "Isaac-MyRobot-Scentience-v0",
    env_entry_point="my_robot.tasks.navigation:NavigationEnv",
    env_cfg_entry_point="my_robot.tasks.navigation:NavigationEnvCfg",
    runner_cfgs={
        "rsl_rl": "my_robot.tasks.agents.rsl_rl_ppo_cfg:NavigationPPORunnerCfg",
        "sb3": "my_robot.tasks.agents:sb3_ppo_cfg.yaml",
    },
)
```

Do not register the same ID again if the generator already registered it. You
can instead use `task_kwargs(env_cfg_entry_point, runner_cfgs)` to replace the
existing `gym.register(..., kwargs=...)` configuration. The helper validates
entry-point structure and known runner names; it does not fabricate missing
config modules, verify external controllers, or install a runner.

Import your task package in the generated train/play script before task lookup.
Then use that external project's runner script, for example:

```bash
python scripts/rsl_rl/train.py --task Isaac-MyRobot-Scentience-v0 --headless
python scripts/rsl_rl/play.py --task Isaac-MyRobot-Scentience-v0 --checkpoint /path/to/checkpoint.pt
```

Those commands run in the **generated external project**, not this repository.
Use the scripts/flags generated for your installed version; runner availability
and wrappers are described by the official
[workflow guide](https://isaac-sim.github.io/IsaacLab/v3.0.0-beta2/source/overview/reinforcement-learning/rl_existing_scripts.html)
and [library comparison](https://isaac-sim.github.io/IsaacLab/v3.0.0-beta2/source/overview/reinforcement-learning/rl_frameworks.html).
PPO is a useful first experiment with parallel simulation; SAC/TD3 comparisons
are available in the standalone runner. No algorithm is established here as
best for plume navigation, and runner support does not imply every algorithm
works with every Isaac task or observation structure.

## Observation history and privileged information

Chemical sensor response, sampling delays and intermittent plumes make temporal
context useful. Isaac already implements per-term history, so use the
[ObservationTermCfg](https://github.com/isaac-sim/IsaacLab/blob/v3.0.0-beta2/source/isaaclab/isaaclab/managers/manager_term_cfg.py)
fields rather than introducing another read-driven history buffer:

```python
from isaaclab.managers import ObservationGroupCfg, ObservationTermCfg, SceneEntityCfg
from isaaclab.utils.configclass import configclass
from scentience_isaaclab import mdp

@configclass
class PolicyCfg(ObservationGroupCfg):
    gas = ObservationTermCfg(
        func=mdp.gas_channels, params={"asset_cfg": SceneEntityCfg("nose")},
        history_length=8, flatten_history_dim=True,
    )
    gas_age = ObservationTermCfg(func=mdp.gas_sample_age)
    gas_valid = ObservationTermCfg(func=mdp.gas_sample_valid)
    gyro = ObservationTermCfg(func=mdp.imu_angular_velocity)
    accelerometer = ObservationTermCfg(func=mdp.imu_specific_force)
    imu_valid = ObservationTermCfg(func=mdp.imu_sample_valid)

    def __post_init__(self):
        self.concatenate_terms = True
        self.enable_corruption = False
```

With C chemical channels this group has `8*C + 9` flattened features. A
Scentience profile has 11 channels; an SCD profile has one. Infer dimensions
from the configured schema instead of retaining the old hard-coded 11-element
observation space. Adding/removing terms or history changes the dimension.
History advances at the observation-manager/control cadence, and can contain
repeated held sensor values. Age/validity expose that fact. Let the manager own
its selective reset and initial-history behavior; direct environments must
explicitly manage their own buffers and reset only the affected rows.

The accelerometer term is proper acceleration, including its gravity response.
All inertial axes belong to the mounted Imu, not implicitly the robot base.
Prefer equal Imu/nose mounting axes, or make any frame conversion explicit.
For both chemical dynamics and the inspected beta2 Imu, use update_period=0 and
`lazy_sensor_update=False`. Let manufacturer models enforce their internal
output cadence. A slow Lab period integrates the final exposure across the
whole skipped interval and can miss intervening whiffs; see the integration guide.

Do not feed `concentration_gt`, source pose, native Sim orientation or Pva ground
truth to a deployable actor. Privileged critic/reward/evaluation code may request
truth explicitly, but `privileged=True` cannot mechanically prove that an actor
never consumes it. Wind observations require a physical anemometer for transfer.
A world wind field and heading/map prior also provide information beyond an IMU.

Compare history durations in seconds, sensor profile, update/control periods,
source and wind distributions, and reset/warmup policy across experiments.
Evaluate held-out seeds/configurations and report success, time/path efficiency,
collisions and failure modes; a successful smoke or brief training run is not
convergence or sim-to-real evidence.
