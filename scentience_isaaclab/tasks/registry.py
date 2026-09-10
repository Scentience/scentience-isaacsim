"""Explicit Gym/Isaac runner registration for a completed, external robot task.

No robot-independent controller or ready-to-train Isaac environment is claimed.
Entry-point keys follow the official Isaac Lab 3 task/template registry.
"""

from types import MappingProxyType

RUNNER_KEYS = MappingProxyType({
    "rsl_rl": "rsl_rl_cfg_entry_point",
    "rl_games": "rl_games_cfg_entry_point",
    "skrl": "skrl_cfg_entry_point",
    "sb3": "sb3_cfg_entry_point",
})


def task_kwargs(env_cfg_entry_point, runner_cfgs):
    """Build real Isaac loader kwargs; configs remain owned by the robot task.

    Values are 'module:ConfigClass' or 'package:runner_config.yaml' as accepted
    by upstream load_cfg_from_registry. This validates structure, not training
    suitability or whether optional runner packages are installed.
    """
    if not runner_cfgs:
        raise ValueError("provide at least one runner config for this task")
    unknown = set(runner_cfgs) - RUNNER_KEYS.keys()
    if unknown:
        raise ValueError(f"unknown runners {sorted(unknown)}; choose {tuple(RUNNER_KEYS)}")
    kwargs = {"env_cfg_entry_point": env_cfg_entry_point}
    kwargs.update({RUNNER_KEYS[key]: value for key, value in runner_cfgs.items()})
    for name, target in kwargs.items():
        if not isinstance(target, str) or len(target.split(":")) != 2 or not all(target.split(":")):
            raise ValueError(f"{name} must be 'module:attribute' or 'package:filename.yaml'")
    return kwargs


def register_task(task_id, *, env_entry_point, env_cfg_entry_point, runner_cfgs):
    """Register once; refuse to overwrite an existing user task."""
    import gymnasium as gym

    kwargs = task_kwargs(env_cfg_entry_point, runner_cfgs)
    if not isinstance(env_entry_point, str) or ":" not in env_entry_point:
        raise ValueError("env_entry_point must be 'module:EnvClass'")
    if task_id in gym.registry:
        raise ValueError(f"task {task_id!r} is already registered")
    gym.register(id=task_id, entry_point=env_entry_point, disable_env_checker=True, kwargs=kwargs)
