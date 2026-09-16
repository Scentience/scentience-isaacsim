"""Registration utilities for completed Isaac tasks; no import-time Gym side effects.

The former Isaac-PlumeNav-Scentience-v0 stub had no robot, controller or valid
observation dimension and has been removed. See docs/TRAINING.md.
"""
from .registry import RUNNER_KEYS, register_task, task_kwargs

__all__ = ["RUNNER_KEYS", "register_task", "task_kwargs"]
