"""
Isaac Lab observation terms for the olfactory sensor.

The one rule this module enforces mechanically: GROUND TRUTH DOES NOT ENTER
THE POLICY OBSERVATION. `gas_channels` is the deployable observation;
`gas_ground_truth` exists for critics/reward/debug and refuses to be imported
into an actor observation group silently -- it warns loudly at construction.
"""

from __future__ import annotations

import warnings

import torch


def gas_channels(env, asset_cfg=None) -> torch.Tensor:
    """(num_envs, C) simulated device channels -- the deployable observation."""
    sensor = env.scene[asset_cfg.name if asset_cfg else "nose"]
    return sensor.data.channels


def gas_ground_truth(env, asset_cfg=None) -> torch.Tensor:
    """(num_envs, S) true ppm. Reward shaping / critic / eval ONLY."""
    warnings.warn(
        "gas_ground_truth is in an observation pipeline. If this feeds the "
        "ACTOR, the trained policy observes information no hardware can "
        "provide and will not transfer. Use gas_channels for the actor.",
        stacklevel=2)
    sensor = env.scene[asset_cfg.name if asset_cfg else "nose"]
    return sensor.data.concentration_gt


def wind_body(env, asset_cfg=None) -> torch.Tensor:
    """(num_envs, 3) simulated anemometer -- deployable (real robots carry
    anemometers; upwind surge needs it)."""
    sensor = env.scene[asset_cfg.name if asset_cfg else "nose"]
    return sensor.data.wind_w
