"""Isaac manager observation terms with explicit sensor frames and truth access.

Use ObservationTermCfg.history_length for policy history; the observation manager
owns its cadence and selective reset. These functions do not append on reads.
"""
from __future__ import annotations

import torch

from ..imu import read_imu


def _sensor(env, asset_cfg, default):
    return env.scene[asset_cfg.name if asset_cfg is not None else default]


def gas_channels(env, asset_cfg=None, channel_names=None) -> torch.Tensor:
    """Device channels (N,C), optionally selected by name in requested order."""
    data = _sensor(env, asset_cfg, "nose").data
    if channel_names is None:
        return data.channels
    if not channel_names or len(set(channel_names)) != len(channel_names):
        raise ValueError("channel_names selection must be nonempty and unique")
    return data.channels[:, [data.channel_names.index(name) for name in channel_names]]


def gas_ground_truth(env, asset_cfg=None, *, privileged=False, right=False) -> torch.Tensor:
    """True ppm (N,S); explicit privileged=True for critic/reward/evaluation.

    This opt-in prevents accidental use, but cannot inspect the caller's actor
    graph. The task author must still keep privileged data out of actor inputs.
    """
    if not privileged:
        raise ValueError("ground truth requires privileged=True; never use for the actor")
    data = _sensor(env, asset_cfg, "nose").data
    value = data.concentration_right_gt if right else data.concentration_gt
    if value is None:
        raise ValueError("set expose_ground_truth=True to use privileged concentration")
    return value


def wind_body(env, asset_cfg=None) -> torch.Tensor:
    """World airflow expressed in the mounted nose frame (N,3).

    This is a simulated anemometer, not implied by chemical hardware or an IMU.
    It is not relative airflow compensated for the robot's translational speed.
    """
    return _sensor(env, asset_cfg, "nose").data.wind_b


def wind_world(env, asset_cfg=None) -> torch.Tensor:
    return _sensor(env, asset_cfg, "nose").data.wind_w


def gas_sample_age(env, asset_cfg=None) -> torch.Tensor:
    """(N,1) age in simulated seconds; useful when control outpaces sensing."""
    from .._torch import as_torch

    sensor = _sensor(env, asset_cfg, "nose")
    data = sensor.data
    return (as_torch(sensor._timestamp) - data.timestamp).clamp_min(0).unsqueeze(-1)


def gas_sample_valid(env, asset_cfg=None) -> torch.Tensor:
    return _sensor(env, asset_cfg, "nose").data.valid.float().unsqueeze(-1)


def imu_angular_velocity(env, asset_cfg=None) -> torch.Tensor:
    """(N,3), rad/s in the standard Imu's mounted sensor frame."""
    return read_imu(_sensor(env, asset_cfg, "imu")).angular_velocity


def imu_specific_force(env, asset_cfg=None) -> torch.Tensor:
    """(N,3), m/s²; includes gravity response (+g upward at rest)."""
    return read_imu(_sensor(env, asset_cfg, "imu")).specific_force


def imu_sample_valid(env, asset_cfg=None) -> torch.Tensor:
    return read_imu(_sensor(env, asset_cfg, "imu")).valid.float().unsqueeze(-1)


def imu_sample_age(env, asset_cfg=None) -> torch.Tensor:
    from .._torch import as_torch

    sensor = _sensor(env, asset_cfg, "imu")
    sample = read_imu(sensor)
    return (as_torch(sensor._timestamp) - sample.timestamp).clamp_min(0).unsqueeze(-1)
