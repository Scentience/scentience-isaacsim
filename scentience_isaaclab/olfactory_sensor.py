"""PhysX olfactory sensor targeting user-tested Isaac Sim 6 / Isaac Lab 3.

New changes require the live checks in scripts/verify_in_isaac.py. All rotations
use Lab 3's xyzw convention. See docs/ISAAC_INTEGRATION.md.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from ._torch import as_torch, indices, quaternion, quat_apply, quat_inverse_apply, quat_mul
from .backends import build_backends

# Broken/pre-AppLauncher Isaac imports must fail visibly. Pure adapters and
# MDP terms remain importable without Isaac; this actual subclass requires it.
from isaaclab.sensors import SensorBase, SensorBaseCfg
from isaaclab.utils.configclass import configclass


@dataclass
class OlfactorySensorData:
    pos_w: torch.Tensor | None = None                 # (N,3) sensor origin
    quat_w: torch.Tensor | None = None                # (N,4) sensor -> world, xyzw
    probe_pos_w: torch.Tensor | None = None           # (N,2,3), left/right
    channels: torch.Tensor | None = None              # (N,C) device observations
    concentration_gt: torch.Tensor | None = None      # (N,S), left; opt-in
    concentration_right_gt: torch.Tensor | None = None
    wind_w: torch.Tensor | None = None                # (N,3) world flow velocity
    wind_b: torch.Tensor | None = None                # (N,3) sensor-frame flow velocity
    timestamp: torch.Tensor | None = None             # (N,) episode seconds at capture
    sample_dt: torch.Tensor | None = None             # (N,) actual elapsed capture time
    valid: torch.Tensor | None = None                 # (N,) false until dt > 0 after reset
    channel_names: tuple[str, ...] = ()
    species_names: tuple[str, ...] = ()


class OlfactorySensor(SensorBase):
    """One device and isolated local plume per environment; any rigid robot link.

    prim_path may name a rigid body or fixed child frame. Bind cloned scene
    origins with set_env_origins(scene.env_origins) before sampling.
    scene.update(physics_dt) advances transport once. Do not also step the plume
    in a DirectRLEnv action hook.
    """
    cfg: "OlfactorySensorCfg"

    def __init__(self, cfg):
        self._data = OlfactorySensorData()
        self._plume = self._device_model = None
        self._env_origins_input = None
        super().__init__(cfg)

    @property
    def data(self):
        self._update_outdated_buffers()
        return self._data

    @property
    def num_instances(self):
        return self._view.count

    def set_env_origins(self, origins):
        """Bind (N,3) translations of local plume domains into the USD world.

        Clones must have identical orientation/scale. This is explicit because
        SensorBase does not own InteractiveScene's origins. Call before the
        first capture, either before or after PHYSICS_READY.
        """
        value = as_torch(origins)
        if value.ndim != 2 or value.shape[1] != 3 or not bool(torch.isfinite(value).all()):
            raise ValueError("env_origins must be finite (N,3) translations")
        self._env_origins_input = value.detach().clone()
        if self._plume is not None:
            self._bind_origins()

    def _bind_origins(self):
        value = self._env_origins_input
        if value is not None and value.shape != (self._num_envs, 3):
            raise ValueError(f"expected {self._num_envs} env_origins, got {value.shape}")
        self._origins = (torch.zeros(self._num_envs, 3, device=self._device) if value is None
                         else value.to(device=self._device, dtype=torch.float32))

    def _initialize_impl(self):
        super()._initialize_impl()
        import isaaclab.sim as sim_utils
        from isaaclab_physx.physics import PhysxManager

        if sim_utils.SimulationContext.instance().physics_manager is not PhysxManager:
            raise RuntimeError("OlfactorySensor currently supports Isaac Lab PhysX only")
        if not math.isfinite(self.cfg.update_period) or self.cfg.update_period < 0:
            raise ValueError("update_period must be finite and nonnegative")
        self._physics_sim_view = PhysxManager.get_physics_sim_view()
        expr, fixed_pos, fixed_quat = self._resolve_rigid_body_ancestor_expr()
        self._view = self._physics_sim_view.create_rigid_body_view(expr.replace(".*", "*"))
        if self._view.count != self._num_envs:
            raise RuntimeError(f"prim_path must resolve one rigid body per environment: "
                               f"{self._view.count} bodies for {self._num_envs} environments")
        n = self._num_envs
        self._offset_pos = torch.tensor(self.cfg.offset.pos, device=self._device).float()
        self._offset_quat = quaternion(self.cfg.offset.rot, device=self._device)
        if self._offset_pos.shape != (3,) or self._offset_quat.shape != (4,):
            raise ValueError("offset must contain one position (3,) and rotation (4,)")
        if fixed_pos is not None:
            fixed_q = quaternion(fixed_quat, device=self._device)
            self._offset_pos = (torch.tensor(fixed_pos, device=self._device)
                                + quat_apply(fixed_q, self._offset_pos))
            self._offset_quat = quat_mul(fixed_q, self._offset_quat)
        self._probes = torch.tensor((self.cfg.left_probe_pos, self.cfg.right_probe_pos),
                                    device=self._device, dtype=torch.float32)
        if self._probes.shape != (2, 3) or not bool(torch.isfinite(self._probes).all()):
            raise ValueError("left/right_probe_pos must be finite xyz vectors")
        if not bool(torch.isfinite(self._offset_pos).all()):
            raise ValueError("offset position must be finite")
        self._plume, self._device_model, species = build_backends(self.cfg, n, self._device)
        self._bind_origins()
        names = tuple(self._device_model.channel_names)
        self._data = OlfactorySensorData(
            pos_w=torch.zeros(n, 3, device=self._device),
            quat_w=torch.tensor((0., 0., 0., 1.), device=self._device).repeat(n, 1),
            probe_pos_w=torch.zeros(n, 2, 3, device=self._device),
            channels=torch.zeros(n, len(names), device=self._device, dtype=self._device_model.dtype),
            concentration_gt=(torch.zeros(n, len(species), device=self._device)
                              if self.cfg.expose_ground_truth else None),
            concentration_right_gt=(torch.zeros(n, len(species), device=self._device)
                                    if self.cfg.expose_ground_truth else None),
            wind_w=torch.zeros(n, 3, device=self._device),
            wind_b=torch.zeros(n, 3, device=self._device),
            timestamp=torch.zeros(n, device=self._device),
            sample_dt=torch.zeros(n, device=self._device),
            valid=torch.zeros(n, dtype=torch.bool, device=self._device),
            channel_names=names, species_names=species,
        )

    def update(self, dt, force_recompute=False):
        if not math.isfinite(dt) or dt < 0:
            raise ValueError("dt must be finite and nonnegative")
        if self.is_initialized and dt > 0:
            self._plume.step(dt)
        super().update(dt, force_recompute)

    def _update_buffers_impl(self, env_mask):
        ids = indices(self._num_envs, self._device, env_mask=env_mask)
        if not ids.numel():
            return
        if self._num_envs > 1 and self._env_origins_input is None:
            raise RuntimeError("call nose.set_env_origins(scene.env_origins) before batched sampling")
        transforms = as_torch(self._view.get_transforms())[ids]
        body_p, body_q = transforms[:, :3], quaternion(transforms[:, 3:])
        p = body_p + quat_apply(body_q, self._offset_pos)
        q = quat_mul(body_q, self._offset_quat)
        probes = p[:, None, :] + quat_apply(q[:, None, :], self._probes)
        self._data.pos_w[ids], self._data.quat_w[ids] = p, q
        self._data.probe_pos_w[ids] = probes
        now = as_torch(self._timestamp)[ids]
        elapsed = now - as_torch(self._timestamp_last_update)[ids]
        # Initial/reset reads do not advance dynamics, draw noise or claim a sample.
        positive = elapsed > 0
        ids, probes, q, now, elapsed = (ids[positive], probes[positive], q[positive],
                                      now[positive], elapsed[positive])
        if not ids.numel():
            return
        conc, wind = self._plume.sample(probes - self._origins[ids, None, :], ids)
        if conc.shape != (len(ids), 2, len(self._data.species_names)):
            raise ValueError("transport must return (selected_envs, 2, species) concentrations")
        channels = self._device_model.step(conc[:, 0], elapsed,
                                           conc_ppm_2=conc[:, 1], env_ids=ids)
        if channels.shape != (len(ids), len(self._data.channel_names)):
            raise ValueError("device must return (selected_envs, channels), in channel_names order")
        self._data.channels[ids] = channels
        self._data.wind_w[ids], self._data.wind_b[ids] = wind, quat_inverse_apply(q, wind)
        self._data.timestamp[ids], self._data.sample_dt[ids] = now, elapsed
        self._data.valid[ids] = True
        if self.cfg.expose_ground_truth:
            self._data.concentration_gt[ids] = conc[:, 0]
            self._data.concentration_right_gt[ids] = conc[:, 1]

    def reset(self, env_ids=None, env_mask=None):
        ids = indices(self._num_envs, self._device, env_ids, env_mask)
        if not ids.numel():
            return
        super().reset(env_ids=ids)
        self._plume.reset(ids)
        self._device_model.reset(ids, randomize=self.cfg.randomize_per_episode)
        for value in vars(self._data).values():
            if isinstance(value, torch.Tensor):
                value[ids] = 0
        self._data.quat_w[ids, 3] = 1


@configclass
class OlfactorySensorCfg(SensorBaseCfg):
    class_type: type = OlfactorySensor

    @configclass
    class OffsetCfg:
        pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
        rot: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)

    offset: OffsetCfg = OffsetCfg()
    # Relative to offset sensor frame: x forward, y left, z up by convention.
    # Coincident by default: measure hardware spacing before enabling stereo.
    left_probe_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    right_probe_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    species: tuple[str, ...] = ("ethanol",)
    plume: object | None = None  # FilamentPlumeConfig; coordinates local to each env
    transport_backend: str = "warp"
    device_backend: str = "torch"
    device_profile: str = "scentience_v1"
    sensor_profile: str = "packaged_slow"
    device_config: object | None = None
    torch_device_factory: object | None = None
    numpy_device_factory: object | None = None
    channel_names: tuple[str, ...] | None = None  # optional schema assertion
    seed: int = 0
    randomize_per_episode: bool = True
    expose_ground_truth: bool = False
