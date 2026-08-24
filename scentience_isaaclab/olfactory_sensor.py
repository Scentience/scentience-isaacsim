"""
Isaac Lab olfactory sensor.

STATUS: ported to and validated against Isaac Lab 3.0.x / Isaac Sim 6.0.x
(scripts/validate_installl.py, scripts/verify_in_isaac.py). Originally
written against Issac Lab 2.3.x. See git history for that version if you 
need to pin to 2.3.x and Isaac Sim 5.1 instead.

Design notes that are load-bearing:

  * `data` MUST call `_update_outdated_buffers()`.  That is the lazy-evaluation
    contract; skipping it returns stale buffers with no error.
  * `_initialize_impl` MUST call `super()._initialize_impl()`.  It sets
    `_num_envs`, `_device`, `_sim_physics_dt`, and the timestamp bookkeeping
    (including the clone-plan-based environment count in Isaac Lab 3.0.x).
  * PhysX handles do not exist before timeline PLAY.  Everything touching
    `_view` belongs in `_initialize_impl`, not `__init__`.
  * `get_transforms()` returns quaternions xyzw; Isaac Lab math utilities want
    wxyz.  Hence the `.roll(1, dims=-1)`.  Getting this wrong yields a sensor
    that is subtly mis-rotated and passes every smoke test.
  * `cfg.update_period` already implements fixed-rate decoupling in simulated
    seconds, vectorised.  Do not imp by hand an accumulator.

Isaac Lab 3.0 changed `_update_buffers_impl(env_ids)` to 
`_update_buffers_impl(env_mask: wp.array), and its own built-in sensors now
store data as warp arrays behind a .torch accessor and route PhysX views through
isaaclab_physx.physics.PhysicsManager. This sensor keeps its own
OlfactorySensorData as plain torch tensors (simpler for a template, and this 
package's plume/device models are torch/numpy native). It just converts the
incoming `env_mask` to a torch bool mask with wp.to_torch()
and uses it exactly like the old env_ids for indexing.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

try:  # Isaac Lab is absent in CI; the physics core must stay importable.
    import warp as wp
    import isaaclab.sim as sim_utils  # noqa: F401  (availability probe)
    import isaaclab.utils.math as math_utils
    from isaaclab.sensors import SensorBase, SensorBaseCfg
    from isaaclab.utils.configclass import configclass

    """
    Note: isaaclab_physx.physics.PhysxManager is intentionally not 
    imported here. It pulls in omni.phyisics.sensors which, like all
    deep isaacsim/omni submodules, may only be imported after
    SimulationApp has booted the Kit runtime.
    Importing it at module level would make this whole module fail to
    import and silently fall back to _HAS_ISAAC = False at any time it
    is imported before the app exists, even with Isaac Lab is genuinely
    installed. It is imported instead inside _initialize_impl, which only
    ever runs after PHYSICS_READY.
    """

    _HAS_ISAAC = True
except ImportError:  # pragma: no cover
    _HAS_ISAAC = False
    SensorBase = object  # type: ignore
    SensorBaseCfg = object  # type: ignore

    def configclass(c):  # type: ignore
        return dataclass(c)


@dataclass
class OlfactorySensorData:
    """Everything the sensor exposes. Ground truth is deliberately separate."""

    pos_w: torch.Tensor | None = None
    """(N, 3) sensor world position."""

    channels: torch.Tensor | None = None
    """(N, C) simulated device channels -- THE POLICY OBSERVATION.
    Ordering is `OlfactorySensorCfg.channel_names`, matching the Scentience
    BLE/Sockets schema so a consumer cannot distinguish sim from hardware."""

    concentration_gt: torch.Tensor | None = None
    """(N, S) ground-truth ppm per species. For reward shaping, labels, and
    evaluation ONLY. Wiring this into an observation term makes the task
    trivially solvable and the resulting policy untransferable. There is an
    assertion against exactly that in `mdp/observations.py`."""

    wind_w: torch.Tensor | None = None
    """(N, 3) local wind -- the simulated anemometer. Real, and useful:
    upwind-surge behaviour needs it."""


if _HAS_ISAAC:

    class OlfactorySensor(SensorBase):
        """Vectorised olfactory sensor. One plume instance per environment."""

        cfg: "OlfactorySensorCfg"

        def __init__(self, cfg: "OlfactorySensorCfg"):
            super().__init__(cfg)
            self._data = OlfactorySensorData()
            self._plume = None
            self._device_model = None

        # -------------------------------------------------------- properties
        @property
        def data(self) -> OlfactorySensorData:
            self._update_outdated_buffers()  # lazy-eval contract; do not remove
            return self._data

        @property
        def num_instances(self) -> int:
            return self._view.count

        # ------------------------------------------------------------ set-up
        def _initialize_impl(self):
            super()._initialize_impl()  # sets _num_envs, _device, timestamps

            from isaaclab_physx.physics import PhysxManager as SimulationManager

            self._physics_sim_view = SimulationManager.get_physics_sim_view()
            self._view = self._physics_sim_view.create_rigid_body_view(
                self.cfg.prim_path.replace(".*", "*")
            )

            from scentience_olfaction.plume.filament import FilamentPlumeConfig
            from scentience_olfaction.transport.filament_warp import WarpFilamentPlume
            from scentience_olfaction.sensors.scentience_v1 import build_device

            n = self._view.count
            self._plume = WarpFilamentPlume(
                FilamentPlumeConfig(), n_envs=n, device=str(self._device)
            )
            self._device_model = build_device(
                self.cfg.device_profile, self.cfg.sensor_profile, n_envs=n,
                device=self._device, randomize=self.cfg.randomize_per_episode,
            )

            c, s = len(self.cfg.channel_names), len(self.cfg.species)
            z = lambda k: torch.zeros(n, k, device=self._device)  # noqa: E731
            self._data.pos_w, self._data.channels = z(3), z(c)
            self._data.concentration_gt, self._data.wind_w = z(s), z(3)

            self._offset_pos_b = torch.tensor(
                list(self.cfg.offset.pos), device=self._device).repeat(n, 1)
            self._offset_quat_b = torch.tensor(
                list(self.cfg.offset.rot), device=self._device).repeat(n, 1)

        # ------------------------------------------------------------ update
        def _update_buffers_impl(self, env_mask: "wp.array"):
            """
            IsaacLab 3 passes a warp bool mask and not a list of ids.
            Convert to a torch bool tensor (zero-copy) and use it exactly like
            the old env_ids for cleaner indexing.
            """
            env_ids = wp.to_torch(env_mask)
            if bool(env_ids.all()):
                env_ids = slice(None)

            transforms = wp.to_torch(self._view.get_transforms())
            pos_w, quat_w = transforms[env_ids].split([3, 4], dim=-1)
            # Careful with quaternion configs...
            # Took forever to debug.
            quat_w = quat_w.roll(1, dims=-1)  # xyzw -> wxyz
            p = pos_w + math_utils.quat_apply(quat_w, self._offset_pos_b[env_ids])
            self._data.pos_w[env_ids] = p

            # Warp samples in place; zero-copy on CUDA.
            self._plume.set_probes_torch(p)
            conc = self._plume.sample_torch()  # (n, S) ppm

            dt = self.cfg.update_period if self.cfg.update_period > 0.0 else self._sim_physics_dt
            self._data.channels[env_ids] = self._device_model.step(conc, dt)
            self._data.wind_w[env_ids] = self._plume.wind_torch()
            if self.cfg.expose_ground_truth:
                self._data.concentration_gt[env_ids] = conc

        def _step_plume(self, dt: float) -> None:
            """Advance transport. Call from the env's `_pre_physics_step`; the
            plume evolves on its own clock, independent of sensor sampling."""
            self._plume.step(dt)

        def reset(self, env_ids: Sequence[int] | None = None, env_mask: "wp.array | None" = None):
            super().reset(env_ids, env_mask)
            """
            Scentience plume/device models take a concrete list of ids (or None
            for "all") instead of a warp mask.
            So resolve any env_mask down to that type and shape.
            """
            if env_ids is None and env_mask is not None:
                mask_torch = wp.to_torch(env_mask)
                env_ids = None if bool(mask_torch.all()) else mask_torch.nonzero(as_tuple=True)[0].tolist()
            if self._plume is not None:
                self._plume.reset(env_ids)
            if self._device_model is not None:
                # Resample unit-to-unit variation. Without this, every episode
                # trains against the same virtual device.
                self._device_model.reset(env_ids, randomize=self.cfg.randomize_per_episode)

    @configclass
    class OlfactorySensorCfg(SensorBaseCfg):
        class_type: type = OlfactorySensor

        @configclass
        class OffsetCfg:
            pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
            rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)

        offset: OffsetCfg = OffsetCfg()

        species: tuple[str, ...] = ("ethanol",)
        device_profile: str = "scentience_v1"
        sensor_profile: str = "packaged_slow"
        """`packaged_slow` or `fast_modulated`. This choice changes how much
        plume structure survives into the observation by roughly 5x. State it
        in every result."""

        channel_names: tuple[str, ...] = (
            "chem_left_red", "chem_left_nh3", "chem_left_ox",
            "chem_right_red", "chem_right_nh3", "chem_right_ox",
            "co2_ppm", "temperature_c", "relative_humidity", "ec1", "ec2",
        )

        randomize_per_episode: bool = True
        """Resample R0, (A, beta), tau, drift per env on reset. Leaving this
        off produces virtual units far more consistent than any two real ones,
        which is the fastest way to train a policy that cannot transfer."""

        expose_ground_truth: bool = False
