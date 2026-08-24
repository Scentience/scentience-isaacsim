"""
DirectRLEnv skeleton for PlumeNav in Isaac Lab. UNVALIDATED outside a live
install; the runnable, tested reference environment is
scentience_olfaction/envs/plume_nav.py. This class exists so the package
follows the standard Isaac Lab task layout and so v0.2's validation pass has
a concrete target rather than a blank file.
"""
from __future__ import annotations

try:
    import torch
    from isaaclab.envs import DirectRLEnv

    class PlumeNavRLEnv(DirectRLEnv):
        cfg = None  # PlumeNavEnvCfg, injected by gym registration

        def _setup_scene(self):
            super()._setup_scene()
            # robot articulation + olfactory sensor come from cfg.scene / cfg.nose

        def _pre_physics_step(self, actions: torch.Tensor):
            self._actions = actions.clamp(-1.0, 1.0)
            # advance the plume on the physics clock
            self.scene["nose"]._step_plume(self.physics_dt * self.cfg.decimation)

        def _apply_action(self):
            pass  # platform-specific: wheeled base / quadruped gait command

        def _get_observations(self) -> dict:
            nose = self.scene["nose"]
            obs = torch.cat([nose.data.channels[:, :6],
                             nose.data.wind_w[:, :2]], dim=-1)
            return {"policy": obs}

        def _get_rewards(self) -> torch.Tensor:
            pos = self.scene["nose"].data.pos_w[:, :2]
            src = torch.tensor(self.cfg.source_pos[:2], device=self.device)
            return -torch.linalg.norm(pos - src, dim=-1) * 0.01

        def _get_dones(self):
            pos = self.scene["nose"].data.pos_w[:, :2]
            src = torch.tensor(self.cfg.source_pos[:2], device=self.device)
            reached = torch.linalg.norm(pos - src, dim=-1) < self.cfg.success_radius
            timeout = self.episode_length_buf >= self.max_episode_length - 1
            return reached, timeout

except ImportError as _e:
    PlumeNavRLEnv = None
    IMPORT_ERROR = _e
    """Why the env is unavailable -- silent Nones cost debugging time."""
