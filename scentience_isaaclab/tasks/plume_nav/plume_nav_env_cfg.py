"""
DirectRLEnv configuration for olfactory source localisation in Isaac Lab.

STATUS: written to the Isaac Lab 2.3.x DirectRLEnvCfg contract; UNVALIDATED
in a live install. The observation deliberately mirrors the standalone
Gymnasium env (envs/plume_nav.py): device channels + wind, never ground truth.
"""
from __future__ import annotations

try:
    from isaaclab.envs import DirectRLEnvCfg
    from isaaclab.scene import InteractiveSceneCfg
    from isaaclab.sim import SimulationCfg
    # Careful: this changes from `from isaaclab.utils import configclass`
    from isaaclab.utils.configclass import configclass
    from scentience_isaaclab.olfactory_sensor import OlfactorySensorCfg

    @configclass
    class PlumeNavEnvCfg(DirectRLEnvCfg):
        decimation = 4
        episode_length_s = 120.0
        action_space = 2            # [forward speed, turn rate]
        observation_space = 11      # 6 defl + ddt + wind(2) + heading(2)
        state_space = 0

        sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=4)
        scene: InteractiveSceneCfg = InteractiveSceneCfg(
            num_envs=1024, env_spacing=40.0, replicate_physics=True)

        nose: OlfactorySensorCfg = OlfactorySensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base",
            update_period=0.05,
            sensor_profile="fast_modulated",
        )
        success_radius = 1.0
        source_pos = (0.0, 0.0, 1.0)

except ImportError as _e:  # Isaac Lab absent: keep module importable for docs/tests
    PlumeNavEnvCfg = None
    IMPORT_ERROR = _e
    """Why the cfg is unavailable -- silent Nones cost debugging time."""
