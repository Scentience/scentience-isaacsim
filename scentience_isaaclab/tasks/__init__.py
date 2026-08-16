"""Gym task registration -- the standard Isaac Lab pattern: importing this
package registers the task IDs with gymnasium; training frameworks then use
`gym.make("Isaac-PlumeNav-Scentience-v0", cfg=...)`."""
import gymnasium as gym

gym.register(
    id="Isaac-PlumeNav-Scentience-v0",
    entry_point=f"{__name__}.plume_nav.plume_nav_env:PlumeNavRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point":
            f"{__name__}.plume_nav.plume_nav_env_cfg:PlumeNavEnvCfg",
    },
)
