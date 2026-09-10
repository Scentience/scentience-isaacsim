"""Optional Gymnasium registration; import this module before gym.make()."""

from gymnasium.envs.registration import register, registry

if "Scentience-PlumeNav-v0" not in registry:
    register("Scentience-PlumeNav-v0", entry_point="scentience_olfaction.envs.plume_nav:PlumeNavEnv")
