"""MDP terms for Isaac Lab manager-based and direct envs.

Was empty, which made `mdp.gas_channels` unreachable through the package --
the exact path Isaac Lab observation-manager configs reference terms by.
Caught by scripts/check_isaaclab_binding.py check 8.
"""
from .observations import gas_channels, gas_ground_truth, wind_body

__all__ = ["gas_channels", "gas_ground_truth", "wind_body"]
