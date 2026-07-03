"""Reward functions for transition-v3.

Identical to transition-v2 rewards. All three new reward functions
(angular_velocity_convergence, linear_velocity_convergence, hold_bonus)
carry over unchanged. The tuning changes (std, weights, episode length)
are all in the env config, not here.
"""

from src.tasks.transition_v2.mdp.rewards import (  # noqa: F401
  angular_velocity_convergence,
  hold_bonus,
  linear_velocity_convergence,
)
