"""MDP components for the transition-v2 task.

Re-exports the full mjlab.envs.mdp namespace and shared utilities from the
velocity and transition tasks, then layers in transition-v2-specific rewards.
"""

from mjlab.envs.mdp import *  # noqa: F401, F403

from src.tasks.velocity.mdp.observations import (  # noqa: F401
  foot_contact,
  foot_contact_forces,
)

from src.tasks.velocity.mdp.rewards import (  # noqa: F401
  angular_momentum_penalty,
  body_angular_velocity_penalty,
  body_orientation_l2,
  self_collision_cost,
)

from src.tasks.transition.mdp.rewards import (  # noqa: F401
  both_feet_contact,
  joint_vel_penalty,
  pose_convergence,
)

from .rewards import *  # noqa: F401, F403
