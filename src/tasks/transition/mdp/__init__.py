"""MDP components for the transition-to-neutral-standing task.

Re-exports the full mjlab.envs.mdp namespace, shared utilities from the
velocity task's MDP (to avoid code duplication), and the transition-specific
reward functions defined in this package.
"""

from mjlab.envs.mdp import *  # noqa: F401, F403

# Shared observation utilities reused from the velocity task.
from src.tasks.velocity.mdp.observations import (  # noqa: F401
  foot_contact,
  foot_contact_forces,
)

# Shared reward utilities reused from the velocity task.
from src.tasks.velocity.mdp.rewards import (  # noqa: F401
  angular_momentum_penalty,
  body_angular_velocity_penalty,
  body_orientation_l2,
  self_collision_cost,
)

from .rewards import *  # noqa: F401, F403
