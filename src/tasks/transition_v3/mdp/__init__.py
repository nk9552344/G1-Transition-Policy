"""MDP components for transition-v3: bent-pose recovery training.

Extends transition-v2's MDP namespace with the custom bent-pose reset event.
All reward functions are re-exported from their original modules (no changes).
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

from src.tasks.transition_v2.mdp.rewards import (  # noqa: F401
  angular_velocity_convergence,
  hold_bonus,
  linear_velocity_convergence,
)

# Custom reset event — the main v3 addition.
from .events import reset_to_bent_pose  # noqa: F401
