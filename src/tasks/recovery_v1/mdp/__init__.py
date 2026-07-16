"""MDP components for recovery-v1: full floor-recovery training.

Extends the v3 MDP namespace with two new modules:
  events  — reset_to_fallen_or_bent_pose (samples fallen + bent templates)
  rewards — orientation_recovery, height_recovery, pose_convergence_gated
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
)

from src.tasks.transition_v2.mdp.rewards import (  # noqa: F401
  angular_velocity_convergence,
  hold_bonus,
  linear_velocity_convergence,
)

# Recovery-v1 additions.
from .events import reset_to_fallen_or_bent_pose   # noqa: F401
from .rewards import (                             # noqa: F401
  airborne_penalty,
  arm_reach_down,
  base_height_obs,
  elbow_push_from_ground,
  feet_proximity_reward,
  head_above_feet_reward,
  head_height_reward,
  height_gated_ang_vel_penalty,
  height_recovery,
  orientation_recovery,
  pose_convergence_gated,
  pushup_support_reward,
  shank_orientation_reward,
  torso_height_reward,
)
from .terminations import (                                # noqa: F401
  bad_orientation_while_elevated,
  joint_velocity_overflow,
)
