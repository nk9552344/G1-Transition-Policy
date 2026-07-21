"""MDP components for recovery-v1: full floor-recovery training."""

from mjlab.envs.mdp import *  # noqa: F401, F403

from src.tasks.velocity.mdp.observations import (  # noqa: F401
  foot_contact,
  foot_contact_forces,
)

from src.tasks.velocity.mdp.rewards import (  # noqa: F401
  self_collision_cost,
)

from src.tasks.transition.mdp.rewards import (  # noqa: F401
  both_feet_contact,
  joint_vel_penalty,
)

from src.tasks.transition_v2.mdp.rewards import (  # noqa: F401
  hold_bonus,
)

# Recovery-v1 reward and observation functions.
from .events import reset_to_fallen_or_bent_pose   # noqa: F401
from .rewards import (                             # noqa: F401
  airborne_penalty,
  arm_reach_down,
  base_height_obs,
  feet_proximity_reward,
  head_above_feet_reward,
  height_gated_ang_vel_penalty,
  height_recovery,
  orientation_rate,
  orientation_recovery,
  pose_convergence_gated,
  pushup_support_reward,
  root_lin_vel_penalty,
  shank_orientation_reward,
  torso_height_reward,
  # elbow_push_from_ground — removed: velocity-based, replaced by pushup_support
  # head_height_reward     — removed: superseded by head_above_feet_reward
)
from .terminations import (                        # noqa: F401
  bad_orientation_while_elevated,
  joint_velocity_overflow,
)
