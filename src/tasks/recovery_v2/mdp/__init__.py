"""MDP module for recovery-v2.

Re-exports everything from recovery-v1 mdp plus the new v2-specific symbols.
"""

from mjlab.envs.mdp import *  # noqa: F401, F403

# Observation helpers
from src.tasks.velocity.mdp.observations import (   # noqa: F401
    foot_contact,
    foot_contact_forces,
)

# Penalty/regularisation rewards shared across the task lineage
from src.tasks.velocity.mdp.rewards import (        # noqa: F401
    angular_momentum_penalty,
    body_angular_velocity_penalty,
    body_orientation_l2,
    self_collision_cost,
)

# Standing-skill rewards from the transition series
from src.tasks.transition.mdp.rewards import (      # noqa: F401
    both_feet_contact,
    joint_vel_penalty,
)

# Momentum-damping rewards (v2+)
from src.tasks.transition_v2.mdp.rewards import (   # noqa: F401
    angular_velocity_convergence,
    hold_bonus,
    linear_velocity_convergence,
)

# Recovery-v1 rewards (carried over unchanged)
from src.tasks.recovery_v1.mdp.rewards import (     # noqa: F401
    height_recovery,
    orientation_recovery,
    pose_convergence_gated,
    upward_base_velocity,
)

# Recovery-v2 events
from .events import (                               # noqa: F401
    ALL_POSE_CONFIGS,
    reset_to_any_fallen_pose,
)

# Recovery-v2 rewards (new)
from .rewards import orientation_velocity_reward    # noqa: F401

# Terminations
from .terminations import bad_orientation_while_elevated  # noqa: F401
