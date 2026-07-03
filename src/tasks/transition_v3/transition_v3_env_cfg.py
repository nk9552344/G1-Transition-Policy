"""Transition-v3: bent-pose recovery task configuration.

Builds on transition-v2 and adds the ability to start from squatting / deep-
squatting configurations, training the policy to stand up to the neutral HOME
pose from significantly bent initial states.

Key additions over v2:
  - Custom reset event (reset_to_bent_pose) that samples from four leg-pose
    templates: home (25%), knees_bent (25%), squat (25%), deep_squat (25%).
    Each template has an FK-verified base height so feet start at ground level
    without ground penetration.
  - Wider pose_convergence std (0.4 vs 0.25): from a deep squat the mean joint
    error is ~0.8–1.5 rad; std=0.25 would give exp(-36) ≈ 0 gradient. std=0.4
    yields exp(-MSE/0.16) ≈ 0.25 gradient at max squat depth — enough to learn.
  - Slightly looser termination angle (75° vs 70°): the robot may lean forward
    during early stand-up; 75° avoids spurious terminations.
  - Longer episode (25 s vs 20 s): standing from a deep squat takes more time.

All v2 rewards (angular/linear velocity convergence, hold_bonus) carry over
unchanged; weight and episode-length adjustments are sufficient.
"""

import math

from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

import src.tasks.transition_v3.mdp as mdp
from src.tasks.transition_v2.transition_v2_env_cfg import make_transition_v2_env_cfg

# ── Pose templates ─────────────────────────────────────────────────────────────
# Each dict specifies absolute joint values (radians) for the bilateral leg
# joints and the pelvis height (metres) that places the ankle roll links at
# the same ground-contact level as the HOME standing configuration.
#
# Heights were computed via MuJoCo FK (g1.xml) by binary-searching the pelvis
# z such that the left ankle_roll_link z = HOME ankle_roll_link z ≈ 0.051 m.
#
#   Config        knee    hip_pitch  ankle   base_z
#   ─────────────────────────────────────────────────
#   home          0.300   -0.100    -0.200   0.8000   (HOME_KEYFRAME)
#   knees_bent    0.669   -0.312    -0.363   0.7725   (KNEES_BENT_KEYFRAME)
#   squat         1.200   -0.700    -0.500   0.6918
#   deep_squat    1.800   -1.000    -0.600   0.5616
# ───────────────────────────────────────────────────────────────────────────────

BENT_POSE_CONFIGS = [
  {"knee": 0.300, "hip_pitch": -0.100, "ankle": -0.200, "base_z": 0.8000},
  {"knee": 0.669, "hip_pitch": -0.312, "ankle": -0.363, "base_z": 0.7725},
  {"knee": 1.200, "hip_pitch": -0.700, "ankle": -0.500, "base_z": 0.6918},
  {"knee": 1.800, "hip_pitch": -1.000, "ankle": -0.600, "base_z": 0.5616},
]


def make_transition_v3_env_cfg():
  """Create transition-v3 task configuration (builds on v2)."""
  cfg = make_transition_v2_env_cfg()

  # ── Replace v2 reset events with single bent-pose reset ──────────────────
  # v2 uses two separate events: reset_base + reset_robot_joints.
  # v3 replaces both with a single comprehensive event that also sets base z.
  del cfg.events["reset_base"]
  del cfg.events["reset_robot_joints"]

  cfg.events["reset_robot"] = EventTermCfg(
    func=mdp.reset_to_bent_pose,
    mode="reset",
    params={
      # Pose templates: uniform sampling from all four.
      "bent_pose_configs": BENT_POSE_CONFIGS,
      # Base position/orientation scatter (same as v1/v2 reset_base).
      "xy_pos_range": 0.5,       # metres — robot placed anywhere in ±0.5 m cell
      "yaw_range": math.pi,      # full 360° initial yaw
      # Joint position noise around each template.
      "leg_perturbation": 0.10,  # ± rad on knee/hip/ankle (tight around template)
      "other_perturbation": 0.35,  # ± rad on arms/waist from default
      # Velocities (inherited from v2).
      "joint_vel_range": 0.15,   # ± rad/s initial joint velocities
      "lin_vel_range": 0.20,     # ± m/s initial base linear velocity (x, y)
      "ang_vel_range": 0.30,     # ± rad/s initial base angular velocity
      # SceneEntityCfg objects provide pre-resolved joint IDs.
      "knee_cfg": SceneEntityCfg("robot", joint_names=(".*_knee_joint",)),
      "hip_pitch_cfg": SceneEntityCfg("robot", joint_names=(".*_hip_pitch_joint",)),
      "ankle_cfg": SceneEntityCfg("robot", joint_names=(".*_ankle_pitch_joint",)),
      "asset_cfg": SceneEntityCfg("robot"),
    },
  )

  # ── Tune pose_convergence for larger initial errors ───────────────────────
  # std=0.25 gives exp(-36) ≈ 0 gradient from a deep squat (MSE ≈ 2.25 rad²
  # for knee alone). std=0.4 → exp(-MSE/0.16) ≈ 0.25 at max squat — useful.
  cfg.rewards["pose_convergence"].params["std"] = 0.4

  # ── Slightly looser termination ───────────────────────────────────────────
  # The robot leans forward during the early phase of standing from a squat.
  # 75° prevents spurious terminations while still catching real falls.
  cfg.terminations["fell_over"].params["limit_angle"] = math.radians(75.0)

  # ── Longer episode ────────────────────────────────────────────────────────
  # Standing up from a deep squat requires more time than a posture adjustment.
  cfg.episode_length_s = 25.0

  return cfg
