"""Recovery-v1: full floor-recovery task configuration.

Extends transition-v3 so the robot learns to stand up from a completely
fallen position (supine, prone) in addition to all the bent-upright starting
configurations introduced in v3.

Key additions over v3
---------------------
Initial states
  Six templates sampled uniformly each episode (2 fallen, 33% + 4 bent, 67%):
    Fallen:  supine, prone                         (base_z ~= 0.25 m)
    Bent:    home, knees_bent, squat, deep_squat   (FK-verified heights)

  The 67/33 split matters: 50% fallen caused complete training failure at
  17k iters because conflicting gradient signals (large actions for recovery,
  near-zero for balance) collapsed the single scalar std to 0.3 -- simultaneously
  too small for floor recovery and too large for standing stability.

Termination: height-gated fell_over (replaces the deleted termination)
  bad_orientation_while_elevated fires when BOTH:
    (a) body tilt > 75 deg  AND  (b) base height > 0.65 m.

  The height gate (0.65 m) is the critical fix:
    - Fallen starts (h ~= 0.10-0.15 m): gate closed -> never fires -> robot has
      the full 35 s episode to discover the get-up motion.
    - Upright/bent starts (h 0.77-0.80 m): gate open -> if the robot tips over
      the episode terminates immediately (same urgency as v3).
    - Mid-recovery (0.25-0.65 m): gate closed -> robot can pass through this
      range while still tilted without early termination. This is the critical
      fix over the original 0.50 m threshold, which punished the very trajectory
      needed for recovery (prone->bridge->squat crosses 0.50 m while tilted).
    - Near-standing (h > 0.65 m): gate open -> unstable intermediate stands
      that tip sideways are penalised.

Reward changes (over v3 / v2 inherited set)
  torso_upward_velocity (+3.0, height_gate=0.90 m, max_vel=1.5)
    Anti-local-optimum signal. Rewards upward velocity of the CHEST (torso_link),
    not the pelvis. A leg-bridge maneuver lifts the pelvis but keeps the chest
    on the floor -- yielding zero reward. Only arm-coordinated recovery (push-ups,
    rolling) that raises the chest generates positive reward. This breaks the
    leg-only local optimum that forms with upward_base_velocity.

  orientation_rate (+1.5)
    Immediate per-step gradient toward upright. Rewards the angular velocity
    component that analytically decreases proj_gz (rotating from flat to upright).
    Provides a signal before the static orientation_recovery can respond, speeding
    up discovery of the flip/roll motion.

  orientation_recovery (+3.0, std=1.0)
    Primary get-up signal. Gaussian on (proj_gz + 1.0)^2 -- distinguishes
    upright (0), flat (1), and inverted (4) unlike body_orientation_l2.

  height_recovery (+2.0, target=0.78 m, std=0.65 m)
    Secondary get-up signal. Gradient nonzero from 0.25 m fallen to 0.80 m.

  pose_convergence_gated (replaces pose_convergence, +1.5, std=0.5)
    Gated by (-proj_gz).clamp(0,1) so the pose reward is suppressed when the
    robot is flat -- prevents "stay flat in default joints" local optimum.

  body_orientation_l2 weight increased to -3.0 (was -2.0 in v3).

  body_ang_vel weight reduced to -0.03 (was -0.10 in v2/v3).
    Recovery requires deliberate rotational motion to flip from flat to upright.
    The inherited -0.10 penalty fought this rotation too strongly.

  angular_momentum weight reduced to -0.008 (was -0.025 in v2/v3).
    Same rationale: angular momentum is essential for dynamic get-up motions.

  joint_vel_penalty weight reduced to -0.008 (was -0.02 in v2/v3).
    Vigorous recovery requires fast joint motions, especially in the arms
    and legs during push-up or rolling phases.

  action_rate_l2 weight reduced to -0.02 (was -0.05 in v2/v3).
    Rapid action changes are needed to transition between recovery phases
    (push -> tuck -> stand). Over-penalising smoothness slows discovery.

  is_terminated KEPT from v3 at weight=-200.
    Since bad_orientation_while_elevated can now fire (for upright episodes),
    the termination penalty is meaningful and must remain.

  push_robot kept from v3 (interval 8-10 s).

Episode length
  35 s (extended from v3's 25 s).
"""

import math

from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg

import src.tasks.recovery_v1.mdp as mdp
from src.tasks.recovery_v1.mdp.events import ALL_POSE_CONFIGS
from src.tasks.transition_v3.transition_v3_env_cfg import make_transition_v3_env_cfg


def make_recovery_v1_env_cfg():
  """Create recovery-v1 task configuration (builds on v3)."""
  cfg = make_transition_v3_env_cfg()

  # -- Replace v3 bent-pose reset with unified fallen + bent reset -----------
  del cfg.events["reset_robot"]

  cfg.events["reset_robot"] = EventTermCfg(
    func=mdp.reset_to_fallen_or_bent_pose,
    mode="reset",
    params={
      # Unified list: 2 fallen (supine/prone) + 4 bent = 6 templates (33 / 67 %).
      "all_pose_configs": ALL_POSE_CONFIGS,
      # Scatter position (same as v3).
      "xy_pos_range": 0.5,       # +/-0.5 m cell scatter
      "yaw_range": math.pi,      # full 360 deg initial yaw
      # Joint noise for fallen templates (all joints treated equally).
      "fallen_joint_perturbation": 0.3,  # +/-0.3 rad -- moderate variety
      # Joint noise for bent templates (same as v3).
      "leg_perturbation": 0.10,          # +/-0.10 rad on knee/hip/ankle
      "other_perturbation": 0.35,        # +/-0.35 rad on arms/waist
      # Initial velocities (same as v3).
      "joint_vel_range": 0.15,   # +/-0.15 rad/s
      "lin_vel_range": 0.20,     # +/-0.20 m/s (x, y)
      "ang_vel_range": 0.30,     # +/-0.30 rad/s
      # SceneEntityCfg objects for leg-joint targeting in bent templates.
      "knee_cfg":      SceneEntityCfg("robot", joint_names=(".*_knee_joint",)),
      "hip_pitch_cfg": SceneEntityCfg("robot", joint_names=(".*_hip_pitch_joint",)),
      "ankle_cfg":     SceneEntityCfg("robot", joint_names=(".*_ankle_pitch_joint",)),
      "asset_cfg":     SceneEntityCfg("robot"),
    },
  )

  # -- Replace fell_over with height-gated version ---------------------------
  # Height threshold raised to 0.65 m (from original 0.50 m).
  #
  # The 0.50 m threshold created a punishment zone that blocked recovery:
  #   prone -> bridge -> squat trajectory crosses 0.50 m while still tilted
  #   -> episode terminates with -200 -> policy learned to AVOID raising
  #   the pelvis past 0.50 m -> settled on shallow leg-bridge below threshold.
  #
  # At 0.65 m:
  #   fallen starts   (h ~= 0.10-0.15 m): gate closed -> full episode
  #   mid-recovery    (0.15-0.65 m):      gate closed -> no early termination
  #   near-standing   (h > 0.65 m):       gate open   -> unstable stands terminated
  #   upright starts  (h 0.77-0.80 m):    gate open   -> same v3 safety signal
  cfg.terminations["fell_over"] = TerminationTermCfg(
    func=mdp.bad_orientation_while_elevated,
    params={
      "limit_angle": math.radians(75.0),
      "height_threshold": 0.65,   # m -- raised from 0.50 to clear mid-recovery path
    },
  )

  # -- Replace pose_convergence with gated version ---------------------------
  del cfg.rewards["pose_convergence"]

  cfg.rewards["pose_convergence_gated"] = RewardTermCfg(
    func=mdp.pose_convergence_gated,
    weight=1.5,
    params={
      "std": 0.5,
      "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
    },
  )

  # -- Add orientation_recovery reward ---------------------------------------
  cfg.rewards["orientation_recovery"] = RewardTermCfg(
    func=mdp.orientation_recovery,
    weight=3.0,
    params={
      "std": 1.0,
      "asset_cfg": SceneEntityCfg("robot"),
    },
  )

  # -- Add height_recovery reward --------------------------------------------
  cfg.rewards["height_recovery"] = RewardTermCfg(
    func=mdp.height_recovery,
    weight=2.0,
    params={
      "target_height": 0.78,
      "std": 0.65,
      "asset_cfg": SceneEntityCfg("robot"),
    },
  )

  # -- Add torso_upward_velocity reward (replaces upward_base_velocity) ------
  # Tracks the CHEST (torso_link) velocity, not the pelvis root.
  # body_names is left empty here; the G1 config sets it to "torso_link".
  # height_gate=0.90 m covers flat (torso ~= 0.15 m) through mid-squat
  # (torso ~= 0.75 m) without firing when fully standing (torso ~= 1.0 m).
  # max_vel=1.5 m/s reduces wild early-training oscillations vs. the old 2.0 m/s.
  cfg.rewards["torso_upward_velocity"] = RewardTermCfg(
    func=mdp.torso_upward_velocity,
    weight=3.0,
    params={
      "height_gate": 0.90,   # m -- set relative to torso_link, not pelvis
      "max_vel": 1.5,        # m/s -- tighter cap reduces wild pelvis swings
      "asset_cfg": SceneEntityCfg("robot", body_names=()),  # set per-robot
    },
  )

  # -- Add orientation_rate reward -------------------------------------------
  # Provides immediate gradient for rotational exploration.
  # Uses root projected_gravity_b and root_link_ang_vel_b -- no body_names needed.
  cfg.rewards["orientation_rate"] = RewardTermCfg(
    func=mdp.orientation_rate,
    weight=1.5,
    params={
      "asset_cfg": SceneEntityCfg("robot"),
    },
  )

  # ── Arm-based recovery rewards ─────────────────────────────────────────────
  # Two rewards create the push-up gradient chain:
  #
  #   arm_reach_down (dense, +1.5):
  #     Robot flat -> hands near waist (~0.2 m) -> Gaussian reward pulls them
  #     toward floor level. Provides gradient BEFORE contact so the robot is
  #     already motivated to reach down before it knows about push-ups.
  #
  #   elbow_push_from_ground (coupled dense, +3.5):
  #     Contact-gated: left arm contact gates left elbow velocity, right arm
  #     gates right. Robot must plant an arm on the floor AND extend the elbow
  #     upward simultaneously to earn reward. This forces the exact push-up
  #     sequence: reach -> plant -> push -> chest rises.
  #
  # Design note: a standalone arm_ground_contact binary reward was tried but
  # removed. The elbow subtree sensor fires passively when G1 lies prone at
  # default joint angles (forearm naturally hangs toward floor), giving +2.5/step
  # for lying still -- a strong local optimum. The contact requirement is already
  # enforced inside elbow_push_from_ground via the per-arm coupling.
  #
  # Both rewards are gated off when the robot is nearly upright (proj_gz < -0.7)
  # so they do not interfere with the standing-balance regime.

  # Dense: pull hands to floor (body_names set per-robot to wrist_yaw_link bodies).
  cfg.rewards["arm_reach_down"] = RewardTermCfg(
    func=mdp.arm_reach_down,
    weight=1.5,
    params={
      "height_gate": 0.60,   # m -- suppress once hand is above mid-recovery height
      "asset_cfg": SceneEntityCfg("robot", body_names=()),  # set per-robot
    },
  )

  # Coupled: elbow upward velocity ONLY when that arm is simultaneously on ground.
  # Weight 3.5: absorbs the intent of the removed arm_ground_contact reward (+2.5)
  # but only fires during actual push-up motion, not passive arm resting.
  cfg.rewards["elbow_push_from_ground"] = RewardTermCfg(
    func=mdp.elbow_push_from_ground,
    weight=3.5,
    params={
      "sensor_name": "arm_ground_contact",   # sensor added in robot config
      "height_gate": 0.70,   # m -- covers flat (elbow ~= 0.15 m) through mid-recovery
      "max_vel": 1.5,
      "asset_cfg": SceneEntityCfg("robot", body_names=()),  # set per-robot
    },
  )

  # -- Remove v2 stillness rewards that fight recovery motion ----------------
  #
  # angular_velocity_convergence (+0.7): rewards zero angular velocity.
  # linear_velocity_convergence (+0.4): rewards zero linear velocity.
  # Together they give the flat, static robot +1.1/step for doing nothing,
  # nearly cancelling body_orientation_l2 (-3.0) and killing recovery incentive.
  # Designed for standing balance damping; wrong for a recovery policy.
  del cfg.rewards["angular_velocity_convergence"]
  del cfg.rewards["linear_velocity_convergence"]

  # -- Reduce both_feet_contact to allow rolling motions ---------------------
  #
  # Rolling from supine requires feet to briefly leave the ground. The inherited
  # +0.5 penalises this by -0.5/step. Reduce to +0.2 so the foot-contact signal
  # does not block rolling while still rewarding stable placement when standing.
  cfg.rewards["both_feet_contact"].weight = 0.2

  # -- Increase body_orientation_l2 penalty ----------------------------------
  cfg.rewards["body_orientation_l2"].weight = -3.0

  # -- Reduce motion penalties that fight recovery rotation ------------------
  # body_ang_vel: recovery requires deliberate body rotation to flip from flat.
  # The inherited -0.10 from v2 (designed for standing balance) actively penalises
  # the rotational motion needed to get up. Reduce to -0.03.
  cfg.rewards["body_ang_vel"].weight = -0.03

  # angular_momentum: same rationale -- angular momentum is the mechanism of
  # dynamic get-up (swing arms/legs to build rotational energy). Reduce to -0.008.
  cfg.rewards["angular_momentum"].weight = -0.008

  # joint_vel_penalty: vigorous recovery needs fast joint motions (arm push-up,
  # leg swing). The inherited -0.02 penalises these too heavily. Reduce to -0.008.
  cfg.rewards["joint_vel_penalty"].weight = -0.008

  # action_rate_l2: recovery phases transition rapidly (push -> tuck -> stand).
  # Allow faster action changes by reducing from -0.05 to -0.02.
  cfg.rewards["action_rate_l2"].weight = -0.02

  # -- Longer episode --------------------------------------------------------
  cfg.episode_length_s = 35.0

  return cfg
