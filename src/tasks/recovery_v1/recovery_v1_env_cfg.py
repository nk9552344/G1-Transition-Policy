"""Recovery-v1: full floor-recovery task configuration.

Extends transition-v3 so the robot learns to stand up from a completely
fallen position (supine, prone) in addition to all the bent-upright starting
configurations introduced in v3.

Key additions over v3
---------------------
Initial states
  Eight templates sampled uniformly each episode (2 fallen + 2 sitting + 4 bent = 25/25/50%):
    Fallen:   supine, prone                          (base_z ~= 0.25 m)
    Sitting:  sitting_low (40° lean), sitting_high (30° lean)  (base_z 0.28–0.38 m)
    Bent:     home, knees_bent, squat, deep_squat   (FK-verified heights)

  The sitting templates give the policy direct experience of the sit-to-stand
  phase without requiring it to first discover the full push-up sequence. They
  use the "fallen" type in the reset function (tilt quaternion + random joints).

  The 50% bent fraction matters: 50% fallen caused complete training failure at
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
  torso_height_reward (+3.0, target=0.90 m, std=0.50 m)
    Anti-oscillation anti-local-optimum signal. Position-based reward on the
    CHEST (torso_link) height. The robot cannot farm this reward by oscillating
    its waist -- oscillation earns only the reward at the mean height. The only
    way to earn significantly more is to SUSTAIN a higher torso position, which
    requires arm support from the floor.

    Replaced torso_upward_velocity (velocity-based, max(0, v_z)) which was
    gameable by waist oscillation: the upward half-cycle earned reward, the
    downward half earned 0 -- net positive over full cycle.

    Replaced orientation_rate (+1.5, max(0, d(proj_gz)/dt)) for the same reason:
    pitch oscillation earned reward on every forward half-cycle. At weight 1.5
    the oscillation benefit (~0.5-1.0/step) far exceeded the body_ang_vel cost.

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
      # Unified list: 2 fallen + 2 sitting + 4 bent = 8 templates (25 / 25 / 50 %).
      "all_pose_configs": ALL_POSE_CONFIGS,
      # Scatter position (same as v3).
      "xy_pos_range": 0.5,       # +/-0.5 m cell scatter
      "yaw_range": math.pi,      # full 360 deg initial yaw
      # Joint noise for fallen templates (all joints treated equally).
      # 0.6 rad (was 0.3): G1 push-up shoulder_pitch is ~0.6-0.8 rad above
      # the default 0.35 rad. At ±0.3 the arm never started in push-up range.
      # At ±0.6 the arm regularly starts near the position needed to reach the
      # floor, making the push-up motion discoverable during exploration.
      "fallen_joint_perturbation": 0.6,  # +/-0.6 rad -- wider arm exploration
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

  # -- Add torso_height_reward (replaces torso_upward_velocity) -------------
  #
  # torso_upward_velocity was velocity-based: max(0, v_z). Any waist oscillation
  # that moves the torso upward during the positive half-cycle earns reward,
  # even if it immediately collapses. The robot discovered it could wiggle its
  # torso to farm this reward without ever recovering (the "torso wiggling" bug).
  #
  # torso_height_reward is position-based: exp(-(torso_z - target)^2 / std^2).
  # Oscillation earns only the reward at the mean height. The robot MUST hold
  # its torso at a sustained higher position -- which requires arm support.
  #
  # body_names left empty; G1 config sets it to "torso_link".
  cfg.rewards["torso_height_reward"] = RewardTermCfg(
    func=mdp.torso_height_reward,
    weight=3.0,
    params={
      "target_height": 0.90,  # m -- G1 torso_link when standing ~= 0.88-0.92 m
      "std": 0.50,            # m -- gradient from 0.15 m (flat) to 0.90 m (upright)
      "asset_cfg": SceneEntityCfg("robot", body_names=()),  # set per-robot
    },
  )

  # orientation_rate was REMOVED.
  #
  # orientation_rate rewarded max(0, d(proj_gz)/dt) -- instantaneous angular
  # velocity toward upright. Like torso_upward_velocity, this can be farmed by
  # oscillation: the robot earns reward on every forward-pitch half-cycle and
  # 0 on the backward half. At weight 1.5 the oscillation benefit (~0.5-1.0/step)
  # far exceeds the body_ang_vel penalty (~0.12/step). It compounded with
  # torso_upward_velocity to make waist-wiggling very profitable.
  #
  # orientation_recovery (position-based, already registered above) provides the
  # same gradient toward upright without rewarding transient oscillations.

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
  # flat_gate_threshold=-0.85 keeps arm guidance active through the sit-to-stand
  # transition (31° from upright), extending past the old -0.7 cutoff (45°).
  cfg.rewards["arm_reach_down"] = RewardTermCfg(
    func=mdp.arm_reach_down,
    weight=1.5,
    params={
      "height_gate": 0.60,             # m -- suppress once hand is above mid-recovery
      "flat_gate_threshold": -0.85,    # extended: active until 31° from upright
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
      "flat_gate_threshold": -0.85,    # extended: matches arm_reach_down
      "asset_cfg": SceneEntityCfg("robot", body_names=()),  # set per-robot
    },
  )

  # -- Stand-up phase rewards ------------------------------------------------
  # Two rewards guide the sit-to-stand transition after the upper body is raised.
  #
  # head_height_reward: The head (geom inside torso_link at +0.43 m local Z)
  #   is at ~0.90 m when sitting up at 45° but at ~1.33 m when standing.
  #   This sharper gradient than torso alone motivates the robot to actually
  #   STAND rather than stay in the seated-with-upper-body-raised local optimum.
  #   head_z ≈ torso_z + 0.43 × (-proj_gz) -- computed analytically.
  #   body_names left empty; G1 config sets it to "torso_link".
  cfg.rewards["head_height_reward"] = RewardTermCfg(
    func=mdp.head_height_reward,
    weight=2.0,
    params={
      "target_height": 1.30,   # m -- G1 standing head ~= 1.30-1.35 m
      "std": 0.60,             # m -- gradient from 0.15 m (flat) to 1.35 m (upright)
      "head_offset": 0.43,     # m -- G1 head geom center above torso_link origin
      "asset_cfg": SceneEntityCfg("robot", body_names=()),  # set per-robot
    },
  )

  # feet_proximity_reward: when pelvis is elevated (upper body raised), reward
  #   feet being horizontally close to the pelvis. This guides the knee-tuck-
  #   and-stand maneuver: legs extended forward in sitting position give ~0.6 m
  #   foot-to-pelvis XY distance; feet under body in squat give ~0.1-0.2 m.
  #   body_names left empty; G1 config sets it to ankle_roll_link bodies.
  cfg.rewards["feet_proximity_reward"] = RewardTermCfg(
    func=mdp.feet_proximity_reward,
    weight=2.0,
    params={
      "height_gate": 0.35,   # m -- activate once pelvis is above prone height
      # std=0.45 (widened from 0.30): the narrower Gaussian gives near-zero
      # gradient when feet are 0.6+ m from pelvis (typical sit-up distance),
      # leaving the robot with no signal for the first ~0.3 m of the knee-tuck.
      # 0.45 m provides a meaningful gradient from the very start of the tuck.
      "std": 0.45,
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
