"""Recovery-v1: full floor-recovery task configuration.

Extends transition-v3 so the robot learns to stand up from a completely
fallen position (supine, prone) in addition to all the bent-upright starting
configurations introduced in v3.

Key additions over v3
─────────────────────
Initial states
  Six templates sampled uniformly each episode (2 fallen, 33 % + 4 bent, 67 %):
    Fallen:  supine, prone                         (base_z ≈ 0.25 m)
    Bent:    home, knees_bent, squat, deep_squat   (FK-verified heights)

  The 67/33 split matters: 50 % fallen caused complete training failure at
  17 k iters because conflicting gradient signals (large actions for recovery,
  near-zero for balance) collapsed the single scalar std to 0.3 — simultaneously
  too small for floor recovery and too large for standing stability.

Termination: height-gated fell_over (replaces the deleted termination)
  bad_orientation_while_elevated fires when BOTH:
    (a) body tilt > 75°  AND  (b) base height > 0.50 m.

  The height gate (0.50 m) is the critical fix:
    • Fallen starts (h ≈ 0.10–0.15 m):  gate closed → never fires → robot has
      the full 35 s episode to discover the get-up motion.
    • Upright/bent starts (h 0.56–0.80 m): gate open → if the robot tips over
      the episode terminates immediately (same urgency as v3).
    • Mid-recovery (h > 0.50 m): a partial stand that falls sideways is also
      terminated — the robot is penalised for unstable intermediate positions.

  This restores the v3 "don't fall while upright" signal that was lost when
  the original fell_over was removed, without breaking fallen initial states.

Reward changes
  orientation_recovery (+3.0, std=1.0)
    Primary get-up signal. Gaussian on (proj_gz + 1.0)² — distinguishes
    upright (0), flat (1), and inverted (4) unlike body_orientation_l2.

  height_recovery (+2.0, target=0.78 m, std=0.65 m)
    Secondary get-up signal. Gradient nonzero from 0.25 m fallen to 0.80 m.

  pose_convergence_gated (replaces pose_convergence, +1.5, std=0.5)
    Gated by (-proj_gz).clamp(0,1) so the pose reward is suppressed when the
    robot is flat — prevents "stay flat in default joints" local optimum.

  upward_base_velocity (+3.0, height_gate=0.60 m, max_vel=2.0)
    The key missing ingredient: immediate positive feedback for ANY upward
    motion of the base.  Fires only below 0.60 m (flat → mid-recovery phase).
    Without this, flat-and-still gives -0.09/step whether the robot tries or
    not — no gradient incentive for exploratory large actions.  With this,
    flat+pushing-up gives +0.9 to +3.0/step, creating a positive feedback
    loop: push off ground → upward velocity → reward → discover get-up motion.

  body_orientation_l2 weight increased to -3.0 (was -2.0 in v3).

  is_terminated KEPT from v3 at weight=-200.
    Since bad_orientation_while_elevated can now fire (for upright episodes),
    the termination penalty is meaningful and must remain.

  push_robot kept from v3 (interval 8–10 s).

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

  # ── Replace v3 bent-pose reset with unified fallen + bent reset ───────────
  del cfg.events["reset_robot"]

  cfg.events["reset_robot"] = EventTermCfg(
    func=mdp.reset_to_fallen_or_bent_pose,
    mode="reset",
    params={
      # Unified list: 2 fallen (supine/prone) + 4 bent = 6 templates (33 / 67 %).
      "all_pose_configs": ALL_POSE_CONFIGS,
      # Scatter position (same as v3).
      "xy_pos_range": 0.5,       # ±0.5 m cell scatter
      "yaw_range": math.pi,      # full 360° initial yaw
      # Joint noise for fallen templates (all joints treated equally).
      "fallen_joint_perturbation": 0.3,  # ±0.3 rad — moderate variety
      # Joint noise for bent templates (same as v3).
      "leg_perturbation": 0.10,          # ±0.10 rad on knee/hip/ankle
      "other_perturbation": 0.35,        # ±0.35 rad on arms/waist
      # Initial velocities (same as v3).
      "joint_vel_range": 0.15,   # ±0.15 rad/s
      "lin_vel_range": 0.20,     # ±0.20 m/s (x, y)
      "ang_vel_range": 0.30,     # ±0.30 rad/s
      # SceneEntityCfg objects for leg-joint targeting in bent templates.
      "knee_cfg":      SceneEntityCfg("robot", joint_names=(".*_knee_joint",)),
      "hip_pitch_cfg": SceneEntityCfg("robot", joint_names=(".*_hip_pitch_joint",)),
      "ankle_cfg":     SceneEntityCfg("robot", joint_names=(".*_ankle_pitch_joint",)),
      "asset_cfg":     SceneEntityCfg("robot"),
    },
  )

  # ── Replace fell_over with height-gated version ───────────────────────────
  # The vanilla fell_over (angle > 75°) fires immediately for fallen initial
  # states (starting at 90°), which is why it was deleted in the first attempt.
  # The fix: add a height gate — only fire when base_height > 0.50 m.
  #
  # Height gate ensures:
  #   fallen starts   (h ≈ 0.10–0.15 m):  gate closed → episode runs full 35 s
  #   upright starts  (h 0.56–0.80 m):    gate open   → termination fires → is_terminated(-200)
  #   mid-recovery    (h > 0.50 m):        gate open   → termination fires if unstable
  #
  # This restores the critical "falling is catastrophic" gradient for upright
  # episodes, which was accidentally lost in the original recovery_v1 design.
  cfg.terminations["fell_over"] = TerminationTermCfg(
    func=mdp.bad_orientation_while_elevated,
    params={
      "limit_angle": math.radians(75.0),
      "height_threshold": 0.50,   # m — all fallen poses settle at h < 0.25 m
    },
  )

  # ── Keep is_terminated reward ─────────────────────────────────────────────
  # bad_orientation_while_elevated can now fire for upright episodes, so the
  # is_terminated(-200) penalty is meaningful and must remain from v3.
  # (No change needed — just don't delete it.)

  # ── Replace pose_convergence with gated version ───────────────────────────
  # The ungated version would reward default joint angles even when the robot
  # is lying flat — creating a "stay flat" local optimum.
  del cfg.rewards["pose_convergence"]

  cfg.rewards["pose_convergence_gated"] = RewardTermCfg(
    func=mdp.pose_convergence_gated,
    weight=1.5,
    params={
      # std=0.5: meaningful gradient from moderate pose errors (~0.5 rad MSE),
      # tighter than v3's 0.4 since the gate already handles the flat case.
      "std": 0.5,
      "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
    },
  )

  # ── Add orientation_recovery reward ──────────────────────────────────────
  # Primary signal for getting off the ground.  Must dominate the reward
  # landscape when the robot is fallen.
  cfg.rewards["orientation_recovery"] = RewardTermCfg(
    func=mdp.orientation_recovery,
    weight=3.0,
    params={
      # std=1.0: at flat (proj_gz=0) reward ≈ 0.37 — clear gradient.
      # At 45° tilt reward ≈ 0.92.  At upright reward = 1.0.
      "std": 1.0,
      "asset_cfg": SceneEntityCfg("robot"),
    },
  )

  # ── Add height_recovery reward ────────────────────────────────────────────
  # Secondary signal for rising off the ground.  Complements orientation_
  # recovery to guide the robot through the "push up" phase.
  cfg.rewards["height_recovery"] = RewardTermCfg(
    func=mdp.height_recovery,
    weight=2.0,
    params={
      # target_height: G1 standing pelvis height minus a small margin.
      "target_height": 0.78,
      # std=0.65: gradient is nonzero from 0.25 m (fallen) through 0.78 m.
      "std": 0.65,
      "asset_cfg": SceneEntityCfg("robot"),
    },
  )

  # ── Add upward_base_velocity reward ─────────────────────────────────────
  # The key missing ingredient for floor recovery.  Flat-and-still gives
  # -0.09/step regardless of action; this reward makes flat+pushing-up give
  # +0.81 to +3.0/step — a direct positive feedback loop for get-up motions.
  # Height gate of 0.60 m covers the entire flat-to-mid-recovery phase and
  # also benefits deep-squat episodes (base_z ≈ 0.56 m).
  cfg.rewards["upward_base_velocity"] = RewardTermCfg(
    func=mdp.upward_base_velocity,
    weight=3.0,
    params={
      "height_gate": 0.60,   # m — gate closes above squat-to-standing transition
      "max_vel": 2.0,        # m/s clip — prevents reward hacking from wild jumps
      "asset_cfg": SceneEntityCfg("robot"),
    },
  )

  # ── Increase body_orientation_l2 penalty ─────────────────────────────────
  # The robot spends more episode time in tilted orientations than in v3.
  # A stronger penalty keeps the upright incentive dominant.
  cfg.rewards["body_orientation_l2"].weight = -3.0

  # ── Longer episode ────────────────────────────────────────────────────────
  # Getting up from the ground takes more time than rising from a deep squat.
  cfg.episode_length_s = 35.0

  return cfg
