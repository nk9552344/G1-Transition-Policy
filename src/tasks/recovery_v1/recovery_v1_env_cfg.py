"""Recovery-v1: full floor-recovery task configuration.

Extends transition-v3 so the robot learns to stand up from a completely
fallen position (supine, prone, or side-lying) in addition to all the
bent-upright starting configurations introduced in v3.

Key additions over v3
─────────────────────
Initial states
  Eight templates sampled uniformly each episode (4 fallen + 4 bent):
    Fallen:  supine, prone, side-left, side-right  (base_z ≈ 0.25 m)
    Bent:    home, knees_bent, squat, deep_squat    (FK-verified heights)

  The "fell_over" termination is removed entirely: the robot starts at 90°
  tilt (fallen), which would immediately trigger the v3 limit of 75°.  Only
  the timeout termination remains.

Reward changes
  orientation_recovery (+3.0, std=1.0)
    Primary get-up signal.  Gaussian reward for the torso being upright.
    Unlike body_orientation_l2 (which is symmetric for upright and upside-
    down), this term uses (proj_gz + 1.0)² so the gradient is meaningful
    from every starting orientation.

  height_recovery (+2.0, target=0.78 m, std=0.65 m)
    Secondary get-up signal.  Rewards the pelvis rising toward the standing
    height.  Wide std provides gradient from the 0.25 m fallen starting
    height to the 0.80 m standing height.

  pose_convergence_gated (replaces pose_convergence, +1.5, std=0.5)
    Joint-pose convergence, but gated by how upright the robot is.  When
    the robot is flat the gate ≈ 0, preventing the policy from learning
    "stay flat in default joints" as a local optimum.  Gate = -proj_gz,
    clamped to [0, 1].

  body_orientation_l2 weight increased to -3.0 (was -2.0 in v3).
    The robot now spends more episode time in tilted orientations; the
    stronger penalty ensures a robust upright signal throughout.

  is_terminated removed.
    The only remaining termination is timeout; is_terminated never fires
    so including it would add noise to the reward without benefit.

  push_robot kept from v3 (interval 8–10 s).
    If the robot stands up within ~8 s, the push tests recovery from a
    disturbance while standing — exactly the scenario recovery_v1 should
    handle.  If the robot is still fallen at 8 s, the push adds realistic
    momentum noise.

Episode length
  35 s (extended from v3's 25 s): getting up from flat ground takes more
  time than rising from a deep squat.
"""

import math

from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

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
      # Unified list: 4 fallen templates + 4 bent templates.
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

  # ── Remove fell_over termination ──────────────────────────────────────────
  # The robot starts at 90° tilt (fallen), exceeding the v3 75° limit.
  # Without this removal the episode would terminate on the very first step.
  del cfg.terminations["fell_over"]

  # ── Remove is_terminated reward ───────────────────────────────────────────
  # With only the timeout termination remaining, is_terminated never fires.
  del cfg.rewards["is_terminated"]

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

  # ── Increase body_orientation_l2 penalty ─────────────────────────────────
  # The robot spends more episode time in tilted orientations than in v3.
  # A stronger penalty keeps the upright incentive dominant.
  cfg.rewards["body_orientation_l2"].weight = -3.0

  # ── Longer episode ────────────────────────────────────────────────────────
  # Getting up from the ground takes more time than rising from a deep squat.
  cfg.episode_length_s = 35.0

  return cfg
