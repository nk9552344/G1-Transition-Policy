"""Recovery-v2: full floor-recovery from any fallen orientation.

Extends recovery-v1 so the robot learns to stand up from all four common
ground-level orientations — supine, prone, side_left, side_right — rather
than only supine and prone.

Key additions over recovery-v1
───────────────────────────────

Initial states
  Eight templates sampled uniformly each episode (4 fallen, 50 % + 4 bent, 50 %):
    Fallen:  supine, prone, side_left, side_right     (base_z ≈ 0.25 m)
    Bent:    home, knees_bent, squat, deep_squat       (FK-verified heights)

  The 50 / 50 split is now stable because orientation_velocity_reward (see below)
  provides direct gradient for the lateral-rolling motion that side-lying recovery
  requires.  Recovery-v1 kept side poses out because the only relevant signal
  (upward_base_velocity) fires on translational upward motion, not lateral rolling.

New reward: orientation_velocity_reward (+2.0)
  Analytic gradient for angular velocity that reduces orientation error.
  Reward = max(0, (g_b - target) · (ω_b × g_b)), scale=1.0, clip=1.0.

  Why this is needed for side-lying
  ──────────────────────────────────
  When side-lying: base_height ≈ 0.25 m (same as supine/prone) → upward_base_
  velocity fires equally.  BUT the recovery motion for side poses is LATERAL
  ROLLING before any upward translation occurs:

    side-lying → roll → prone or supine → push up → stand

  During the rolling phase, base_z barely changes → upward_base_velocity = 0.
  The orientation_recovery reward increases, but slowly (Gaussian, not linear).
  orientation_velocity_reward gives an IMMEDIATE dense reward for any angular
  velocity that is geometrically aligned with reducing the tilt, creating the
  same positive-feedback loop that upward_base_velocity created for flat-ground
  push-up in v1.

Modified rewards
  orientation_recovery weight: +3.0 → +4.0
    Side poses increase the average per-step orientation error.  A stronger
    orientation signal keeps recovery the dominant objective when 50 % of
    starts are fallen.

  upward_base_velocity height_gate: 0.60 m → 0.55 m
    Side-lying has the same base_z as supine/prone (≈ 0.25 m).  No change
    needed for the gate; however slightly tightening to 0.55 m provides a
    sharper cut-off at the transition from mid-recovery to near-standing, which
    avoids giving upward velocity reward during the final balance phase.

Termination
  bad_orientation_while_elevated unchanged (height_threshold=0.50 m).
  Side-lying also settles at h < 0.25 m — gate is closed at episode start.

Episode length
  45 s (extended from 35 s): lateral rolling adds an extra phase before push-up,
  requiring roughly 5–10 extra seconds compared to supine/prone recovery.
"""

import math

from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

import src.tasks.recovery_v2.mdp as mdp
from src.tasks.recovery_v2.mdp.events import ALL_POSE_CONFIGS
from src.tasks.recovery_v1.recovery_v1_env_cfg import make_recovery_v1_env_cfg


def make_recovery_v2_env_cfg():
    """Create recovery-v2 task configuration (builds on recovery-v1)."""
    cfg = make_recovery_v1_env_cfg()

    # ── Replace v1 reset (2 fallen + 4 bent) with all-4-fallen version ───────
    # ALL_POSE_CONFIGS (v2) contains all 4 fallen orientations + 4 bent poses.
    del cfg.events["reset_robot"]

    cfg.events["reset_robot"] = EventTermCfg(
        func=mdp.reset_to_any_fallen_pose,
        mode="reset",
        params={
            # Unified list: 4 fallen + 4 bent = 8 templates (50 / 50 %).
            "all_pose_configs": ALL_POSE_CONFIGS,
            # Position / orientation scatter (same as v1).
            "xy_pos_range": 0.5,
            "yaw_range": math.pi,
            # Joint noise for fallen templates.
            # Slightly higher than v1 (0.3 → 0.4) to increase variety for side
            # poses, which have more arm/leg positions relative to ground.
            "fallen_joint_perturbation": 0.4,
            # Joint noise for bent templates (same as v1).
            "leg_perturbation": 0.10,
            "other_perturbation": 0.35,
            # Initial velocities (same as v1).
            "joint_vel_range": 0.15,
            "lin_vel_range": 0.20,
            "ang_vel_range": 0.30,
            # SceneEntityCfg objects for leg-joint targeting in bent templates.
            "knee_cfg":      SceneEntityCfg("robot", joint_names=(".*_knee_joint",)),
            "hip_pitch_cfg": SceneEntityCfg("robot", joint_names=(".*_hip_pitch_joint",)),
            "ankle_cfg":     SceneEntityCfg("robot", joint_names=(".*_ankle_pitch_joint",)),
            "asset_cfg":     SceneEntityCfg("robot"),
        },
    )

    # ── Add orientation_velocity_reward (new in v2) ───────────────────────────
    # This reward provides an analytic gradient for lateral rolling — the key
    # motion required for side-lying recovery that upward_base_velocity misses.
    cfg.rewards["orientation_velocity"] = RewardTermCfg(
        func=mdp.orientation_velocity_reward,
        weight=2.0,
        params={
            # scale=1.0: the raw improvement value (rad/s magnitude) is already
            # in a useful range; no additional scaling needed.
            "scale": 1.0,
            # clip_val=1.0: prevents reward hacking from wild angular velocities.
            # At 1.0 rad/s of effective orienting angular velocity the reward
            # saturates — fast enough to cover the get-up motion.
            "clip_val": 1.0,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # ── Strengthen orientation_recovery for broader fallen distribution ───────
    # 50 % of starts are now fallen (vs 33 % in v1).  A stronger orientation
    # signal keeps recovery the dominant objective.
    cfg.rewards["orientation_recovery"].weight = 4.0

    # ── Tighten upward_base_velocity gate ────────────────────────────────────
    # Reduce height_gate from 0.60 to 0.55 m for a sharper transition between
    # mid-recovery (upward momentum needed) and near-standing (balance needed).
    cfg.rewards["upward_base_velocity"].params["height_gate"] = 0.55

    # ── Longer episode ────────────────────────────────────────────────────────
    # Side-lying recovery adds a rolling phase before the push-up phase,
    # requiring roughly 5–10 extra seconds over supine/prone recovery.
    cfg.episode_length_s = 45.0

    return cfg
