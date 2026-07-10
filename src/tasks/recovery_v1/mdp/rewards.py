"""Reward functions for recovery-v1: full floor-recovery training.

Five reward functions added on top of the v3 reward set:

  orientation_recovery
    Primary get-up signal. Rewards the torso having its gravity projection
    close to [0, 0, -1] (world up mapped to body -Z = upright). Uses
    (proj_gz + 1.0)^2 as the distance metric so the reward correctly
    distinguishes upright (0), flat (1), and inverted (4) unlike body_orientation_l2.

  height_recovery
    Secondary get-up signal. Rewards the robot base (pelvis) rising toward
    the standing pelvis height. The Gaussian std is wide enough to give a
    nonzero gradient from the 0.25 m fallen starting height all the way up
    to the ~0.80 m standing height.

  torso_upward_velocity
    Key anti-local-optimum signal. Rewards upward velocity of the torso
    (chest) body link, NOT the pelvis root. A leg-bridge maneuver lifts the
    pelvis while keeping the chest on the floor -- yielding zero reward here.
    Only arm-assisted recovery (push-ups, rolling) that raises the chest
    generates a positive signal, breaking the leg-only local optimum.

  orientation_rate
    Immediate gradient toward upright. Rewards the angular velocity component
    that decreases proj_gz (rotating from flat toward upright). Provides
    step-by-step feedback for exploratory roll/flip motions before the static
    orientation_recovery signal can respond.

  pose_convergence_gated
    Joint-position convergence reward gated by how upright the robot is.
    When flat (proj_gz ~= 0) the gate is ~0 so the robot is not rewarded
    for holding default joint angles while on the ground. When upright
    (proj_gz ~= -1) the gate is 1 and the full pose_convergence signal
    applies. This prevents the policy from learning to "stay flat in
    default pose" as a local optimum.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def orientation_recovery(
  env: ManagerBasedRlEnv,
  std: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward the torso being upright (primary floor-recovery signal).

  Uses the Z component of the projected gravity vector to measure tilt:
    proj_gz = -1.0  ->  robot is upright     (reward = 1.0)
    proj_gz =  0.0  ->  robot is lying flat  (reward = exp(-1 / std^2))
    proj_gz = +1.0  ->  robot is upside-down (reward = exp(-4 / std^2))

  Unlike body_orientation_l2 (which gives identical values for upright and
  upside-down), this reward correctly distinguishes all orientations and
  provides a gradient from every ground-lying starting pose.

  Recommended std=1.0: gives reward ~= 0.37 when flat and ~= 0.018 when
  fully inverted, with a clear gradient throughout.
  """
  asset = env.scene[asset_cfg.name]
  proj_gz = asset.data.projected_gravity_b[:, 2]  # [B]: -1 upright, 0 flat, +1 inverted
  dist_sq = torch.square(proj_gz + 1.0)           # [B]: 0 upright, 1 flat, 4 inverted
  return torch.exp(-dist_sq / std**2)


def height_recovery(
  env: ManagerBasedRlEnv,
  target_height: float,
  std: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward the robot base (pelvis) rising toward the standing height.

  Computes exp(-(base_z - target_height)^2 / std^2) relative to the env
  origin terrain height.

  Recommended target_height=0.78 (G1 standing pelvis ~= 0.80 m, slight
  margin), std=0.65 so the gradient is nonzero from the 0.25 m fallen
  starting height all the way to target.

  At fallen (base_z ~= 0.25 m): reward ~= 0.45.
  At mid-rise (base_z ~= 0.50 m): reward ~= 0.84.
  At target   (base_z ~= 0.78 m): reward = 1.0.
  """
  asset = env.scene[asset_cfg.name]
  base_height = asset.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]  # [B]
  dist_sq = torch.square(base_height - target_height)
  return torch.exp(-dist_sq / std**2)


def torso_upward_velocity(
  env: ManagerBasedRlEnv,
  height_gate: float,
  max_vel: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward upward velocity of the torso (chest) body when below standing height.

  Unlike upward_base_velocity which tracked the pelvis root link, this tracks
  the torso_link (chest body). This distinction is critical:

    Leg-bridge maneuver (local optimum):
      Pelvis rises to 0.35-0.45 m -> would trigger upward_base_velocity
      Torso (chest) stays on ground at 0.10-0.20 m -> zero reward here

    Arm push-up / rolling recovery (correct behavior):
      Torso rises -> positive reward -> policy learns arm-coordinated get-up

  The height gate should match the termination threshold (0.65 m) so the
  reward covers the entire flat-to-mid-recovery phase without overlapping
  with the standing-stability regime.

  asset_cfg must have body_names set to the torso body (e.g. "torso_link")
  by the robot-specific config (same pattern as body_orientation_l2).

  Args:
    height_gate: Gate closes above this torso height (m). Recommended 0.90 m
      for G1: covers flat (0.15 m) through mid-squat (0.75 m) without firing
      when the robot is fully standing (torso ~= 1.0 m).
    max_vel: Clamp upward torso velocity at this value (m/s). Recommended 1.5
      to prevent wild swings while still rewarding aggressive recovery.
    asset_cfg: Resolved SceneEntityCfg for the torso body (body_names set
      per robot, e.g. "torso_link" for G1).

  Returns:
    Reward tensor [B], range [0, max_vel].
  """
  asset = env.scene[asset_cfg.name]
  # body_link_lin_vel_w: (B, num_bodies, 3) world-frame linear velocities.
  # asset_cfg.body_ids selects the torso body; squeeze removes the body dim.
  lin_vel_z = asset.data.body_link_lin_vel_w[:, asset_cfg.body_ids, 2].squeeze(1)  # (B,)
  pos_z = (
    asset.data.body_link_pos_w[:, asset_cfg.body_ids, 2].squeeze(1)
    - env.scene.env_origins[:, 2]
  )  # (B,)
  gate = (pos_z < height_gate).float()
  return lin_vel_z.clamp(0.0, max_vel) * gate


def orientation_rate(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward angular velocity in the direction that rotates the robot toward upright.

  Computes d(proj_gz)/dt analytically:

    d(g_b)/dt = -(omega_b x g_b)
    d(g_b_z)/dt = -(omega_b_x * g_b_y - omega_b_y * g_b_x)

  We want proj_gz to DECREASE (from 0 flat to -1 upright), so we reward when
  d(proj_gz)/dt is negative:

    reward = max(0,  omega_b_x * g_b_y - omega_b_y * g_b_x)

  This gives immediate per-step gradient toward the correct rotational motion,
  complementing the static orientation_recovery signal. Suppressed once the
  robot is nearly upright (proj_gz < -0.9) to avoid noise during balance.

  Physical verification (supine, g_b = [-1, 0, 0]):
    reward = max(0, omega_b_x * 0 - omega_b_y * (-1)) = max(0, omega_b_y)
    -> positive omega_b_y (pitch forward in body frame) correctly rotates
       the robot from supine toward upright.

  Returns:
    Reward tensor [B], range [0, inf) -- typical range [0, 2] for 1 rad/s omegas.
  """
  asset = env.scene[asset_cfg.name]
  g_b = asset.data.projected_gravity_b    # (B, 3): gravity in body frame
  ang_vel_b = asset.data.root_link_ang_vel_b  # (B, 3): angular velocity in body frame

  # Positive when omega rotates proj_gz toward -1 (upright).
  improvement = (ang_vel_b[:, 0] * g_b[:, 1] - ang_vel_b[:, 1] * g_b[:, 0]).clamp(0.0)

  # Suppress when already upright -- the gradient is no longer needed.
  not_upright = (g_b[:, 2] > -0.9).float()

  return improvement * not_upright


def arm_reach_down(
  env: ManagerBasedRlEnv,
  height_gate: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward the hands being close to ground level when the robot is flat.

  Dense reward that pulls hands toward the floor, providing a gradient BEFORE
  actual ground contact is made. This is the first step of arm-based recovery:

    Robot is flat -> hands drift near body height (~0.15-0.25 m)
    arm_reach_down pulls them toward z=0 (ground)
    -> hands approach floor -> arm_ground_contact fires
    -> hands on floor, arm extends -> elbow_upward_velocity fires
    -> push-up motion -> chest rises -> torso_upward_velocity fires

  Gaussian on (hand_z - terrain_z): peak at z=0, std=0.15 m gives clear gradient
  from 0 to ~0.4 m. Only active when flat (proj_gz > -0.7, tilt > ~45 deg)
  and when hand is below height_gate (not when standing upright).

  asset_cfg must have body_names set to the hand bodies (e.g., left_wrist_yaw_link
  and right_wrist_yaw_link for G1). Set per-robot in the robot-specific config.

  Args:
    height_gate: Suppress reward above this hand height (m). Recommended 0.60 m
      so the reward only fires during the flat-to-mid-recovery phase.
    asset_cfg: Resolved SceneEntityCfg for the hand bodies (body_names set
      per robot, two bodies -- left and right).
  """
  asset = env.scene[asset_cfg.name]
  origins_z = env.scene.env_origins[:, 2].unsqueeze(1)         # (B, 1)
  hand_pos_z = (
    asset.data.body_link_pos_w[:, asset_cfg.body_ids, 2] - origins_z
  )  # (B, 2)

  # Dense proximity reward: peak when hand is at ground level, decays with height.
  hand_ground_proximity = torch.exp(-torch.square(hand_pos_z) / 0.15**2)  # (B, 2)

  # Height gate: suppress when hand is already near or above standing height.
  height_gate_mask = (hand_pos_z < height_gate).float()  # (B, 2)

  # Orientation gate: only reward when flat (not yet upright).
  proj_gz = asset.data.projected_gravity_b[:, 2]
  flat_gate = (proj_gz > -0.7).float().unsqueeze(1)  # (B, 1)

  return (hand_ground_proximity * height_gate_mask * flat_gate).mean(dim=1)  # (B,)


def arm_ground_contact(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward any arm contact with the ground when the robot is flat.

  Confirms that the arm has reached and is touching the floor -- the key
  enabling condition for arm-based push-up recovery. Works in conjunction
  with arm_reach_down (which pulls the hands down) and elbow_upward_velocity
  (which rewards the push motion once contact is established).

  Sparse signal: returns 1.0 when any tracked arm body (elbow subtree =
  elbow + wrist + hand geoms) touches the terrain, 0.0 otherwise.

  Gated by orientation so the reward is off when the robot is upright (we
  do not want to reward pushing hands into the ground when standing).

  The contact sensor must cover the forearm subtree (left_elbow_link and
  right_elbow_link subtrees for G1) and is registered in the robot-specific
  config.

  Args:
    sensor_name: Name of the arm ground contact sensor in the scene.
    asset_cfg: Resolved SceneEntityCfg for the robot (root, for proj_gz gate).
  """
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.found is not None
  # found: (B, 2) -- left arm and right arm contact with terrain.
  any_arm_contact = (sensor.data.found > 0).any(dim=1).float()  # (B,)

  # Gate: only reward arm ground contact when the robot is sufficiently flat.
  asset = env.scene[asset_cfg.name]
  proj_gz = asset.data.projected_gravity_b[:, 2]
  flat_gate = (proj_gz > -0.7).float()  # 1 when flat/tilted, 0 when upright

  return any_arm_contact * flat_gate


def elbow_push_from_ground(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  height_gate: float,
  max_vel: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward elbow going up ONLY when that arm is simultaneously on the ground.

  This is the key coupling that produces true push-up behavior:

    WITHOUT coupling: robot can earn elbow_upward_velocity by just swinging
    arms freely in the air, learning to flail rather than push.

    WITH coupling: elbow upward reward only fires when the arm is actually
    pressing against the terrain. This forces the exact sequence:
      1. arm_reach_down  -> hand reaches to floor level
      2. arm_ground_contact -> arm presses floor (sensor fires)
      3. elbow_push_from_ground -> elbow extends WHILE arm on floor = push-up!
      4. torso_upward_velocity -> chest rises (physics delivers the result)

  Per-arm coupling: left arm contact gates left elbow velocity, right arm
  contact gates right elbow velocity. The robot can push with one or both arms.

  sensor_name must point to the arm ground contact sensor (shape (B, 2) found,
  left arm first). asset_cfg.body_ids must resolve to the two elbow_link bodies
  in the same left/right order.

  Args:
    sensor_name: Name of the arm ground contact sensor (same sensor used by
      arm_ground_contact). Provides per-arm (B, 2) contact signal.
    height_gate: Gate closes above this elbow height (m). Recommended 0.70 m.
    max_vel: Clamp upward elbow velocity (m/s). Recommended 1.5.
    asset_cfg: Resolved SceneEntityCfg for the two elbow bodies (left, right).
  """
  asset = env.scene[asset_cfg.name]
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.found is not None

  # Per-arm contact signal: (B, 2) -- 1 if that arm is touching terrain, else 0.
  arm_contact = (sensor.data.found > 0).float()  # (B, 2)

  # Upward velocity of each elbow: (B, 2).
  elbow_vel_z = asset.data.body_link_lin_vel_w[:, asset_cfg.body_ids, 2]  # (B, 2)
  origins_z = env.scene.env_origins[:, 2].unsqueeze(1)                    # (B, 1)
  elbow_pos_z = (
    asset.data.body_link_pos_w[:, asset_cfg.body_ids, 2] - origins_z
  )  # (B, 2)

  # Height gate: suppress once elbow is above mid-recovery height.
  height_gate_mask = (elbow_pos_z < height_gate).float()  # (B, 2)

  # Orientation gate: only reward when robot is sufficiently flat.
  proj_gz = asset.data.projected_gravity_b[:, 2]
  flat_gate = (proj_gz > -0.7).float().unsqueeze(1)  # (B, 1)

  # Critical: elbow velocity reward ONLY fires when that arm is on the ground.
  return (
    elbow_vel_z.clamp(0.0, max_vel) * arm_contact * height_gate_mask * flat_gate
  ).mean(dim=1)


def pose_convergence_gated(
  env: ManagerBasedRlEnv,
  std: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Joint-pose convergence reward, smoothly gated by upright orientation.

  Gate: upright_weight = clamp(-proj_gz, 0, 1)
    When flat        (proj_gz ~=  0): gate ~= 0  -> no pose reward
    When upright     (proj_gz ~= -1): gate = 1  -> full pose reward
    When partially tilted (45 deg):   gate ~= 0.7 -> partial pose reward

  This prevents the policy exploiting "stay flat in default joint config"
  as a pose-convergence reward shortcut. The pose reward only activates
  as the robot rises toward vertical.

  Same Gaussian kernel as pose_convergence: exp(-MSE(q, q_default) / std^2).
  """
  asset = env.scene[asset_cfg.name]

  q = asset.data.joint_pos[:, asset_cfg.joint_ids]
  q_default = asset.data.default_joint_pos[:, asset_cfg.joint_ids]
  mse = torch.mean(torch.square(q - q_default), dim=1)  # [B]

  proj_gz = asset.data.projected_gravity_b[:, 2]  # [B]
  upright_weight = (-proj_gz).clamp(0.0, 1.0)     # [B]: 0 when flat, 1 when upright

  return torch.exp(-mse / std**2) * upright_weight
