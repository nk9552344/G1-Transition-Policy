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


def torso_height_reward(
  env: ManagerBasedRlEnv,
  target_height: float,
  std: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward the torso (chest) body being elevated toward the standing height.

  Position-based (not velocity-based). This distinction is critical:

    torso_upward_velocity (REMOVED - velocity-based, gameable):
      Rewards max(0, v_z) per step. Oscillation (torso up-then-down) earns
      reward on the upward half-cycle and 0 on the downward half. Net per
      oscillation = 50% of the amplitude reward. The robot discovers it can
      farm this by wiggling waist joints -- the "torso wiggling" behavior.

    torso_height_reward (this function - position-based, not gameable):
      Rewards exp(-(torso_z - target)^2 / std^2) -- the CURRENT height.
      Oscillation between z_low and z_high gives average reward equal to
      the reward at the mean position. No benefit over holding still.
      The robot MUST find a way to SUSTAIN a higher torso position, which
      requires arm support from the ground.

  Gradient chain:
    Flat,  torso=0.15 m: exp(-(0.15-0.90)^2 / 0.50^2) = 0.105 * 3.0 = +0.32
    Push-up torso=0.50 m: exp(-(0.50-0.90)^2 / 0.50^2) = 0.527 * 3.0 = +1.58
    Standing torso=0.90 m: 1.0 * 3.0 = +3.0

  Sustained elevation (push-up position) gives 5x more reward than lying flat.
  Only arm support can sustain the push-up height, closing the oscillation trap.

  asset_cfg must have body_names set to the torso body (e.g. "torso_link")
  by the robot-specific config (same pattern as body_orientation_l2).

  Args:
    target_height: Torso height (m) target. G1 torso_link when standing ~= 0.90 m.
    std: Gaussian width (m). std=0.50 gives nonzero gradient from 0.15 m (flat)
      to 0.90 m (standing).
    asset_cfg: Resolved SceneEntityCfg for the torso body (body_names set per robot).

  Returns:
    Reward tensor [B], range [0, 1].
  """
  asset = env.scene[asset_cfg.name]
  torso_z = (
    asset.data.body_link_pos_w[:, asset_cfg.body_ids, 2].squeeze(1)
    - env.scene.env_origins[:, 2]
  )  # (B,)
  dist_sq = torch.square(torso_z - target_height)
  return torch.exp(-dist_sq / std**2)


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


def head_height_reward(
  env: ManagerBasedRlEnv,
  target_height: float,
  std: float,
  head_offset: float = 0.43,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward the head being at standing height — stronger sit-to-stand gradient.

  The G1 head has no separate body link: it is a geom inside torso_link at
  local position (0, 0, 0.43 m). The head world Z is:

      head_z ≈ torso_z + head_offset × cos(tilt)
             = torso_z + head_offset × (-proj_gz)

  because R_{22} = cos(tilt) = -proj_gz (proj_gz = -1 when upright).

  This creates a much sharper sitting-vs-standing distinction than torso alone:
    Flat   (torso=0.15, proj_gz= 0.0): head_z = 0.15 + 0.00 = 0.15 m
    Sit-up (torso=0.60, proj_gz=-0.77): head_z = 0.60 + 0.33 = 0.93 m
    Stand  (torso=0.90, proj_gz=-1.0 ): head_z = 0.90 + 0.43 = 1.33 m

  The reward at target=1.30 m, std=0.60 m:
    Flat:    exp(-4.0) = 0.018   -> near-zero
    Sit-up:  exp(-0.44) = 0.644  -> moderate
    Stand:   exp(-0.00) = 1.000  -> maximum

  This creates strong gradient from sit-up -> stand without rewarding oscillation.

  asset_cfg must have body_names set to the torso body (same as body_orientation_l2).
  proj_gz is taken from root_link projected_gravity_b (pelvis frame), which is a
  good approximation of the torso orientation when waist joints are small.

  Args:
    target_height: Head height (m) target. G1 standing head ~= 1.30-1.35 m.
    std: Gaussian width (m). 0.60 m gives gradient from 0.15 m to 1.35 m.
    head_offset: Head center height above torso_link origin (m). G1 = 0.43 m.
    asset_cfg: Resolved SceneEntityCfg for the torso body (body_names set per robot).
  """
  asset = env.scene[asset_cfg.name]
  torso_z = (
    asset.data.body_link_pos_w[:, asset_cfg.body_ids, 2].squeeze(1)
    - env.scene.env_origins[:, 2]
  )  # (B,)
  proj_gz = asset.data.projected_gravity_b[:, 2]          # (B,): -1 upright, 0 flat
  head_z = torso_z + head_offset * (-proj_gz)             # (B,): head world Z approx
  dist_sq = torch.square(head_z - target_height)
  return torch.exp(-dist_sq / std**2)


def feet_proximity_reward(
  env: ManagerBasedRlEnv,
  height_gate: float,
  std: float = 0.30,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """When pelvis is elevated, reward feet being horizontally close to the pelvis.

  This guides the "tuck feet under body" phase of standing up. After the robot
  raises its upper body (push-up phase), it must bring its feet under the pelvis
  to be able to stand. This requires bending the knees deeply and pivoting the
  hips. Without an explicit reward, the robot stays in the seated-with-legs-
  -extended local optimum even though the upper body is raised.

  The reward measures horizontal (XY) distance from each foot to the pelvis.
  It is gated off when the pelvis is below height_gate (flat ground: feet
  distance to pelvis is meaningless when the robot is fully prone).

    Sitting, legs extended: foot XY ≈ 0.5-0.8 m forward of pelvis -> low reward
    Squatting / tucked:     foot XY ≈ 0.0-0.2 m of pelvis         -> high reward
    Standing:               foot XY ≈ 0.1 m of pelvis              -> high reward

  std controls the gradient width. With std=0.30 m the Gaussian is nearly zero
  when feet are 0.6+ m from the pelvis, so the robot receives no guidance for
  the initial portion of the knee-tuck. With std=0.45 m the gradient is
  meaningful from the start of the tuck motion:
    foot at 0.70 m: exp(-2.44) ≈ 0.087  (vs. exp(-5.44) ≈ 0.004 at std=0.30)
    foot at 0.30 m: exp(-0.44) ≈ 0.644
    foot at 0.10 m: exp(-0.05) ≈ 0.951

  asset_cfg must have body_names set to the ankle bodies (e.g. left_ankle_roll_link
  and right_ankle_roll_link for G1), providing the foot world positions.

  Args:
    height_gate: Minimum pelvis height (m) for the reward to activate.
      Recommended 0.35 m -- just above prone pelvis height (~0.25 m) so the
      reward only fires when the robot has raised itself off the floor.
    std: Gaussian width (m) for the foot-to-pelvis XY distance.
      Recommended 0.45 m -- provides gradient from the start of the knee-tuck.
    asset_cfg: Resolved SceneEntityCfg for the ankle/foot bodies (body_names
      set per robot, two bodies -- left and right).
  """
  asset = env.scene[asset_cfg.name]
  origins = env.scene.env_origins                                   # (B, 3)

  # Pelvis height gate.
  pelvis_z = asset.data.root_link_pos_w[:, 2] - origins[:, 2]      # (B,)
  gate = (pelvis_z > height_gate).float()                           # (B,)

  # Pelvis XY position (reference point).
  pelvis_xy = asset.data.root_link_pos_w[:, :2]                    # (B, 2)

  # Foot XY positions: (B, 2_feet, 2).
  foot_xy = asset.data.body_link_pos_w[:, asset_cfg.body_ids, :2]  # (B, 2, 2)

  # Horizontal distance from each foot to the pelvis.
  diff = foot_xy - pelvis_xy.unsqueeze(1)                          # (B, 2, 2)
  dist_sq_xy = (diff ** 2).sum(dim=-1)                             # (B, 2)

  # Gaussian reward peaking when foot is directly under pelvis.
  proximity = torch.exp(-dist_sq_xy / std**2)                      # (B, 2)

  return proximity.mean(dim=1) * gate                              # (B,)


def arm_reach_down(
  env: ManagerBasedRlEnv,
  height_gate: float,
  flat_gate_threshold: float = -0.7,
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

  # Orientation gate: only reward when flat or partially tilted (not when upright).
  # flat_gate_threshold=-0.85 extends arm support through the sit-to-stand phase
  # (robot is ~31° from upright) rather than cutting off at -0.7 (45°).
  proj_gz = asset.data.projected_gravity_b[:, 2]
  flat_gate = (proj_gz > flat_gate_threshold).float().unsqueeze(1)  # (B, 1)

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
  flat_gate_threshold: float = -0.7,
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

  # Orientation gate: active when robot is sufficiently flat/tilted.
  # flat_gate_threshold=-0.85 keeps arm support active through the sit-to-stand
  # transition phase (31° from upright), not just during flat recovery (45°).
  proj_gz = asset.data.projected_gravity_b[:, 2]
  flat_gate = (proj_gz > flat_gate_threshold).float().unsqueeze(1)  # (B, 1)

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


def shank_orientation_reward(
  env: ManagerBasedRlEnv,
  height_gate: float,
  std: float = 0.50,
  knee_asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ankle_asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward the shanks (knee-to-ankle) pointing straight down — the key anti-sitting signal.

  In the sit-up position with legs extended forward the shank runs at 30-50°
  from vertical (cosine ≈ 0.6–0.7). Standing or squatting requires the shank to
  be nearly vertical (cosine ≈ 1.0). This reward therefore creates a direct
  gradient from the sitting local optimum toward the knee-tuck-and-stand motion
  that is completely absent from the position-based height rewards.

  Physical check (std=0.50):
    Sitting, legs forward on floor: shank cosine ≈ 0.55–0.70 → exp(-0.81 to -0.36) ≈ 0.44–0.70
    Squat, feet under body:         ankle directly below knee → cosine = 1.0 → 1.0
    Standing:                        same geometry             → cosine = 1.0 → 1.0

  Gated off when pelvis is below height_gate so the reward does not interfere
  during the flat push-up phase (shanks on floor = horizontal, which is correct
  behaviour at that stage).

  knee_asset_cfg.body_ids  must resolve to 2 bodies: [left_knee, right_knee].
  ankle_asset_cfg.body_ids must resolve to 2 bodies: [left_ankle, right_ankle],
  in the same left/right order.

  Args:
    height_gate: Minimum pelvis height (m) for the reward to activate.
      Recommended 0.30 m — activates once the pelvis clears the floor but
      before the full sit-up phase, so the policy starts receiving the signal
      early in the recovery trajectory.
    std: Gaussian width on (shank_cosine − 1). std=0.50 gives:
      cosine=0 (horizontal): exp(-4.0) ≈ 0.02;  cosine=0.5: exp(-1.0) ≈ 0.37;
      cosine=0.7: exp(-0.36) ≈ 0.70;  cosine=0.9: exp(-0.08) ≈ 0.92;  cosine=1.0: 1.0.
      Wider than 0.30 so the sitting range (cosine 0.55–0.70) receives a clear
      gradient signal rather than near-zero values (std=0.30 gave ≈ 0.06 there).
    knee_asset_cfg:  SceneEntityCfg with 2 body_ids (left and right knee links).
    ankle_asset_cfg: SceneEntityCfg with 2 body_ids (left and right ankle roll links).
  """
  asset = env.scene[knee_asset_cfg.name]

  knee_pos  = asset.data.body_link_pos_w[:, knee_asset_cfg.body_ids,  :]  # (B, 2, 3)
  ankle_pos = asset.data.body_link_pos_w[:, ankle_asset_cfg.body_ids, :]  # (B, 2, 3)

  shank_vec    = ankle_pos - knee_pos                                   # (B, 2, 3): knee→ankle
  shank_len    = ((shank_vec ** 2).sum(dim=-1)).sqrt().clamp(min=1e-6)  # (B, 2)
  shank_unit_z = shank_vec[:, :, 2] / shank_len                        # (B, 2)

  # Vertical cosine: +1 when shank points straight down (ankle directly below knee).
  # 0 when horizontal, negative when inverted (impossible in normal poses).
  shank_cosine = -shank_unit_z                                         # (B, 2)

  # Gaussian: peaks at cosine=1 (vertical), near-zero at cosine=0 (horizontal).
  dist_sq = torch.square(shank_cosine - 1.0)
  reward  = torch.exp(-dist_sq / std**2).mean(dim=1)                   # (B,)

  # Height gate: suppress during the flat push-up phase.
  origins_z = env.scene.env_origins[:, 2]
  pelvis_z  = asset.data.root_link_pos_w[:, 2] - origins_z             # (B,)
  gate      = (pelvis_z > height_gate).float()                         # (B,)

  return reward * gate


def pushup_support_reward(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  target_height: float,
  std: float,
  height_gate: float,
  flat_gate_threshold: float = -0.70,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Position-based arm push-up reward: elbow elevated while arm contacts the ground.

  This is the position-based replacement for the velocity-based
  ``elbow_push_from_ground``.  The key difference:

    elbow_push_from_ground (REMOVED — velocity-based, farmable):
      Rewards max(0, elbow_vel_z) × arm_contact.  The robot earns reward on
      EVERY upward elbow bounce even if the elbow returns to floor level each
      cycle.  Bouncing the elbow at 2 Hz while lying flat earns reward on
      50 % of steps indefinitely → discovered early, dominates the policy.

    pushup_support_reward (this function — position-based, not farmable):
      Rewards exp(-(elbow_z − target_height)² / std²) × arm_contact.
      The reward is a function of CURRENT elbow height, not velocity:
        Bouncing, elbow at floor level (~0.10 m):  0.21 × contact
        Mid push-up, elbow at 0.25 m:              0.78 × contact
        Full push-up sustained, elbow at 0.35 m:   1.00 × contact
      The robot maximises this by HOLDING the push-up position, not
      by oscillating.  Bouncing earns 0.21; holding earns 1.0.

  How it guides the push-up sequence:
    1. arm_reach_down (kept)  → hand reaches floor; arm_contact becomes 1
    2. this reward fires immediately at moderate level (0.21)
    3. arm extends: elbow rises from 0.10 → 0.35 m → reward 1.0
    4. chest rises: torso_height_reward and orientation_recovery fire strongly
    5. policy learns to SUSTAIN the elevated elbow = real push-up

  asset_cfg.body_ids must resolve to the two elbow_link bodies (left, right)
  in the same left/right order as the sensor.

  Args:
    sensor_name: Arm ground contact sensor (same as former elbow_push_from_ground).
      Shape (B, 2) found — left arm / right arm.
    target_height: Target elbow z (m) = push-up elbow height above terrain.
      G1 in push-up: elbow at ~0.30–0.40 m. Recommended 0.35 m.
    std: Gaussian width (m). 0.20 m gives clear gradient from floor (0.10 m) to
      push-up height (0.35 m): exp(-(0.10-0.35)²/0.04) = 0.21 at floor.
    height_gate: Suppress above this pelvis height (m). 0.65 m = not yet standing.
    flat_gate_threshold: Orientation gate (proj_gz threshold). Active when
      proj_gz > threshold (robot is flat or partially upright). -0.70 ≈ 46° from
      vertical — keeps arm support active through the critical sit-to-stand phase.
    asset_cfg: SceneEntityCfg for the two elbow bodies (body_names set per robot).
  """
  asset = env.scene[asset_cfg.name]
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.found is not None

  origins_z = env.scene.env_origins[:, 2].unsqueeze(1)                   # (B, 1)
  elbow_z = (
    asset.data.body_link_pos_w[:, asset_cfg.body_ids, 2] - origins_z
  )  # (B, 2)

  # Position-based Gaussian: peaks when elbow is at target_height.
  elbow_reward = torch.exp(-torch.square(elbow_z - target_height) / std**2)  # (B, 2)

  # Gate 1: per-arm contact — reward only fires when arm presses on ground.
  arm_contact = (sensor.data.found > 0).float()  # (B, 2)

  # Gate 2: flat-phase gate — active when robot is sufficiently flat.
  proj_gz = asset.data.projected_gravity_b[:, 2]
  flat_gate = (proj_gz > flat_gate_threshold).float().unsqueeze(1)        # (B, 1)

  # Gate 3: pelvis height gate — suppress once robot is nearly standing.
  pelvis_z = asset.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]  # (B,)
  height_gate_mask = (pelvis_z < height_gate).float().unsqueeze(1)            # (B, 1)

  return (elbow_reward * arm_contact * flat_gate * height_gate_mask).mean(dim=1)  # (B,)


def height_gated_ang_vel_penalty(
  env: ManagerBasedRlEnv,
  gate_min_height: float,
  gate_max_height: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Angular velocity penalty gated to zero during the floor-recovery phase.

  Why the ungated penalty (body_angular_velocity_penalty) breaks floor recovery
  ─────────────────────────────────────────────────────────────────────────────
  Rolling from supine to prone / flipping from prone to push-up requires
  sustained XY body angular velocity of 1–2 rad/s.  With weight -0.03 and
  ang_vel² = 2.25 per step, the penalty is -0.068/step.  Over a 1-second roll
  (50 steps) this accumulates to -3.4.  The orientation_recovery improvement
  over the same roll is +9.0.  On paper the net is positive — BUT the GAE
  discount (λ=0.95, γ=0.99 → horizon ≈ 17 steps) means the future orientation
  gain is visible over only ~17 steps, not 50.  The IMMEDIATE angular penalty
  sees full weight; the distant orientation gain is heavily discounted.
  Net discounted advantage ≈ -1.16 (penalty) + 0.42 (orientation) = -0.74.
  The policy correctly learns NOT to roll because it looks unprofitable at the
  GAE horizon.

  This function removes the penalty below gate_min_height (floor phase) where
  rolling/flipping are necessary, and restores it smoothly above gate_max_height
  (elevated/standing phase) where large angular velocity indicates instability.

  Smooth transition:
    gate(h) = clamp((h - gate_min_height) / (gate_max_height - gate_min_height), 0, 1)
    penalty  = -weight × ang_vel_xy² × gate(h)

    h < gate_min_height:    gate = 0 → no penalty  (floor, rolling allowed)
    h between min/max:      gate ∈ (0, 1)          (mid-recovery, graduated)
    h > gate_max_height:    gate = 1 → full penalty (elevated, prevent falls)

  asset_cfg.body_ids must resolve to the torso body (same as body_angular_velocity_penalty).

  Args:
    gate_min_height: Pelvis height (m) below which the penalty is zero.
      Recommended 0.40 m — just above the floor push-up phase (~0.30 m).
    gate_max_height: Pelvis height (m) above which the full penalty applies.
      Recommended 0.65 m — start of the standing zone.
    asset_cfg: SceneEntityCfg with body_ids resolved to torso body (set per robot).
  """
  asset = env.scene[asset_cfg.name]

  pelvis_z = asset.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]  # (B,)
  gate = ((pelvis_z - gate_min_height) / (gate_max_height - gate_min_height)).clamp(0.0, 1.0)  # (B,)

  # XY angular velocity of the torso body (same as body_angular_velocity_penalty).
  ang_vel = asset.data.body_link_ang_vel_w[:, asset_cfg.body_ids, :].squeeze(1)  # (B, 3)
  ang_vel_xy_sq = torch.sum(torch.square(ang_vel[:, :2]), dim=1)                 # (B,)

  return ang_vel_xy_sq * gate  # (B,)  — multiply by -weight in the reward config


def base_height_obs(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Pelvis height above terrain — critical phase-detection signal for the actor.

  The actor cannot infer absolute height from projected_gravity + joint_pos
  alone: a robot in a push-up at 0.40 m and one nearly-standing at 0.70 m
  can have similar projected gravity and joint configurations, yet need
  completely different actions.  Height disambiguates the phase.

  Returns a single scalar per env (B, 1) normalised so:
    0.0 ≈ floor-level (fallen)      ~0.15 m
    1.0 ≈ standing height            ~0.80 m
  Actual range is not clamped; the obs normalisation in the actor handles
  values outside [0, 1].
  """
  asset = env.scene[asset_cfg.name]
  height = (
    asset.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
  ).unsqueeze(1)  # (B, 1)
  return height


def airborne_penalty(
  env: ManagerBasedRlEnv,
  min_height: float,
  foot_sensor_name: str,
  arm_sensor_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalise being elevated with no ground contact (prevents jump-hacking).

  The Phase 2 rewards (shank_orientation + head_above_feet + feet_proximity)
  sum to ~+9/step and fire simultaneously during a brief jump from squat
  position.  Over 5–10 airborne steps this earns +45–90, outweighing
  is_terminated=-50.  This penalty fires at -1.0 per step whenever the robot
  is elevated AND has no foot OR arm contact, directly making that jump
  unprofitable:

    airborne 5 steps × (-10.0) = -50
    Phase 2 reward 5 steps      = +45
    is_terminated               = -50
    Net = -55  →  NOT profitable

  Does NOT fire during:
    Push-up   (arm contact, pelvis ~0.30–0.35 m < min_height):  contact AND below gate
    Kneeling  (foot contact):                                    has_contact = 1
    Standing  (foot contact):                                    has_contact = 1

  DOES fire during:
    Jump from squat (pelvis > min_height, no foot or arm contact): = 1.0/step

  Args:
    min_height: Pelvis height (m) below which penalty is inactive.
      Recommended 0.40 m — push-up pelvis (~0.30–0.35 m) is below this.
    foot_sensor_name: Ground contact sensor for feet.
    arm_sensor_name:  Ground contact sensor for arms/elbows.
    asset_cfg: SceneEntityCfg for the robot.

  Returns:
    Tensor [B]: 1.0 when airborne+elevated, 0.0 otherwise.
    Multiply by negative weight (e.g. -10.0) in the reward config.
  """
  asset = env.scene[asset_cfg.name]
  foot_sensor: ContactSensor = env.scene[foot_sensor_name]
  arm_sensor:  ContactSensor = env.scene[arm_sensor_name]

  base_height = asset.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]   # (B,)
  elevated    = (base_height > min_height).float()                                  # (B,)

  any_foot = (foot_sensor.data.found > 0).any(dim=1).float()                       # (B,)
  any_arm  = (arm_sensor.data.found  > 0).any(dim=1).float()                       # (B,)

  # Arm contact exemption is only valid when the robot is NOT inverted.
  # Without this gate the "bridge trick" is possible: one arm drags on the floor
  # while the robot slowly arcs backward (proj_gz goes from -0.3 to +0.5),
  # earning full Phase rewards at low root velocity without triggering airborne_penalty.
  # Gate: proj_gz < 0.3 = robot is upright or flat (not backward-tilted past ~73°).
  #   Push-up (proj_gz ≈ -0.3): not_inverted=1 → arm contact exempts ✓
  #   Bridge  (proj_gz ≈ +0.5): not_inverted=0 → arm contact does NOT exempt ✓
  proj_gz     = asset.data.projected_gravity_b[:, 2]                               # (B,)
  not_inverted = (proj_gz < 0.3).float()                                           # (B,)

  has_contact = (any_foot + any_arm * not_inverted).clamp(0.0, 1.0)               # (B,)

  return elevated * (1.0 - has_contact)                                             # (B,)


def root_lin_vel_penalty(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalise explosive whole-body linear velocity (anti-jump, anti-launch).

  Normal recovery motions earn near-zero penalty:
    push-up rise (~0.3 m/s):  −0.3 × 0.09 = −0.027/step  (negligible)
    stand-up transition (~0.5 m/s): −0.3 × 0.25 = −0.075/step (negligible)

  An explosive jump or physics launch earns a large penalty:
    explosive jump push-off (~5 m/s): −0.3 × 25 = −7.5/step
    full jump trajectory  (~8 m/s):   −0.3 × 64 = −19.2/step

  This makes the GROUND-CONTACT PUSH-OFF PHASE of a jump unprofitable, which
  the airborne_penalty (fires only after feet leave floor) cannot address.
  Combined with airborne_penalty, a 10-step jump from squat incurs:
    push-off (5 steps, ~5 m/s):  −0.3 × 25 × 5  = −37.5
    airborne (5 steps, ~8 m/s):  −0.3 × 64 × 5  = −96.0
    airborne_penalty:             −10.0 × 5      = −50.0
    is_terminated:                                 −50.0
    Phase rewards:                +14/step × 10  = +140
    Net: +140 − 37.5 − 96 − 50 − 50 = −93.5  →  NOT profitable.

  Also catches rigid-body physics launches at reset (all joints move together,
  joint_velocity_overflow may not fire, but root velocity spikes to 10–30 m/s).

  Returns root linear velocity squared (B,). Multiply by negative weight in config.
  """
  asset = env.scene[asset_cfg.name]
  lin_vel = asset.data.root_link_lin_vel_w  # (B, 3)
  return (lin_vel ** 2).sum(dim=1)          # (B,)


def head_above_feet_reward(
  env: ManagerBasedRlEnv,
  target_height: float,
  std: float,
  head_offset: float = 0.43,
  torso_asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  foot_asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward the head being target_height ABOVE the average foot height.

  Unlike head_height_reward (which uses an absolute world Z target), this reward
  measures head height RELATIVE to the feet. This is the approach used by HoST
  and has two key advantages:

  1. Sharper sitting-vs-standing distinction:
       Sit-up: head at ~0.93 m, feet at ~0.05 m → head above feet ≈ 0.88 m
       Standing: head at ~1.33 m, feet at ~0.05 m → head above feet ≈ 1.28 m
     The 0.40 m gap is 67% larger than the absolute gap (0.40 vs 0.40 m), and
     with std=0.25 m the rewards differ by 2.5× instead of 1.5× for absolute.

  2. Terrain-agnostic: on sloped terrain the relative height stays meaningful
     even as absolute head and foot heights both shift.

  head_z ≈ torso_z + head_offset × (−proj_gz), same approximation as head_height_reward.
  foot_z is the mean of the two ankle body world heights.

  torso_asset_cfg.body_ids: 1 body (e.g. torso_link).
  foot_asset_cfg.body_ids:  2 bodies (e.g. left_ankle_roll_link, right_ankle_roll_link).

  Args:
    target_height: Target head-above-feet height (m).
      G1 standing: head ~1.28 m above feet. Recommended 1.15 m — gives max
      reward when upright without requiring perfect posture.
    std: Gaussian width (m). 0.25 m produces:
      Sitting (0.88 m above feet): exp(-1.17) ≈ 0.31
      Standing (1.28 m above feet): exp(-0.27) ≈ 0.76
      A 2.5× difference vs the 1.5× at the absolute height formulation.
    head_offset: Head geom center height above torso_link origin (m). G1 = 0.43 m.
    torso_asset_cfg: SceneEntityCfg for the torso body (body_names set per robot).
    foot_asset_cfg:  SceneEntityCfg for the foot bodies (body_names set per robot,
      two bodies — left and right ankle roll links).
  """
  asset     = env.scene[torso_asset_cfg.name]
  origins_z = env.scene.env_origins[:, 2]                               # (B,)

  torso_z = (
    asset.data.body_link_pos_w[:, torso_asset_cfg.body_ids, 2].squeeze(1) - origins_z
  )  # (B,)
  proj_gz = asset.data.projected_gravity_b[:, 2]                        # (B,)
  head_z  = torso_z + head_offset * (-proj_gz)                          # (B,)

  # Average foot Z (both ankles), relative to terrain.
  foot_z  = (
    asset.data.body_link_pos_w[:, foot_asset_cfg.body_ids, 2].mean(dim=1) - origins_z
  )  # (B,)

  head_above_feet = head_z - foot_z                                     # (B,)
  dist_sq = torch.square(head_above_feet - target_height)
  return torch.exp(-dist_sq / std**2)
