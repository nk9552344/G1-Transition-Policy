"""Custom reset events for transition-v3: bent-pose recovery training.

The core new event is `reset_to_bent_pose`, a comprehensive reset function that
replaces the two separate `reset_base` + `reset_robot_joints` events used in v1/v2.
It samples a pose template (home, knees_bent, squat, deep_squat) independently for
each environment, sets the base height to the FK-computed correct value so feet land
on the ground without penetration, then applies small perturbations and initial
momentum before writing both root state and joint state to the simulator.

Base heights were computed via MuJoCo forward kinematics for each template pose
such that the ankle roll links are at the same ground-contact height as the HOME
standing configuration (≈ 0.051 m above the terrain plane).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.envs.mdp.events import quat_from_euler_xyz, quat_mul
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def reset_to_bent_pose(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  bent_pose_configs: list[dict],
  xy_pos_range: float,
  yaw_range: float,
  leg_perturbation: float,
  other_perturbation: float,
  joint_vel_range: float,
  lin_vel_range: float,
  ang_vel_range: float,
  knee_cfg: SceneEntityCfg,
  hip_pitch_cfg: SceneEntityCfg,
  ankle_cfg: SceneEntityCfg,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
  """Reset each environment to a randomly sampled bent-leg pose template.

  This function is a drop-in replacement for the separate reset_base +
  reset_robot_joints event pair used in v1/v2. It handles base position,
  base orientation, joint positions, and all velocities in one call.

  For each environment in env_ids:
    1. Sample one pose template uniformly from bent_pose_configs.
    2. Set the pelvis z to the template's base_z (FK-verified for ground contact).
    3. Set leg joints (knee, hip_pitch, ankle) to template values + leg_perturbation.
    4. Set all other joints to default + other_perturbation.
    5. Clamp all joint positions to soft limits.
    6. Apply small random initial linear and angular body velocity (like v2).
    7. Apply small random joint velocities.

  Args:
    bent_pose_configs: List of dicts, each with:
      - "knee": absolute knee joint target (rad)
      - "hip_pitch": absolute hip pitch target (rad, both sides)
      - "ankle": absolute ankle pitch target (rad, both sides)
      - "base_z": pelvis height above terrain (m) for ground contact at this pose
    xy_pos_range: ±range (m) for scattering robot x, y positions in each env cell.
    yaw_range: ±range (rad) for random initial yaw orientation.
    leg_perturbation: ±noise (rad) added to knee/hip/ankle template values.
    other_perturbation: ±noise (rad) added to all non-leg joints from their default.
    joint_vel_range: ±range (rad/s) for initial joint velocities.
    lin_vel_range: ±range (m/s) for initial base linear velocity (x, y axes).
    ang_vel_range: ±range (rad/s) for initial base angular velocity (all axes).
    knee_cfg: Resolved SceneEntityCfg for knee joints (provides joint IDs).
    hip_pitch_cfg: Resolved SceneEntityCfg for hip pitch joints.
    ankle_cfg: Resolved SceneEntityCfg for ankle pitch joints.
    asset_cfg: Resolved SceneEntityCfg for the full robot.
  """
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)

  n = len(env_ids)
  device = env.device
  asset = env.scene[asset_cfg.name]

  # ── 1. Sample a pose template index per env (uniform over configs) ──
  pose_idx = torch.randint(len(bent_pose_configs), (n,), device=device)

  # Build per-env tensors for knee, hip_pitch, ankle, and base_z.
  config_tensor = torch.tensor(
    [
      [c["knee"], c["hip_pitch"], c["ankle"], c["base_z"]]
      for c in bent_pose_configs
    ],
    dtype=torch.float32,
    device=device,
  )  # (num_templates, 4)
  chosen = config_tensor[pose_idx]  # (N, 4)

  target_knee      = chosen[:, 0]  # (N,)
  target_hip_pitch = chosen[:, 1]  # (N,)
  target_ankle     = chosen[:, 2]  # (N,)
  target_base_z    = chosen[:, 3]  # (N,)

  # ── 2. Build joint position tensor ──
  # Start from default joint positions (HOME_KEYFRAME values).
  default_joint_pos = asset.data.default_joint_pos[env_ids].clone()  # (N, J)
  joint_pos = default_joint_pos.clone()

  # Apply other_perturbation to ALL joints first (baseline noise).
  joint_pos += torch.empty_like(joint_pos).uniform_(-other_perturbation, other_perturbation)

  # Override leg joints with template values + leg_perturbation.
  def _set_leg_joints(joint_ids, target_per_env):
    if len(joint_ids) == 0:
      return
    # joint_ids may be a plain list; convert to tensor for indexing.
    ids = torch.tensor(joint_ids, dtype=torch.long, device=device)
    # Broadcast target (N,) to (N, num_matching_joints)
    noise = torch.empty(n, len(ids), device=device).uniform_(-leg_perturbation, leg_perturbation)
    joint_pos[:, ids] = target_per_env.unsqueeze(1) + noise

  _set_leg_joints(knee_cfg.joint_ids,      target_knee)
  _set_leg_joints(hip_pitch_cfg.joint_ids, target_hip_pitch)
  _set_leg_joints(ankle_cfg.joint_ids,     target_ankle)

  # Clamp to soft joint limits to ensure physical validity.
  soft_limits = asset.data.soft_joint_pos_limits[env_ids]  # (N, J, 2)
  joint_pos.clamp_(soft_limits[..., 0], soft_limits[..., 1])

  # ── 3. Build joint velocity tensor ──
  default_joint_vel = asset.data.default_joint_vel[env_ids].clone()  # (N, J)
  joint_vel = default_joint_vel + torch.empty_like(default_joint_vel).uniform_(
    -joint_vel_range, joint_vel_range
  )

  # Write joint state to sim.
  all_joint_ids = torch.arange(joint_pos.shape[1], device=device)
  asset.write_joint_state_to_sim(joint_pos, joint_vel, joint_ids=all_joint_ids, env_ids=env_ids)

  # ── 4. Build root state ──
  default_root_state = asset.data.default_root_state[env_ids].clone()  # (N, 13)

  # Position: scatter in x/y, set z from pose template + small height noise.
  # env.scene.env_origins handles the per-env grid offset.
  xy_noise = torch.empty(n, 2, device=device).uniform_(-xy_pos_range, xy_pos_range)
  z_noise  = torch.empty(n,    device=device).uniform_(-0.02, 0.02)

  positions = torch.cat(
    [
      default_root_state[:, 0:2] + xy_noise + env.scene.env_origins[env_ids, 0:2],
      (target_base_z + z_noise + env.scene.env_origins[env_ids, 2]).unsqueeze(1),
    ],
    dim=1,
  )  # (N, 3)

  # Orientation: random yaw rotation applied on top of default (upright).
  yaw_angles = torch.empty(n, device=device).uniform_(-yaw_range, yaw_range)
  zeros_n    = torch.zeros(n, device=device)
  yaw_quat   = quat_from_euler_xyz(zeros_n, zeros_n, yaw_angles)
  orientation = quat_mul(default_root_state[:, 3:7], yaw_quat)  # (N, 4)

  # Velocity: small initial linear (x, y; no vertical launch) and angular.
  lin_vel = torch.zeros(n, 3, device=device)
  lin_vel[:, :2] = torch.empty(n, 2, device=device).uniform_(-lin_vel_range, lin_vel_range)
  ang_vel = torch.empty(n, 3, device=device).uniform_(-ang_vel_range, ang_vel_range)
  # Yaw rate: half the roll/pitch range (spinning is less destabilising).
  ang_vel[:, 2].mul_(0.5)

  # Assemble root state (pos, quat, lin_vel, ang_vel) and write.
  root_state = torch.cat([positions, orientation, lin_vel, ang_vel], dim=1)  # (N, 13)
  asset.write_root_state_to_sim(root_state, env_ids=env_ids)
