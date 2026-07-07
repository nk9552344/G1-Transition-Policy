"""Reward functions for recovery-v1: full floor-recovery training.

Three new reward functions added on top of the v3 reward set:

  orientation_recovery
    Primary get-up signal.  Rewards the torso having its gravity projection
    close to [0, 0, -1] (world up mapped to body -Z = upright).  Uses
    (proj_gz + 1.0)² as the distance metric so the reward correctly
    distinguishes upright (0), flat (1), and upside-down (4), providing a
    gradient from every starting orientation.

  height_recovery
    Secondary get-up signal.  Rewards the robot base rising toward the
    standing pelvis height.  The Gaussian std is wide enough to give a
    nonzero gradient from the 0.25 m fallen starting height all the way up
    to the ~0.80 m standing height.

  pose_convergence_gated
    Joint-position convergence reward gated by how upright the robot is.
    When flat (proj_gz ≈ 0) the gate is ~0 so the robot is not rewarded
    for holding default joint angles while on the ground.  When upright
    (proj_gz ≈ -1) the gate is 1 and the full pose_convergence signal
    applies.  This prevents the policy from learning to "stay flat in
    default pose" as a local optimum.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg

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
    proj_gz = -1.0  →  robot is upright     (reward = 1.0)
    proj_gz =  0.0  →  robot is lying flat  (reward = exp(-1 / std²))
    proj_gz = +1.0  →  robot is upside-down (reward = exp(-4 / std²))

  Unlike body_orientation_l2 (which gives identical values for upright and
  upside-down), this reward correctly distinguishes all orientations and
  provides a gradient from every ground-lying starting pose.

  Recommended std=1.0: gives reward ≈ 0.37 when flat and ≈ 0.018 when
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

  Computes exp(-(base_z - target_height)² / std²) relative to the env
  origin terrain height.

  Recommended target_height=0.78 (G1 standing pelvis ≈ 0.80 m, slight
  margin), std=0.65 so the gradient is nonzero from the 0.25 m fallen
  starting height all the way to target.

  At fallen (base_z ≈ 0.25 m): reward ≈ 0.45.
  At mid-rise (base_z ≈ 0.50 m): reward ≈ 0.84.
  At target   (base_z ≈ 0.78 m): reward = 1.0.
  """
  asset = env.scene[asset_cfg.name]
  base_height = asset.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]  # [B]
  dist_sq = torch.square(base_height - target_height)
  return torch.exp(-dist_sq / std**2)


def pose_convergence_gated(
  env: ManagerBasedRlEnv,
  std: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Joint-pose convergence reward, smoothly gated by upright orientation.

  Gate: upright_weight = clamp(-proj_gz, 0, 1)
    When flat        (proj_gz ≈  0): gate ≈ 0  → no pose reward
    When upright     (proj_gz ≈ -1): gate = 1  → full pose reward
    When partially tilted (45°):     gate ≈ 0.7 → partial pose reward

  This prevents the policy exploiting "stay flat in default joint config"
  as a pose-convergence reward shortcut.  The pose reward only activates
  as the robot rises toward vertical.

  Same Gaussian kernel as pose_convergence: exp(-MSE(q, q_default) / std²).
  """
  asset = env.scene[asset_cfg.name]

  q = asset.data.joint_pos[:, asset_cfg.joint_ids]
  q_default = asset.data.default_joint_pos[:, asset_cfg.joint_ids]
  mse = torch.mean(torch.square(q - q_default), dim=1)  # [B]

  proj_gz = asset.data.projected_gravity_b[:, 2]  # [B]
  upright_weight = (-proj_gz).clamp(0.0, 1.0)     # [B]: 0 when flat, 1 when upright

  return torch.exp(-mse / std**2) * upright_weight
