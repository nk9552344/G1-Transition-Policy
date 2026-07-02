"""Reward functions for transition-v2: momentum-aware neutral-standing task.

New additions over the base transition policy:
  - angular_velocity_convergence: explicitly reward damping body angular velocity to zero.
  - linear_velocity_convergence: explicitly reward damping body linear velocity to zero.
  - hold_bonus: binary bonus when the robot simultaneously holds neutral pose
    AND near-zero angular AND linear velocity (the "locked in" state).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def angular_velocity_convergence(
  env: ManagerBasedRlEnv,
  std: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward low root body angular velocity (robot not rocking or spinning).

  Returns exp(-|ω_b|² / std²) ∈ [0, 1].
  At std-sized angular velocity magnitude: reward ≈ 0.37.
  At zero angular velocity: reward = 1.0.

  std is in rad/s. For std=0.3: at ±0.3 rad/s norm reward ≈ 0.37,
  at the initial ±0.3 rad/s angular velocity the policy has a clear
  gradient to damp toward zero.
  """
  asset = env.scene[asset_cfg.name]
  ang_vel_b = asset.data.root_link_ang_vel_b  # [B, 3]
  ang_vel_sq = torch.sum(torch.square(ang_vel_b), dim=1)  # [B]
  return torch.exp(-ang_vel_sq / std**2)


def linear_velocity_convergence(
  env: ManagerBasedRlEnv,
  std: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward low root body linear velocity (robot not drifting).

  Returns exp(-|v_b|² / std²) ∈ [0, 1].
  At std-sized linear velocity magnitude: reward ≈ 0.37.
  At zero linear velocity: reward = 1.0.

  std is in m/s. For std=0.2: at ±0.2 m/s norm reward ≈ 0.37,
  matching the initial ±0.2 m/s linear momentum range.
  """
  asset = env.scene[asset_cfg.name]
  lin_vel_b = asset.data.root_link_lin_vel_b  # [B, 3]
  lin_vel_sq = torch.sum(torch.square(lin_vel_b), dim=1)  # [B]
  return torch.exp(-lin_vel_sq / std**2)


def hold_bonus(
  env: ManagerBasedRlEnv,
  pose_threshold: float,
  ang_vel_threshold: float,
  lin_vel_threshold: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Binary bonus for simultaneously holding neutral pose with near-zero momentum.

  Returns 1.0 iff ALL three conditions hold simultaneously:
    - mean |q - q_default| < pose_threshold  (joints at neutral, in radians)
    - |ω_b| < ang_vel_threshold              (not rocking, in rad/s)
    - |v_b| < lin_vel_threshold              (not drifting, in m/s)

  This incentivizes "locking in" to neutral rather than oscillating through it.
  It provides no gradient direction — the gradient comes from the other reward
  terms driving each condition independently. The hold_bonus fires as a bonus
  once the policy has already learned to satisfy all three.
  """
  asset = env.scene[asset_cfg.name]

  q = asset.data.joint_pos[:, asset_cfg.joint_ids]
  q_default = asset.data.default_joint_pos[:, asset_cfg.joint_ids]
  pose_ok = torch.mean(torch.abs(q - q_default), dim=1) < pose_threshold  # [B]

  ang_vel_b = asset.data.root_link_ang_vel_b  # [B, 3]
  ang_ok = torch.norm(ang_vel_b, dim=1) < ang_vel_threshold  # [B]

  lin_vel_b = asset.data.root_link_lin_vel_b  # [B, 3]
  lin_ok = torch.norm(lin_vel_b, dim=1) < lin_vel_threshold  # [B]

  return (pose_ok & ang_ok & lin_ok).float()
