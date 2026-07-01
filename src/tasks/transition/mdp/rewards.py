"""Reward functions for the transition-to-neutral-standing task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def pose_convergence(
  env: ManagerBasedRlEnv,
  std: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward closeness to the default (neutral standing) joint configuration.

  Computes exp(-mean(error²) / std²) over all controlled joints.
  At neutral (error=0): reward = 1.0.
  At std-sized error: reward ≈ 0.37.

  Tune std to balance learning signal range vs. precision requirement:
  larger std → nonzero gradient from far away; smaller std → reward only
  when very close to neutral.
  """
  asset: Entity = env.scene[asset_cfg.name]
  q = asset.data.joint_pos[:, asset_cfg.joint_ids]
  q_default = asset.data.default_joint_pos[:, asset_cfg.joint_ids]
  mse = torch.mean(torch.square(q - q_default), dim=1)
  return torch.exp(-mse / std**2)


def joint_vel_penalty(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize joint velocities to encourage settling at the target pose.

  Returns the summed squared joint velocities. Penalizing this term drives
  the policy to arrive at neutral smoothly and remain stationary.
  """
  asset: Entity = env.scene[asset_cfg.name]
  vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
  return torch.sum(torch.square(vel), dim=1)


def both_feet_contact(
  env: ManagerBasedRlEnv,
  sensor_name: str,
) -> torch.Tensor:
  """Reward having both feet simultaneously in contact with the ground.

  Returns 1.0 when all tracked foot bodies are in contact, 0.0 otherwise.
  Encourages the robot to maintain a stable two-foot stance throughout the
  transition rather than shifting weight to a single foot.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.found is not None
  in_contact = sensor.data.found > 0  # [B, N]
  return in_contact.all(dim=1).float()
