"""Reward functions for recovery-v2.

Adds one new reward over recovery-v1:

orientation_velocity_reward
───────────────────────────
Provides an analytic reward signal for angular velocity that improves body
orientation toward upright.  This is the key addition that makes side-lying
recovery (side_left, side_right) learnable.

Mathematical derivation
-----------------------
Let g_b = projected_gravity_b (body-frame gravity, -1 when upright, 0 when flat).
Let target = [0, 0, -1] (desired upright gravity direction in body frame).
Let error = g_b - target.

The time derivative of g_b under body rotation ω_b is:
    dg_b/dt = -(ω_b × g_b)

The rate of change of the squared orientation error is:
    d(||error||²)/dt = 2 * error · dg_b/dt
                     = -2 * error · (ω_b × g_b)

Improvement happens when this rate is negative (error shrinking).  The reward is:
    r = max(0, error · (ω_b × g_b))

which is positive exactly when the angular velocity reduces the orientation error.

Verification (all four fallen poses):
  supine    g_b ≈ [±1, 0, 0]  correct ω → reward > 0 ✓
  prone     g_b ≈ [∓1, 0, 0]  correct ω → reward > 0 ✓
  side_left  g_b ≈ [0,-1, 0]  correct ω → reward > 0 ✓
  side_right g_b ≈ [0, 1, 0]  correct ω → reward > 0 ✓
  upright   g_b = [0, 0,-1]  error = 0  → reward = 0 ✓ (no spurious gradient)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")

_UPRIGHT_GRAVITY_B = torch.tensor([0.0, 0.0, -1.0])


def orientation_velocity_reward(
    env: ManagerBasedRlEnv,
    scale: float,
    clip_val: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward angular velocity that reduces orientation error toward upright.

    Returns a per-environment scalar in [0, clip_val * scale] that is positive
    when the body's angular velocity is aligned with reducing the tilt from
    upright, and zero otherwise (no gradient for angular velocity that worsens
    orientation or when already upright).

    Args:
        scale:    Multiplied onto the clamped improvement value.
        clip_val: Upper bound on the raw improvement before scaling.  Prevents
                  reward hacking from extremely large angular velocities.
        asset_cfg: SceneEntityCfg for the robot articulation.
    """
    asset = env.scene[asset_cfg.name]
    g_b = asset.data.projected_gravity_b                    # (N, 3)
    omega_b = asset.data.root_link_ang_vel_b                # (N, 3) body-frame ω

    target = _UPRIGHT_GRAVITY_B.to(device=env.device)       # [0, 0, -1]
    error = g_b - target.unsqueeze(0)                       # (N, 3)

    cross = torch.linalg.cross(omega_b, g_b)                # (N, 3): ω_b × g_b
    improvement = (error * cross).sum(dim=-1)               # (N,)

    return improvement.clamp(min=0.0, max=clip_val) * scale
