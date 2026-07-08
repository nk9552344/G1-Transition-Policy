"""Custom termination functions for recovery-v1.

The key new function is bad_orientation_while_elevated, which restores the
"fell over" termination signal that v1/v2/v3 relied on, but ONLY for robots
that are currently elevated (base height > height_threshold).

This height gate is the key difference from the vanilla bad_orientation:
  - Robot starts FLAT (h ≈ 0.10–0.15 m < threshold): no termination → the
    episode continues and the robot has a full 35 s to learn to get up.
  - Robot starts UPRIGHT/BENT (h ≈ 0.56–0.80 m > threshold): if it tips
    past limit_angle, the episode terminates with is_terminated(-200) → a
    strong gradient signal to maintain balance.
  - Robot is MID-RECOVERY (h between threshold and top): partial stands above
    threshold that fall back are also terminated → discourages unstable
    intermediate attempts, not the recovery trajectory itself (which passes
    through much smaller tilt angles during a successful stand-up).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def bad_orientation_while_elevated(
  env: ManagerBasedRlEnv,
  limit_angle: float,
  height_threshold: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Terminate when the robot tips past limit_angle while elevated.

  Combines the classic bad_orientation check with a base-height gate:

    terminate = (tilt_angle > limit_angle) AND (base_height > height_threshold)

  Where tilt_angle is the angle between the body +Z axis and world +Z axis,
  computed from the root-link projected gravity.

  Height gate rationale
  ─────────────────────
  Fallen initial states (supine/prone) settle at base_height ≈ 0.10–0.15 m,
  well below the recommended threshold of 0.50 m.  Because these episodes
  never raise the height above 0.50 m during the first few thousand steps
  of training, the termination never fires for them — the robot has the
  full episode length to discover the get-up motion.

  Once the robot has partially recovered and base_height exceeds 0.50 m, it
  is subject to the same "don't fall" signal as episodes that started upright.
  This is intentional: a robot at 0.55 m should not tip sideways by 75°.

  Args:
    limit_angle: Tilt angle (radians) above which termination fires.
      Recommended: math.radians(75.0) — same as v3.
    height_threshold: Minimum base height (m) for termination to be active.
      Recommended: 0.50 — below all bent/standing poses, above fallen poses.
    asset_cfg: Resolved SceneEntityCfg for the robot.

  Returns:
    Bool tensor [B]: True = terminate this env.
  """
  asset = env.scene[asset_cfg.name]

  # Tilt angle from the projected gravity vector.
  # projected_gravity_b[:, 2] = -1 when upright, 0 when horizontal.
  proj_gz = asset.data.projected_gravity_b[:, 2]  # [B]
  tilt_angle = torch.acos((-proj_gz).clamp(-1.0 + 1e-6, 1.0 - 1e-6))  # [B], radians

  # Current base height above terrain.
  base_height = (
    asset.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
  )  # [B]

  bad_orient   = tilt_angle > limit_angle          # [B]
  elevated     = base_height > height_threshold    # [B]

  return bad_orient & elevated
