"""Custom termination functions for recovery-v1.

The key new function is bad_orientation_while_elevated, which restores the
"fell over" termination signal that v1/v2/v3 relied on, but ONLY for robots
that are currently elevated (base height > height_threshold).

This height gate is the key difference from the vanilla bad_orientation:
  - Robot starts FLAT (h ≈ 0.10–0.15 m < threshold): no termination → the
    episode continues and the robot has a full 35 s to learn to get up.
  - Robot starts UPRIGHT/BENT (h ≈ 0.56–0.80 m > threshold): if it tips
    past limit_angle AFTER the grace period, the episode terminates.
  - Robot is MID-RECOVERY (h between threshold and top): partial stands above
    threshold that fall back are also terminated → discourages unstable
    intermediate attempts, not the recovery trajectory itself.

Grace period (grace_period_steps):
  Bent starts (home=0.80 m, knees_bent=0.77 m, squat=0.69 m) begin above the
  0.65 m threshold.  The reset event applies angular velocity perturbations
  (ang_vel_range=0.30 rad/s) which can tip the robot before the policy has
  executed even one meaningful action.  Without a grace period, the -50
  termination penalty is attributed by GAE (lam=0.97) to the FIRST step,
  even though the policy had no control over the initial perturbation.
  The grace period suppresses termination for the first N steps, giving the
  policy a chance to respond before the "don't fall" signal fires.
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
  grace_period_steps: int = 0,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Terminate when the robot tips past limit_angle while elevated.

  Combines the classic bad_orientation check with a base-height gate and an
  optional grace period that suppresses termination in the first N steps:

    terminate = (tilt_angle > limit_angle)
              AND (base_height > height_threshold)
              AND (episode_step >= grace_period_steps)

  Height gate rationale
  ─────────────────────
  Fallen initial states (supine/prone) settle at base_height ≈ 0.10–0.15 m,
  well below the recommended threshold of 0.65 m.  These episodes never
  trigger the termination — the robot has the full 35 s to learn to get up.

  Once the robot has partially recovered and base_height exceeds 0.65 m, it
  is subject to the "don't fall" signal.  This is intentional: a robot at
  0.70 m should not tip sideways by 75°.

  Grace period rationale
  ──────────────────────
  Bent starts (home=0.80 m, knees_bent=0.77 m, squat=0.69 m) start above
  the 0.65 m threshold with angular velocity perturbations up to 0.30 rad/s.
  Without a grace period, tipping-from-perturbation terminates the episode
  before the policy has taken a meaningful action, and the -50 penalty is
  attributed by GAE back to step 0.  A grace_period_steps=20 (0.4 s) gives
  the policy time to respond to the initial perturbation before the signal
  fires.  The 0.4 s corresponds to the onset of the GAE effective horizon
  at lam=0.97 — rewards beyond this horizon still influence the policy but
  with less than 50 % of their undiscounted weight.

  Args:
    limit_angle: Tilt angle (radians) above which termination fires.
      Recommended: math.radians(75.0) — same as v3.
    height_threshold: Minimum base height (m) for termination to be active.
      Recommended: 0.65 m — raised from v3's 0.50 m to avoid penalising the
      prone→bridge→squat path which passes through 0.50 m while still tilted.
    grace_period_steps: Number of steps at the start of each episode during
      which the termination is suppressed.  Recommended: 20 (= 0.4 s at 50 Hz).
      Set to 0 to disable (original behaviour).
    asset_cfg: Resolved SceneEntityCfg for the robot.

  Returns:
    Bool tensor [B]: True = terminate this env.
  """
  asset = env.scene[asset_cfg.name]

  # Tilt angle from the projected gravity vector.
  proj_gz = asset.data.projected_gravity_b[:, 2]                            # [B]
  tilt_angle = torch.acos((-proj_gz).clamp(-1.0 + 1e-6, 1.0 - 1e-6))      # [B], radians

  # Current base height above terrain.
  base_height = (
    asset.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
  )                                                                          # [B]

  bad_orient = tilt_angle > limit_angle      # [B]
  elevated   = base_height > height_threshold  # [B]

  # Grace period: suppress termination for the first grace_period_steps steps.
  # env.episode_length_buf tracks steps elapsed in the current episode per env.
  if grace_period_steps > 0:
    past_grace = env.episode_length_buf >= grace_period_steps               # [B]
    return bad_orient & elevated & past_grace

  return bad_orient & elevated
