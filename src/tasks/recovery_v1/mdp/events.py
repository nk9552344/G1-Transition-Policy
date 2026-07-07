"""Custom reset events for recovery-v1: full floor-recovery training.

The core new event is ``reset_to_fallen_or_bent_pose``, which extends v3's
``reset_to_bent_pose`` with four completely fallen orientations (supine, prone,
side-left, side-right) so the robot learns to stand up from any ground-level
starting position.

Eight templates are sampled uniformly each episode:
  Fallen (50 %)                  Bent-upright (50 %)
  ─────────────────────────────  ────────────────────────────────────────────
  supine     (on back, face up)  home        (default standing, small bends)
  prone      (face down)         knees_bent  (moderate squat)
  side_left  (left side down)    squat       (deep knee bend)
  side_right (right side down)   deep_squat  (maximum knee bend)

Fallen templates set the pelvis height to 0.25 m and apply a random world-
frame yaw so the robot faces a different direction every episode.  Bent
templates use the FK-verified heights from v3.

Base heights for fallen poses (0.25 m) are intentionally set slightly above
the true contact height (~0.10–0.15 m) so the robot settles onto the ground
without floor penetration.  The small drop (<0.1 s at regular gravity) is
negligible for a 35 s episode.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from mjlab.envs.mdp.events import quat_from_euler_xyz, quat_mul
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")

# ── Fallen pose templates ──────────────────────────────────────────────────────
# Each quaternion encodes a 90° tilt from the default upright stance.
# Convention: wxyz (MuJoCo / mjlab standard).
#
#   supine     — fell backward: 90° around -Y axis → body +X points world +Z
#   prone      — fell forward:  90° around +Y axis → body +X points world -Z
#   side_left  — fell left:     90° around -X axis → body +Y points world +Z
#   side_right — fell right:    90° around +X axis → body +Y points world -Z
#
# base_z is the pelvis center height above terrain when lying flat.
# 0.25 m provides a safe drop margin above the actual body contact height.
# ──────────────────────────────────────────────────────────────────────────────
_SIN45 = math.sin(math.pi / 4)
_COS45 = math.cos(math.pi / 4)

FALLEN_POSE_CONFIGS: list[dict] = [
  {
    "label": "supine",
    "base_z": 0.25,
    "quat_wxyz": [_COS45, 0.0, -_SIN45, 0.0],    # 90° around -Y
  },
  {
    "label": "prone",
    "base_z": 0.25,
    "quat_wxyz": [_COS45, 0.0, _SIN45, 0.0],     # 90° around +Y
  },
  {
    "label": "side_left",
    "base_z": 0.25,
    "quat_wxyz": [_COS45, -_SIN45, 0.0, 0.0],    # 90° around -X
  },
  {
    "label": "side_right",
    "base_z": 0.25,
    "quat_wxyz": [_COS45, _SIN45, 0.0, 0.0],     # 90° around +X
  },
]

# ── Bent-upright pose templates (same as v3, FK-verified) ─────────────────────
BENT_POSE_CONFIGS: list[dict] = [
  {"knee": 0.300, "hip_pitch": -0.100, "ankle": -0.200, "base_z": 0.8000},
  {"knee": 0.669, "hip_pitch": -0.312, "ankle": -0.363, "base_z": 0.7725},
  {"knee": 1.200, "hip_pitch": -0.700, "ankle": -0.500, "base_z": 0.6918},
  {"knee": 1.800, "hip_pitch": -1.000, "ankle": -0.600, "base_z": 0.5616},
]

# Unified list for uniform sampling: 4 fallen + 4 bent = 8 templates total.
ALL_POSE_CONFIGS: list[dict] = [
  {**cfg, "type": "fallen"} for cfg in FALLEN_POSE_CONFIGS
] + [
  {**cfg, "type": "bent"}   for cfg in BENT_POSE_CONFIGS
]


def reset_to_fallen_or_bent_pose(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  all_pose_configs: list[dict],
  xy_pos_range: float,
  yaw_range: float,
  fallen_joint_perturbation: float,
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
  """Reset each environment to a randomly sampled fallen or bent-leg pose.

  Samples uniformly from ``all_pose_configs``.  Templates marked "fallen"
  set the robot flat on the ground with a random world-frame yaw.  Templates
  marked "bent" use the v3 logic: upright orientation + random yaw + leg-joint
  targeting.

  For fallen templates:
    1. Set pelvis z to template's base_z (safe drop height).
    2. Set orientation to the 90°-tilt quaternion, composed with random yaw
       so the robot lies in a different world direction each episode.
    3. Set all joints to default + fallen_joint_perturbation noise.

  For bent templates:
    1. Set pelvis z to template's FK-verified base_z.
    2. Set orientation to random yaw only (robot upright).
    3. Set leg joints (knee, hip_pitch, ankle) to template values + leg_perturbation.
    4. Set all other joints to default + other_perturbation.

  Both types clamp joint positions to soft limits and apply small random
  initial body and joint velocities.

  Args:
    all_pose_configs: Unified list of template dicts; each has a "type" key
      of "fallen" or "bent" plus the template-specific fields.
    xy_pos_range: ±range (m) for scattering robot x, y in each env cell.
    yaw_range: ±range (rad) for random initial yaw orientation.
    fallen_joint_perturbation: ±noise (rad) on ALL joints for fallen templates.
    leg_perturbation: ±noise (rad) on knee/hip/ankle for bent templates.
    other_perturbation: ±noise (rad) on non-leg joints for bent templates.
    joint_vel_range: ±range (rad/s) for initial joint velocities.
    lin_vel_range: ±range (m/s) for initial base linear velocity (x, y).
    ang_vel_range: ±range (rad/s) for initial base angular velocity.
    knee_cfg, hip_pitch_cfg, ankle_cfg: Resolved SceneEntityCfg objects
      providing joint IDs for the leg joints (used by bent templates).
    asset_cfg: Resolved SceneEntityCfg for the full robot.
  """
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)

  n = len(env_ids)
  device = env.device
  asset = env.scene[asset_cfg.name]

  # ── 1. Sample a template index per env (uniform over all templates) ──────
  template_idx = torch.randint(len(all_pose_configs), (n,), device=device)

  # Precompute boolean masks: which envs use a fallen vs. bent template.
  is_fallen = torch.tensor(
    [all_pose_configs[i]["type"] == "fallen" for i in range(len(all_pose_configs))],
    device=device,
    dtype=torch.bool,
  )[template_idx]  # (N,)

  # ── 2. Build joint positions ─────────────────────────────────────────────
  default_joint_pos = asset.data.default_joint_pos[env_ids].clone()  # (N, J)
  joint_pos = default_joint_pos.clone()

  # --- Fallen envs: all joints at default + fallen_joint_perturbation -------
  if is_fallen.any():
    fallen_noise = torch.empty_like(joint_pos).uniform_(
      -fallen_joint_perturbation, fallen_joint_perturbation
    )
    joint_pos[is_fallen] = (default_joint_pos + fallen_noise)[is_fallen]

  # --- Bent envs: other_perturbation base, then leg-joint overrides ---------
  if (~is_fallen).any():
    # Extract bent config fields into per-env tensors.
    knee_vals = torch.tensor(
      [all_pose_configs[i].get("knee", 0.0) for i in range(len(all_pose_configs))],
      dtype=torch.float32, device=device,
    )[template_idx]         # (N,)
    hip_vals = torch.tensor(
      [all_pose_configs[i].get("hip_pitch", 0.0) for i in range(len(all_pose_configs))],
      dtype=torch.float32, device=device,
    )[template_idx]
    ankle_vals = torch.tensor(
      [all_pose_configs[i].get("ankle", 0.0) for i in range(len(all_pose_configs))],
      dtype=torch.float32, device=device,
    )[template_idx]

    bent_mask = ~is_fallen
    bent_noise = torch.empty_like(joint_pos).uniform_(
      -other_perturbation, other_perturbation
    )
    joint_pos[bent_mask] = (default_joint_pos + bent_noise)[bent_mask]

    def _set_leg_joints(joint_ids, target_per_env, mask):
      if len(joint_ids) == 0:
        return
      ids = torch.tensor(joint_ids, dtype=torch.long, device=device)
      n_bent = int(mask.sum().item())
      noise = torch.empty(n_bent, len(ids), device=device).uniform_(
        -leg_perturbation, leg_perturbation
      )
      # Advanced indexing: joint_pos[mask][:, ids] = ... creates a temporary
      # copy and does NOT write back.  Use row/column indices instead.
      bent_rows = torch.where(mask)[0]  # (n_bent,)
      joint_pos[bent_rows.unsqueeze(1), ids.unsqueeze(0)] = (
        target_per_env[mask].unsqueeze(1) + noise
      )

    _set_leg_joints(knee_cfg.joint_ids,      knee_vals,  bent_mask)
    _set_leg_joints(hip_pitch_cfg.joint_ids, hip_vals,   bent_mask)
    _set_leg_joints(ankle_cfg.joint_ids,     ankle_vals, bent_mask)

  # Clamp all joint positions to soft limits.
  soft_limits = asset.data.soft_joint_pos_limits[env_ids]  # (N, J, 2)
  joint_pos.clamp_(soft_limits[..., 0], soft_limits[..., 1])

  # ── 3. Build joint velocities ─────────────────────────────────────────────
  joint_vel = asset.data.default_joint_vel[env_ids].clone() + torch.empty(
    n, joint_pos.shape[1], device=device
  ).uniform_(-joint_vel_range, joint_vel_range)

  all_joint_ids = torch.arange(joint_pos.shape[1], device=device)
  asset.write_joint_state_to_sim(joint_pos, joint_vel, joint_ids=all_joint_ids, env_ids=env_ids)

  # ── 4. Build root state ───────────────────────────────────────────────────
  default_root_state = asset.data.default_root_state[env_ids].clone()  # (N, 13)

  # --- Base heights: read from template for each env -----------------------
  base_z_vals = torch.tensor(
    [all_pose_configs[i]["base_z"] for i in range(len(all_pose_configs))],
    dtype=torch.float32, device=device,
  )[template_idx]  # (N,)

  # --- Positions -----------------------------------------------------------
  xy_noise = torch.empty(n, 2, device=device).uniform_(-xy_pos_range, xy_pos_range)
  z_noise  = torch.empty(n,    device=device).uniform_(-0.02, 0.05)  # asymmetric: don't go below ground

  positions = torch.cat(
    [
      default_root_state[:, 0:2] + xy_noise + env.scene.env_origins[env_ids, 0:2],
      (base_z_vals + z_noise + env.scene.env_origins[env_ids, 2]).unsqueeze(1),
    ],
    dim=1,
  )  # (N, 3)

  # --- Orientations --------------------------------------------------------
  # Random yaw for all envs (applied as world-frame rotation).
  yaw_angles = torch.empty(n, device=device).uniform_(-yaw_range, yaw_range)
  zeros_n = torch.zeros(n, device=device)
  yaw_quat = quat_from_euler_xyz(zeros_n, zeros_n, yaw_angles)  # (N, 4)

  # Fallen: compose random yaw with the 90° tilt quaternion.
  # q_final = q_yaw * q_tilt  — world-frame yaw applied to the tilted pose.
  # Bent: yaw only (same as v3).
  orientation = torch.zeros(n, 4, device=device)

  # --- Bent orientations (copy from v3) ------------------------------------
  if (~is_fallen).any():
    bent_default_quat = default_root_state[:, 3:7]  # identity for standing pose
    bent_orient = quat_mul(bent_default_quat, yaw_quat)  # (N, 4)
    orientation[~is_fallen] = bent_orient[~is_fallen]

  # --- Fallen orientations -------------------------------------------------
  if is_fallen.any():
    # Build per-env fallen base quaternion tensor from template.
    fallen_quat_table = torch.tensor(
      [
        all_pose_configs[i].get("quat_wxyz", [1.0, 0.0, 0.0, 0.0])
        for i in range(len(all_pose_configs))
      ],
      dtype=torch.float32, device=device,
    )[template_idx]  # (N, 4)

    # World-frame yaw: left-multiply (applied after tilt).
    fallen_orient = quat_mul(yaw_quat, fallen_quat_table)  # (N, 4)
    orientation[is_fallen] = fallen_orient[is_fallen]

  # --- Velocities ----------------------------------------------------------
  lin_vel = torch.zeros(n, 3, device=device)
  lin_vel[:, :2] = torch.empty(n, 2, device=device).uniform_(-lin_vel_range, lin_vel_range)
  ang_vel = torch.empty(n, 3, device=device).uniform_(-ang_vel_range, ang_vel_range)
  ang_vel[:, 2].mul_(0.5)  # yaw rate is less destabilising than roll/pitch

  # Assemble and write root state.
  root_state = torch.cat([positions, orientation, lin_vel, ang_vel], dim=1)  # (N, 13)
  asset.write_root_state_to_sim(root_state, env_ids=env_ids)
