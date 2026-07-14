"""Custom reset events for recovery-v1: full floor-recovery training.

The core new event is ``reset_to_fallen_or_bent_pose``, which extends v3's
``reset_to_bent_pose`` with two completely fallen orientations (supine, prone)
so the robot learns to stand up from the most common ground-level positions.

Nine templates are sampled uniformly each episode (22 % fallen, 22 % sitting, 11 % squat-lean, 44 % bent):
  Fallen (22 %)                  Sitting-up (22 %)                 Bent-upright (44 %)
  ─────────────────────────────  ───────────────────────────────── ──────────────────────────────────────────
  supine     (on back, face up)  sitting_low  (40° backward lean)  home        (default standing, small bends)
  prone      (face down)         sitting_high (30° backward lean)  knees_bent  (moderate squat)
                                                                    squat       (deep knee bend)
                                                                    deep_squat  (maximum knee bend)
  Squat-lean (11 %): squat_lean — 20° lean, knee=1.2 rad, pelvis 0.52 m  (sit-to-stand transition zone)

Sampling ratio rationale
────────────────────────
Starting with 50 % fallen caused complete training failure: the 50/50 split
produces conflicting gradient signals — fallen episodes need *large* exploratory
actions while standing episodes need *near-zero* actions — but RSL-RL's single
scalar std cannot accommodate both simultaneously.  At std=0.3 (the collapsed
value after 17 k iters) the policy was simultaneously too timid for floor
recovery and too noisy for balance maintenance.

Reducing fallen fraction to 22 % lets the policy first stabilise the standing
skill (where the reward is high and the gradient is clear), then generalise to
floor recovery.  The two sitting-up templates (22 %) give the policy direct
experience of the sit-to-stand transition without requiring it to first discover
the full push-up sequence.  The squat_lean template (11 %) covers the transition
zone that exploration from a sitting start rarely reaches: 20° lean + knee=1.2 rad
+ pelvis at 0.52 m, exactly where shank_orientation_reward provides its strongest
gradient toward standing.  Side-lying poses are deferred to recovery_v2.

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
_SIN20 = math.sin(math.radians(20))  # half-angle for 40° backward lean
_COS20 = math.cos(math.radians(20))
_SIN15 = math.sin(math.radians(15))  # half-angle for 30° backward lean
_COS15 = math.cos(math.radians(15))
_SIN10 = math.sin(math.radians(10))  # half-angle for 20° lean (squat_lean templates)
_COS10 = math.cos(math.radians(10))

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

# ── Sitting-up pose templates ──────────────────────────────────────────────────
# These templates initialise the robot in a partial sit-up orientation, giving
# the policy DIRECT experience of the sit-to-stand phase without requiring it to
# first succeed at the full push-up sequence.
#
# Both use type="fallen" so the existing reset logic applies:
#   tilt quaternion + random world-frame yaw + ±0.6 rad joint perturbation.
# The joint perturbation gives a diverse set of arm/leg positions to explore from.
#
# sitting_low  — 40° backward lean (proj_gz ≈ -0.77), pelvis at 0.28 m.
#   Robot is in the classic sit-up position: pelvis near the floor, upper body
#   raised. Covers the region where orientation_recovery and torso_height_reward
#   provide the primary upward gradient.
#
# sitting_high — 30° backward lean (proj_gz ≈ -0.87), pelvis at 0.38 m.
#   Pelvis is above the feet_proximity_reward gate (0.35 m), so the policy
#   immediately receives feet-proximity gradient and can explore the knee-tuck
#   motion. Covers the transition region between sit-up and squat.
# ──────────────────────────────────────────────────────────────────────────────
SITTING_POSE_CONFIGS: list[dict] = [
  {
    "label": "sitting_low",
    "base_z": 0.28,
    "quat_wxyz": [_COS20, 0.0, -_SIN20, 0.0],   # 40° backward lean around -Y
  },
  {
    "label": "sitting_high",
    "base_z": 0.38,
    "quat_wxyz": [_COS15, 0.0, -_SIN15, 0.0],   # 30° backward lean; pelvis above feet_proximity gate (0.35 m)
  },
]

# ── Bent-upright pose templates (same as v3, FK-verified) ─────────────────────
BENT_POSE_CONFIGS: list[dict] = [
  {"knee": 0.300, "hip_pitch": -0.100, "ankle": -0.200, "base_z": 0.8000},
  {"knee": 0.669, "hip_pitch": -0.312, "ankle": -0.363, "base_z": 0.7725},
  {"knee": 1.200, "hip_pitch": -0.700, "ankle": -0.500, "base_z": 0.6918},
  {"knee": 1.800, "hip_pitch": -1.000, "ankle": -0.600, "base_z": 0.5616},
]

# ── Squat-lean pose templates ───────────────────────────────────────────────────
# Starting state in the sit-to-stand transition zone: pelvis elevated (0.52 m),
# knees deeply bent (1.2 rad), slight backward lean (20°). Exploration from
# sitting templates rarely reaches this configuration because it requires
# coordinated knee-bend + lean reduction, yet this is exactly where
# shank_orientation_reward gives the strongest gradient toward standing.
# type="squat_lean": uses bent-style leg-joint targeting PLUS fallen-style tilt
# quaternion (override applied in the is_squat_lean block of the reset function).
# ──────────────────────────────────────────────────────────────────────────────
SQUAT_LEAN_CONFIGS: list[dict] = [
  {
    "label": "squat_lean",
    "base_z": 0.52,
    "quat_wxyz": [_COS10, 0.0, -_SIN10, 0.0],   # 20° backward lean
    "knee": 1.20,
    "hip_pitch": -0.60,
    "ankle": -0.40,
  },
]

# Unified list: 4 fallen + 2 sitting + 1 squat_lean + 4 bent = 11 templates (36 / 18 / 9 / 36 %).
# Side-lying templates now included: G1 frequently falls to its side and the
# policy must learn to recover from that configuration.  Without side_left /
# side_right starts the policy never trains for the lateral-roll-to-prone step.
#
# Fallen fraction increased from 22 % → 36 %: more fallen experience reduces
# the std-bias that 44 % upright starts produced (balance skill dominated,
# leaving too little exploration capacity for recovery).
ALL_POSE_CONFIGS: list[dict] = [
  {**cfg, "type": "fallen"}     for cfg in FALLEN_POSE_CONFIGS        # supine, prone, side_left, side_right
] + [
  {**cfg, "type": "fallen"}     for cfg in SITTING_POSE_CONFIGS       # sitting-up states
] + [
  {**cfg, "type": "squat_lean"} for cfg in SQUAT_LEAN_CONFIGS         # lean + knees-bent transition
] + [
  {**cfg, "type": "bent"}       for cfg in BENT_POSE_CONFIGS
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
  fallen_lin_vel_range: float | None = None,
  fallen_ang_vel_range: float | None = None,
  fallen_joint_vel_range: float | None = None,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
  """Reset each environment to a randomly sampled fallen or bent-leg pose.

  Samples uniformly from ``all_pose_configs``.  Templates marked "fallen"
  set the robot flat on the ground with a random world-frame yaw.  Templates
  marked "bent" use the v3 logic: upright orientation + random yaw + leg-joint
  targeting.  Templates marked "squat_lean" combine both: bent-style leg-joint
  targeting (knee/hip/ankle from template) plus fallen-style tilt quaternion,
  placing the robot in the mid-recovery transition zone (lean + knees bent).

  For fallen templates:
    1. Set pelvis z to template's base_z (safe drop height).
    2. Set orientation to the 90°-tilt quaternion, composed with random yaw
       so the robot lies in a different world direction each episode.
    3. Set all joints to default + fallen_joint_perturbation noise.
    4. Apply fallen-specific (small) initial velocities if provided.

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
    joint_vel_range: ±range (rad/s) for initial joint velocities (bent states).
    lin_vel_range: ±range (m/s) for initial base linear velocity (bent states).
    ang_vel_range: ±range (rad/s) for initial base angular velocity (bent states).
    fallen_lin_vel_range: ±range (m/s) for fallen-state base linear velocity.
      If None, falls back to lin_vel_range. Recommend 0.05 m/s — large initial
      velocities collide with ground contact geometry and create the oscillation
      that the policy can mistakenly exploit as a velocity-based reward signal.
    fallen_ang_vel_range: ±range (rad/s) for fallen-state base angular velocity.
      If None, falls back to ang_vel_range. Recommend 0.10 rad/s.
    fallen_joint_vel_range: ±range (rad/s) for fallen-state joint velocities.
      If None, falls back to joint_vel_range. Recommend 0.05 rad/s.
    knee_cfg, hip_pitch_cfg, ankle_cfg: Resolved SceneEntityCfg objects
      providing joint IDs for the leg joints (used by bent templates).
    asset_cfg: Resolved SceneEntityCfg for the full robot.
  """
  # Resolve fallen-specific velocity ranges (default to generic if not provided).
  _fallen_lin_vel   = fallen_lin_vel_range   if fallen_lin_vel_range   is not None else lin_vel_range
  _fallen_ang_vel   = fallen_ang_vel_range   if fallen_ang_vel_range   is not None else ang_vel_range
  _fallen_joint_vel = fallen_joint_vel_range if fallen_joint_vel_range is not None else joint_vel_range
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
  # Use smaller velocity range for fallen states to prevent immediate floor
  # bouncing: large initial joint velocities cause arms/legs to thrash against
  # ground contact geometry on the first few steps, generating oscillations that
  # the policy can exploit as velocity-based reward signals.
  joint_vel = asset.data.default_joint_vel[env_ids].clone()
  bent_jvel_noise = torch.empty(n, joint_pos.shape[1], device=device).uniform_(
    -joint_vel_range, joint_vel_range
  )
  fallen_jvel_noise = torch.empty(n, joint_pos.shape[1], device=device).uniform_(
    -_fallen_joint_vel, _fallen_joint_vel
  )
  joint_vel[~is_fallen]  += bent_jvel_noise[~is_fallen]
  joint_vel[is_fallen]   += fallen_jvel_noise[is_fallen]

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

  # --- Squat-lean orientations: tilt quaternion overrides upright from bent block -
  # squat_lean type is NOT is_fallen, so the ~is_fallen branch wrote upright
  # orientation for these envs first. Override here with the template's tilt
  # quaternion. Leg joints are already correct: the ~is_fallen branch reads
  # "knee"/"hip_pitch"/"ankle" from the config dict and sets them via _set_leg_joints.
  is_squat_lean = torch.tensor(
    [all_pose_configs[i]["type"] == "squat_lean" for i in range(len(all_pose_configs))],
    device=device,
    dtype=torch.bool,
  )[template_idx]

  if is_squat_lean.any():
    squat_lean_quat_table = torch.tensor(
      [
        all_pose_configs[i].get("quat_wxyz", [1.0, 0.0, 0.0, 0.0])
        for i in range(len(all_pose_configs))
      ],
      dtype=torch.float32, device=device,
    )[template_idx]
    orientation[is_squat_lean] = quat_mul(yaw_quat, squat_lean_quat_table)[is_squat_lean]

  # --- Velocities ----------------------------------------------------------
  # Fallen states get near-zero initial velocities; bent states get the full
  # perturbation range.  See fallen_lin_vel_range / fallen_ang_vel_range args.
  lin_vel = torch.zeros(n, 3, device=device)
  bent_lin = torch.empty(n, 2, device=device).uniform_(-lin_vel_range, lin_vel_range)
  fallen_lin = torch.empty(n, 2, device=device).uniform_(-_fallen_lin_vel, _fallen_lin_vel)
  lin_vel[~is_fallen, :2] = bent_lin[~is_fallen]
  lin_vel[is_fallen,  :2] = fallen_lin[is_fallen]

  ang_vel = torch.zeros(n, 3, device=device)
  bent_ang = torch.empty(n, 3, device=device).uniform_(-ang_vel_range, ang_vel_range)
  fallen_ang = torch.empty(n, 3, device=device).uniform_(-_fallen_ang_vel, _fallen_ang_vel)
  ang_vel[~is_fallen] = bent_ang[~is_fallen]
  ang_vel[is_fallen]  = fallen_ang[is_fallen]
  ang_vel[:, 2].mul_(0.5)  # yaw rate is less destabilising than roll/pitch

  # Assemble and write root state.
  root_state = torch.cat([positions, orientation, lin_vel, ang_vel], dim=1)  # (N, 13)
  asset.write_root_state_to_sim(root_state, env_ids=env_ids)
