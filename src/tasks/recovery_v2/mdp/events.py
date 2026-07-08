"""Custom reset events for recovery-v2: all-orientation floor-recovery training.

Extends recovery-v1 by including all four fallen orientations:
  Fallen (50 %)                          Bent-upright (50 %)
  ─────────────────────────────────────  ─────────────────────────────────────
  supine      (on back, face up)         home        (default standing)
  prone       (face down)                knees_bent  (moderate squat)
  side_left   (fell to the left)         squat       (deep knee bend)
  side_right  (fell to the right)        deep_squat  (maximum knee bend)

Recovery-v1 explicitly deferred side_left and side_right to recovery-v2 because
the lateral-rolling motion requires a reward signal that v1 lacked.  Recovery-v2
adds orientation_velocity_reward (see rewards.py) which gives an analytic gradient
for ANY angular velocity that reduces the orientation error, making side-lying
recovery learnable.

Sampling ratio rationale
────────────────────────
With orientation_velocity_reward providing direct gradient signal even for lateral
rolls, the 50 / 50 split is now stable.  The key failure mode in earlier designs
(collapsed std at 50 / 50 in recovery-v1) was due to missing gradient signal for
flat episodes; upward_base_velocity (v1 addition) solved supine/prone, and
orientation_velocity_reward solves side-lying.

This module re-exports the reset function from recovery-v1 unchanged — only the
pose config list differs.
"""

from __future__ import annotations

from src.tasks.recovery_v1.mdp.events import (   # noqa: F401
    BENT_POSE_CONFIGS,
    FALLEN_POSE_CONFIGS,
    reset_to_fallen_or_bent_pose,
)

# All 4 fallen + 4 bent = 8 templates (50 / 50 split).
# Recovery-v1 used only supine + prone (2/6 = 33 %).
# Recovery-v2 includes all four fallen orientations (4/8 = 50 %).
ALL_POSE_CONFIGS: list[dict] = [
    {**cfg, "type": "fallen"} for cfg in FALLEN_POSE_CONFIGS       # supine, prone, side_left, side_right
] + [
    {**cfg, "type": "bent"}   for cfg in BENT_POSE_CONFIGS          # home, knees_bent, squat, deep_squat
]

# Explicit alias used by the env cfg for clarity.
reset_to_any_fallen_pose = reset_to_fallen_or_bent_pose
