"""Recovery-v1: full floor-recovery task configuration.

Self-contained — does NOT inherit from the transition chain (transition_v3,
transition_v2, or transition). Every term is defined explicitly here so the
full configuration is readable without tracing through multiple base classes.

Why a clean rewrite?
  The transition chain inherited njmax=300 (MuJoCo constraint equations per
  world). G1 in complex floor contact (14 foot geoms + arm geoms + ground)
  uses up to 454 constraints — causing "nefc overflow" which corrupts physics
  and produces NaN observations that crash training. Setting njmax=600 fixes
  the root cause. The full rewrite also removes all the del / weight-override
  boilerplate accumulated across five training iterations.

Task goal
  Stand up from any fallen ground position (supine, prone, side-lying,
  sitting, or partially tucked). Eleven initial-state templates cover the
  full trajectory so every sub-skill gets gradient:
    Fallen (36 %):     supine, prone, side_left, side_right  (base_z ~0.25 m)
    Sitting (18 %):    sitting_low (40° lean), sitting_high (30° lean)
    Squat-lean (9 %):  squat_lean (20° lean, knee=1.2 rad)  (transition zone)
    Bent (36 %):       home, knees_bent, squat, deep_squat   (FK-verified)

Reward design — four root causes of previous failure addressed
──────────────────────────────────────────────────────────────
Root cause 1 — velocity-based elbow_push_from_ground (REMOVED):
  Rewarded max(0, elbow_vel_z) × arm_contact every step.  The robot discovers
  that bouncing elbows on the ground earns reward on every upward half-cycle
  while in contact.  This causes the observed "bouncing in initial state" and
  prevents real push-up learning.  Replaced by pushup_support_reward (position-
  based Gaussian on elbow height).

Root cause 2 — ungated body_ang_vel penalty (REPLACED):
  body_angular_velocity_penalty had no height gate.  Rolling from supine to
  prone requires ~1.5 rad/s XY angular velocity.  With weight -0.03 and the
  GAE discount horizon of ~17 steps, the immediate angular penalty (-0.068/step)
  outweighed the discounted orientation gain.  Net discounted advantage for
  rolling was negative → policy learned NOT to roll.  Replaced by
  height_gated_ang_vel_penalty which is zero below 0.40 m and ramps to full
  above 0.65 m, allowing rolling while preventing elevated instability.

Root cause 3 — GAE λ=0.95 gives 0.34-second effective horizon (FIXED IN rl_cfg):
  Floor recovery takes 5–15 seconds.  With λ=0.95, γ=0.99 the effective
  GAE horizon = 1/(1-γλ) ≈ 17 steps = 0.34 s.  A roll that pays off over
  50 steps (1 s) has only 4.7 % of its terminal reward credited to the first
  action.  Increasing λ to 0.97 doubles the horizon to 0.67 s, making multi-
  second recovery sequences profitable from the first action.  See rl_cfg.py.

Root cause 4 — 44 % upright starts collapse action std (FIXED — now 36 %):
  PPO uses a scalar std.  Upright balance needs std ≈ 0.1; fallen recovery
  needs std ≈ 1.5.  44 % upright starts pulled std toward the balance level,
  making random actions from fallen states too small to initiate rolling or
  push-ups.  Fallen + sitting now cover 54 % of starts; upright bent is 36 %.

Reward design summary
  Phase 1 — get off the floor:
    orientation_recovery (+3.0): pelvis gravity projection toward upright
    height_recovery (+2.0):      pelvis rising toward 0.78 m
    torso_height_reward (+3.0):  CHEST (not pelvis) rising — breaks leg-bridge
    arm_reach_down (+1.5):       hands toward floor when flat (pre-contact guidance)
    pushup_support_reward (+4.0): elbow sustained at push-up height while in
                                   contact (position-based, not farmable by bounce)
  Phase 2 — sit-to-stand:
    shank_orientation_reward (+3.5): shanks vertical (not forward-extended)
    head_above_feet_reward (+2.5):   head 1.15 m above feet (relative, not abs)
    feet_proximity_reward (+2.0):    feet under pelvis XY
  Phase 3 — hold standing:
    pose_convergence_gated (+1.5):   joints toward default, gated by upright
    hold_bonus (+1.0):               locked-in bonus
    both_feet_contact (+0.2):        both feet on ground

  Penalties (reduced for floor recovery, phase-gated where necessary):
    height_gated_ang_vel   (-0.05, gated 0.40–0.65 m): zero on floor, full elevated
    angular_momentum       (-0.002, reduced from -0.008): reduced to allow floor roll
    joint_vel_penalty      (-0.005, was -0.002): restored to damp explosive standing
    joint_acc_l2           (-5e-8,  reduced from -2.5e-7)
    action_rate_l2         (-0.012, was -0.005): restored to prevent NaN-causing contact spikes

NaN crash + jump-hack fix
─────────────────────────
Iter ~500 crash: MuJoCo contact forces spiked on hard landings (PGS iterations=10
insufficient). Robot learned to jump (brief +60 reward) before termination (−50),
making jumping profitable (net +10). Ground clipping from fallen_joint_perturbation=0.6
caused "below-ground" artifact at episode start (arms clipped into floor → large
corrective impulse → robot flung upward).

Fixes applied:
  iterations=30, ls_iterations=40: better PGS convergence on hard landings.
  joint_vel_penalty -0.002 → -0.005: damp explosive joint motion.
  action_rate_l2 -0.005 → -0.012: penalise large inter-step action jumps.
  joint_vel_overflow at 100 rad/s: reset exploding episodes before NaN crash.
    (was 50 rad/s — too sensitive to vigorous leg-kick; raised to 100.)
  nconmax=600 explicit (was None): prevents silent contact overflow.
  torso_height_reward 3.0 → 1.0: jump net = (1.0+3.0)×10 − 50 = −10 (unprofitable).
  body_orientation_l2 -3.0 → -1.0: flat-state net was −1.89/step → +0.11/step.
    Eliminates constant pressure to aggressively escape flat state.
  fallen_joint_perturbation 0.6 → 0.4: eliminates arm-clip-into-floor jump artifact.

Key design constraints preserved (do not revert):
  - All position-based (Gaussian) except penalties. NO velocity-based rewards —
    they are gameable by oscillation.
  - height_gated_ang_vel zero below 0.40 m: allows rolling/flipping on the floor.
  - fallen_joint_perturbation=0.25: reduced from 0.6→0.4→0.25 to eliminate arm
    clipping in prone/side-lying.  Push-up discovery now relies on arm_reach_down
    + pushup_support_reward gradient rather than starting near the push-up position.
  - shank_orientation_reward std=0.50: at 0.30 the sitting cosine range gives
    near-zero reward → PPO can't see the gradient.
  - feet_proximity_reward std=0.45: meaningful gradient from knee-tuck start.
  - height-gated termination at 0.65 m: prone→bridge trajectory crosses 0.50 m
    while tilted; 0.50 m created a zone that punished rising past it.
  - fallen_lin_vel_range=0.05, fallen_ang_vel_range=0.10, fallen_joint_vel_range=0.05:
    near-zero initial velocities for fallen states prevent immediate floor
    bouncing from large velocity perturbations colliding with ground contact.
"""

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

import src.tasks.recovery_v1.mdp as mdp
from src.tasks.recovery_v1.mdp.events import ALL_POSE_CONFIGS


def make_recovery_v1_env_cfg() -> ManagerBasedRlEnvCfg:
  """Create the recovery-v1 task configuration.

  Self-contained: every observation, action, event, reward, and termination
  term is defined here. No inheritance from the transition chain.
  """

  # ── Observations ─────────────────────────────────────────────────────────────
  # Actor and critic share the same core terms; critic also receives privileged
  # state (linear velocity, foot contact forces) that is unavailable on hardware.
  #
  # Critical actor additions for floor recovery:
  #   base_height — pelvis height above terrain. Without it, a robot at 0.40 m
  #     (push-up phase) and one at 0.70 m (near-standing) may have similar
  #     projected gravity and joint angles but require completely different
  #     actions. The actor is blind to the phase without explicit height.
  #   foot_contact — which feet are on the ground. Essential for knowing whether
  #     to plant feet (standing) or tuck them (recovery). Previously only in
  #     critic, leaving the actor without phase-detection context.
  #   arm_contact — which arms are in ground contact. Tells the actor whether
  #     push-up support is available. Gates correct arm timing for recovery.
  actor_terms = {
    "base_ang_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_ang_vel"},
      noise=Unoise(n_min=-0.2, n_max=0.2),
    ),
    "projected_gravity": ObservationTermCfg(
      func=mdp.projected_gravity,
      noise=Unoise(n_min=-0.05, n_max=0.05),
    ),
    "base_height": ObservationTermCfg(
      func=mdp.base_height_obs,
      noise=Unoise(n_min=-0.02, n_max=0.02),
    ),
    "joint_pos": ObservationTermCfg(
      func=mdp.joint_pos_rel,
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    "joint_vel": ObservationTermCfg(
      func=mdp.joint_vel_rel,
      noise=Unoise(n_min=-1.5, n_max=1.5),
    ),
    "foot_contact": ObservationTermCfg(
      func=mdp.foot_contact,
      params={"sensor_name": "feet_ground_contact"},  # wired per-robot in G1 config
    ),
    "arm_contact": ObservationTermCfg(
      func=mdp.foot_contact,  # same pattern as foot_contact — reads sensor.data.found
      params={"sensor_name": "arm_ground_contact"},   # wired per-robot in G1 config
    ),
    "actions": ObservationTermCfg(func=mdp.last_action),
  }

  critic_terms = {
    **actor_terms,
    "base_lin_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_lin_vel"},
      noise=Unoise(n_min=-0.5, n_max=0.5),
    ),
    "foot_contact_forces": ObservationTermCfg(
      func=mdp.foot_contact_forces,
      params={"sensor_name": "feet_ground_contact"},
    ),
  }

  # ── Actions ───────────────────────────────────────────────────────────────────
  actions: dict[str, ActionTermCfg] = {
    "joint_pos": JointPositionActionCfg(
      entity_name="robot",
      actuator_names=(".*",),
      scale=0.25,             # overridden per-robot (G1_ACTION_SCALE in G1 config)
      use_default_offset=True,
    )
  }

  # ── Events ────────────────────────────────────────────────────────────────────
  events = {
    # ── Reset: 11-template curriculum
    #    36 % fallen (supine, prone, side_left, side_right)
    #    18 % sitting (sitting_low 40°, sitting_high 30°)
    #     9 % squat-lean (squat_lean 20° lean, knee=1.2 rad)
    #    36 % bent-upright (home, knees_bent, squat, deep_squat)
    #
    # Side-lying added (was deferred): G1 often falls to its side; without
    # side-lying starts the policy never learns to roll from that position.
    #
    # Fallen fraction increased 22 % → 36 %: more fallen starts give the policy
    # more direct floor-recovery experience and reduce the std-bias toward the
    # balance skill that dominated at 44 % upright.
    "reset_robot": EventTermCfg(
      func=mdp.reset_to_fallen_or_bent_pose,
      mode="reset",
      params={
        "all_pose_configs": ALL_POSE_CONFIGS,
        "xy_pos_range": 0.5,
        "yaw_range": math.pi,
        "fallen_joint_perturbation": 0.25, # arm/waist joints: 0.25 rad prevents shoulder+elbow from clipping floor in prone/side-lying
        "fallen_leg_perturbation":   0.05, # leg joints only: 0.05 rad prevents ankle+hip clipping in sitting templates
        "leg_perturbation": 0.05,  # was 0.10; deep_squat ankle=-0.60→-0.70 at 0.10 clips floor
        "other_perturbation": 0.35,
        # Fallen-state-specific initial velocities (small to prevent floor bounce).
        # Large initial velocities cause the body to thrash against ground contact
        # geometry on the first few steps, generating chaotic forces that look like
        # reward-positive bouncing to the policy.
        "fallen_lin_vel_range":   0.05,   # m/s  (was implicitly lin_vel_range=0.20)
        "fallen_ang_vel_range":   0.10,   # rad/s (was implicitly ang_vel_range=0.30)
        "fallen_joint_vel_range": 0.05,   # rad/s (was implicitly joint_vel_range=0.15)
        # Bent/upright-state velocities (unchanged from v1 — robot is already stable).
        "joint_vel_range": 0.15,
        "lin_vel_range":   0.20,
        "ang_vel_range":   0.30,
        "knee_cfg":      SceneEntityCfg("robot", joint_names=(".*_knee_joint",)),
        "hip_pitch_cfg": SceneEntityCfg("robot", joint_names=(".*_hip_pitch_joint",)),
        "ankle_cfg":     SceneEntityCfg("robot", joint_names=(".*_ankle_pitch_joint",)),
        "asset_cfg":     SceneEntityCfg("robot"),
      },
    ),
    "push_robot": EventTermCfg(
      func=mdp.push_by_setting_velocity,
      mode="interval",
      interval_range_s=(8.0, 10.0),
      params={
        "velocity_range": {
          "x": (-0.3, 0.3),
          "y": (-0.3, 0.3),
          "z": (-0.2, 0.2),
          "roll":  (-0.3, 0.3),
          "pitch": (-0.3, 0.3),
          "yaw":   (-0.5, 0.5),
        },
      },
    ),
    "foot_friction": EventTermCfg(
      mode="startup",
      func=dr.geom_friction,
      params={
        "asset_cfg": SceneEntityCfg("robot", geom_names=()),
        "operation": "abs",
        "ranges": (0.3, 1.6),
        "shared_random": True,
      },
    ),
    "encoder_bias": EventTermCfg(
      mode="startup",
      func=dr.encoder_bias,
      params={
        "asset_cfg": SceneEntityCfg("robot"),
        "bias_range": (-0.015, 0.015),
      },
    ),
    "base_com": EventTermCfg(
      mode="startup",
      func=dr.body_com_offset,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=()),
        "operation": "add",
        "ranges": {0: (-0.05, 0.05), 1: (-0.05, 0.05), 2: (-0.05, 0.05)},
      },
    ),
  }

  # ── Rewards ───────────────────────────────────────────────────────────────────
  rewards = {

    # ── Phase 1: get off the floor ────────────────────────────────────────────

    "orientation_recovery": RewardTermCfg(
      func=mdp.orientation_recovery,
      weight=3.0,
      params={
        "std": 1.0,
        "asset_cfg": SceneEntityCfg("robot"),
      },
    ),
    "height_recovery": RewardTermCfg(
      func=mdp.height_recovery,
      weight=2.0,
      params={
        "target_height": 0.78,
        "std": 0.65,
        "asset_cfg": SceneEntityCfg("robot"),
      },
    ),
    # torso_height_reward: weight restored 1.0 → 3.0.
    # Was reduced to 1.0 to make jumping unprofitable, but this also removes 2/3
    # of the "get tall" learning signal.  With root_lin_vel_penalty=-0.3 now
    # handling explosive jumps (push-off at 5+ m/s earns −37.5 over 5 steps),
    # the full weight 3.0 is safe.  Jump profitability math with root_lin_vel_penalty:
    #   Phase rewards (14/step × 10 steps) = +140
    #   root_lin_vel_penalty (5 m/s push-off 5 steps + 8 m/s airborne 5 steps) = −133.5
    #   airborne_penalty (−10 × 5 steps) = −50
    #   is_terminated = −50
    #   Net = 140 − 133.5 − 50 − 50 = −93.5  →  NOT profitable.
    # Legitimate recovery (200+ steps at torso=0.50 m): 3.0 × 0.527 × 200 = +316 → essential!
    "torso_height_reward": RewardTermCfg(
      func=mdp.torso_height_reward,
      weight=3.0,  # restored from 1.0; root_lin_vel_penalty now prevents jump hacking
      params={
        "target_height": 0.90,
        "std": 0.50,
        "asset_cfg": SceneEntityCfg("robot", body_names=()),  # set per-robot
      },
    ),
    # arm_reach_down: pull hands toward floor level when robot is flat.
    # Provides gradient BEFORE arm-ground contact so the arm motion is
    # discoverable without already knowing how to push up.
    "arm_reach_down": RewardTermCfg(
      func=mdp.arm_reach_down,
      weight=1.5,
      params={
        "height_gate": 0.60,
        "flat_gate_threshold": -0.85,
        "asset_cfg": SceneEntityCfg("robot", body_names=()),  # set per-robot
      },
    ),
    # pushup_support_reward: position-based replacement for the removed
    # elbow_push_from_ground (velocity-based, caused bouncing).
    # Rewards elbow being at push-up height (0.35 m) WHILE arm is in contact.
    # Cannot be farmed by bouncing: oscillating elbow earns 0.21 vs 1.0 for
    # sustained push-up position. The robot learns to HOLD the push-up, not
    # to bounce. Weight 4.0 (was 3.5 for elbow_push_from_ground — slightly
    # higher because this reward is harder to achieve by accident).
    # pushup_support_reward: height_gate lowered 0.65 → 0.42 m.
    # At 0.65 m the reward fires throughout the all-fours phase (pelvis 0.42–0.50 m),
    # creating a local optimum: the robot earns +3.1/step from pushup_support in
    # all-fours, making the transition to kneeling (arms lift, immediate −4.0 loss)
    # look unprofitable over the GAE horizon (~33 steps).
    # At 0.42 m the reward is zero once the robot rises above the push-up zone.
    # Phase boundaries after the fix:
    #   Push-up  (pelvis 0.30–0.42 m): reward fires → push-up learning ✓
    #   All-fours (pelvis > 0.42 m):   reward = 0   → no incentive to keep arms down ✓
    #   Kneeling  (pelvis ~0.55 m):    +12.73/step > all-fours +11.68 → kneeling better ✓
    #   Standing  (pelvis ~0.78 m):    +18.2/step  → best → natural endpoint ✓
    "pushup_support_reward": RewardTermCfg(
      func=mdp.pushup_support_reward,
      weight=4.0,
      params={
        "sensor_name": "arm_ground_contact",  # set per-robot in G1 config
        "target_height": 0.35,   # m — G1 elbow in push-up position
        "std": 0.20,             # m — gradient from floor (0.10 m) to push-up (0.35 m)
        "height_gate": 0.42,     # m — was 0.65; suppress above push-up zone, prevents all-fours optimum
        "flat_gate_threshold": -0.70,  # active until 46° from upright
        "asset_cfg": SceneEntityCfg("robot", body_names=()),  # set per-robot (elbow bodies)
      },
    ),

    # ── Phase 2: sit-to-stand ─────────────────────────────────────────────────

    # shank_orientation_reward: gate lowered 0.30 → 0.25 m.
    # At 0.30 m the margin above push-up pelvis height (~0.30–0.32 m) is only
    # 0–20 mm — the reward may fire intermittently or not at all depending on
    # the exact push-up configuration.  At 0.25 m it fires reliably once the
    # robot is in any raised-arm pose, giving a clear gradient toward vertical
    # shanks (knees tucked under body) throughout Phase 1 and Phase 2.
    "shank_orientation_reward": RewardTermCfg(
      func=mdp.shank_orientation_reward,
      weight=3.5,
      params={
        "height_gate": 0.25,  # was 0.30; must fire reliably during push-up phase
        "std": 0.50,
        "knee_asset_cfg":  SceneEntityCfg("robot", body_names=()),  # set per-robot
        "ankle_asset_cfg": SceneEntityCfg("robot", body_names=()),  # set per-robot
      },
    ),
    "head_above_feet_reward": RewardTermCfg(
      func=mdp.head_above_feet_reward,
      weight=2.5,
      params={
        "target_height": 1.15,
        "std": 0.25,
        "head_offset": 0.43,
        "torso_asset_cfg": SceneEntityCfg("robot", body_names=()),  # set per-robot
        "foot_asset_cfg":  SceneEntityCfg("robot", body_names=()),  # set per-robot
      },
    ),
    # feet_proximity_reward: gate lowered 0.35 → 0.28 m.
    # G1 pelvis in extended push-up position is ~0.30–0.32 m.  At gate=0.35 m
    # this reward is INACTIVE during push-up, so the policy has no gradient to
    # pull its feet toward its body.  Result: push-up local optimum — robot
    # gets full pushup_support_reward with zero incentive to advance to Phase 2.
    # At 0.28 m the reward fires from push-up height onward, providing the
    # "tuck feet under pelvis" gradient immediately after Phase 1 succeeds.
    "feet_proximity_reward": RewardTermCfg(
      func=mdp.feet_proximity_reward,
      weight=2.0,
      params={
        "height_gate": 0.28,  # was 0.35; must be below push-up pelvis height (~0.30m)
        "std": 0.45,
        "asset_cfg": SceneEntityCfg("robot", body_names=()),  # set per-robot
      },
    ),

    # ── Phase 3: hold standing ────────────────────────────────────────────────

    "pose_convergence_gated": RewardTermCfg(
      func=mdp.pose_convergence_gated,
      weight=1.5,
      params={
        "std": 0.5,
        "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
      },
    ),
    "hold_bonus": RewardTermCfg(
      func=mdp.hold_bonus,
      weight=1.0,
      params={
        "pose_threshold":    0.08,
        "ang_vel_threshold": 0.15,
        "lin_vel_threshold": 0.10,
        "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
      },
    ),
    "both_feet_contact": RewardTermCfg(
      func=mdp.both_feet_contact,
      weight=0.2,
      params={"sensor_name": "feet_ground_contact"},
    ),

    # ── Penalties ─────────────────────────────────────────────────────────────

    # body_orientation_l2: weight −3.0 → −1.0.
    # Computes sum(proj_gravity_XY²): 0 when upright, 1 when flat, 0 when INVERTED.
    # The zero-when-inverted blind spot is intentional: orientation_recovery (+3.0)
    # already distinguishes upright (reward=3.0) from inverted (reward=0.055) with
    # a large gradient, so an extra penalty is not needed for that case.
    # At −3.0, the flat-state L2 penalty is ≈ −3.0/step when lying (max error).
    # Combined with orientation_recovery (+1.11 when flat), the net per-step is
    # −1.89 for the entire lying phase. This creates constant pressure to escape
    # flat state aggressively → driving the explosive jump behavior.
    # At −1.0 the net is +0.11/step when flat (neutral), 0 penalty when upright.
    "body_orientation_l2": RewardTermCfg(
      func=mdp.body_orientation_l2,
      weight=-1.0,  # was -3.0; −3.0 makes lying state net-negative, driving aggressive jump escape
      params={"asset_cfg": SceneEntityCfg("robot", body_names=())},  # set per-robot
    ),
    # airborne_penalty: direct anti-jump-hacking reward.
    # min_height lowered 0.40 → 0.20 m: the original 0.40 m gate was too high.
    # Prone/supine robots at base_z=0.35 m are already ABOVE 0.40 m when launched
    # by a physics explosion — the penalty would not fire for the first ~3 steps
    # while the robot rises from 0.35→0.40 m, providing a brief positive-reward
    # window that reinforces the launch behaviour.
    # At 0.20 m: any robot above push-up floor level (~0.15-0.20 m) with no
    # arm/foot contact is immediately penalised.  The 3–5 step settling phase
    # (robot drops from 0.35 m to floor) fires airborne_penalty briefly, but
    # this incentivises the policy to reach for floor contact quickly — correct.
    # Combined with root_lin_vel_penalty, jumping is fully unprofitable (see
    # torso_height_reward comment above for the full net-reward calculation).
    "airborne_penalty": RewardTermCfg(
      func=mdp.airborne_penalty,
      weight=-10.0,
      params={
        "min_height": 0.20,             # m — lowered from 0.40; catches launches from prone base_z=0.35 m
        "foot_sensor_name": "feet_ground_contact",
        "arm_sensor_name":  "arm_ground_contact",
        "asset_cfg": SceneEntityCfg("robot"),
      },
    ),
    # root_lin_vel_penalty: penalises explosive whole-body velocity.
    # The airborne_penalty fires AFTER feet leave the floor, but a jump is already
    # decided during the ground-contact push-off phase (legs extend, body accelerates
    # upward BEFORE feet lift).  root_lin_vel_penalty fires during the push-off:
    #   slow recovery (0.5 m/s): −0.3 × 0.25 = −0.075/step (negligible)
    #   explosive push-off (5 m/s): −0.3 × 25 = −7.5/step × 5 steps = −37.5
    # This makes the push-off phase itself unprofitable, closing the loophole
    # that airborne_penalty leaves open.  See torso_height_reward comment for
    # the complete net-reward calculation.
    "root_lin_vel_penalty": RewardTermCfg(
      func=mdp.root_lin_vel_penalty,
      weight=-0.3,
      params={"asset_cfg": SceneEntityCfg("robot")},
    ),
    # height_gated_ang_vel: replaces body_angular_velocity_penalty (ungated).
    # Zero below 0.40 m (floor recovery phase) → rolling/flipping are allowed.
    # Ramps to full penalty above 0.65 m → prevents instability when elevated.
    # Weight -0.05 (slightly higher than old -0.03 because it only fires when
    # elevated, so per-step average penalty over an episode is similar).
    "height_gated_ang_vel": RewardTermCfg(
      func=mdp.height_gated_ang_vel_penalty,
      weight=-0.05,
      params={
        "gate_min_height": 0.40,  # m — penalty zero below this (floor phase)
        "gate_max_height": 0.65,  # m — full penalty above this (standing phase)
        "asset_cfg": SceneEntityCfg("robot", body_names=()),  # set per-robot (torso)
      },
    ),
    # angular_momentum: reduced weight -0.002 (was -0.008).
    # The body rolling and arm swinging needed for floor recovery generates
    # large whole-body angular momentum. -0.008 over-penalised this.
    "angular_momentum": RewardTermCfg(
      func=mdp.angular_momentum_penalty,
      weight=-0.002,
      params={"sensor_name": "robot/root_angmom"},
    ),
    # joint_vel_penalty: -0.005 (was -0.002, was -0.008).
    # -0.002 was too weak: with 23 joints at 4 rad/s each, the penalty was
    # negligible (~-0.7/step) against the +8 orientation+height rewards.
    # Explosive joint velocities during jumping/landing cause MuJoCo contact
    # forces to overflow → NaN in kinematics → training crash. -0.005 damps
    # explosive motions while still permitting vigorous push-up / leg-swing.
    "joint_vel_penalty": RewardTermCfg(
      func=mdp.joint_vel_penalty,
      weight=-0.005,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
    ),
    # joint_acc_l2: kept at -5e-8 (was -2.5e-7).
    "joint_acc_l2": RewardTermCfg(func=mdp.joint_acc_l2, weight=-5e-8),
    "joint_pos_limits": RewardTermCfg(func=mdp.joint_pos_limits, weight=-10.0),
    # action_rate_l2: -0.012 (was -0.005, was -0.02).
    # -0.005 allowed action deltas that, compounded over 4-step decimation,
    # generate joint accelerations large enough to spike contact forces.
    # -0.012 damps inter-step action jumps to prevent the explosive
    # contact that caused the NaN crash at iteration ~500.
    "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.012),
    # is_terminated: reduced -200 → -50.
    # At lam=0.97 a termination at step 20 attributes -89 to the first action
    # (vs -60 at lam=0.95). With 27 % of bent starts above the 0.65 m threshold,
    # frequent tipping-from-perturbation events cause the -200 penalty to
    # dominate and push the policy into a "don't do anything" collapse spiral.
    # The episode ending already punishes implicitly (no more positive rewards).
    # -50 still provides a strong deterrent without overwhelming the ~22/step
    # maximum positive reward budget.
    "is_terminated": RewardTermCfg(func=mdp.is_terminated, weight=-50.0),
  }

  # ── Terminations ───────────────────────────────────────────────────────────────
  terminations = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    # fell_over: grace_period_steps=20 (= 0.4 s at 50 Hz).
    # Bent starts (home=0.80m, knees_bent=0.77m, squat=0.69m) begin above the
    # 0.65m threshold. The reset applies up to 0.30 rad/s angular perturbation
    # which can tip the robot before the policy acts. Without a grace period the
    # -50 termination penalty is attributed by GAE (lam=0.97) directly to the
    # first policy action, even though the policy had no control over it.
    "fell_over": TerminationTermCfg(
      func=mdp.bad_orientation_while_elevated,
      params={
        "limit_angle": math.radians(75.0),
        "height_threshold": 0.65,
        "grace_period_steps": 20,   # 0.4 s — policy gets time to respond before termination fires
      },
    ),
    # joint_vel_overflow: physics explosion safety net (joint velocity + root velocity).
    # Two detection paths:
    # 1. Joint velocity >100 rad/s: MuJoCo constraint overflow manifesting as joint
    #    velocity divergence.  100 rad/s allows aggressive leg-kicks (~15-20 rad/s)
    #    while catching constraint overflow (>500+ rad/s, usually NaN).
    # 2. Root link linear velocity >15 m/s (NEW): rigid-body launches where all
    #    joints move together (relative joint velocities stay low) but the entire
    #    robot is flung upward by a ground-clipping corrective impulse.  Normal
    #    recovery: max ~4 m/s (free-fall from 0.8 m standing height).  Physics
    #    explosion from arm clipping: 10–30 m/s.  15 m/s is a safe threshold.
    "joint_vel_overflow": TerminationTermCfg(
      func=mdp.joint_velocity_overflow,
      params={
        "threshold": 100.0,           # rad/s — joint velocity overflow
        "root_vel_threshold": 15.0,   # m/s — rigid-body launch from ground clipping
        "asset_cfg": SceneEntityCfg("robot"),
      },
    ),
  }

  # ── Assemble ──────────────────────────────────────────────────────────────────
  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane"),
      num_envs=1,
      extent=2.0,
    ),
    observations={
      "actor": ObservationGroupCfg(
        terms=actor_terms,
        concatenate_terms=True,
        enable_corruption=True,
        history_length=1,
      ),
      "critic": ObservationGroupCfg(
        terms=critic_terms,
        concatenate_terms=True,
        enable_corruption=False,
        history_length=1,
      ),
    },
    actions=actions,
    commands={},
    events=events,
    rewards=rewards,
    terminations=terminations,
    curriculum={},
    metrics={},
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="",   # set per-robot in G1 config
      distance=3.0,
      elevation=-5.0,
      azimuth=90.0,
    ),
    decimation=4,
    sim=SimulationCfg(
      nconmax=600,  # explicit; None may map to a compiled default too small for G1's 14 foot geoms + arm geoms
      njmax=600,
      mujoco=MujocoCfg(
        timestep=0.005,
        iterations=30,       # was 10→20→30; more PGS iterations = better contact resolution on hard landings
        ls_iterations=40,    # was 20→30→40; more line-search iterations = better convergence
        ccd_iterations=50,
      ),
    ),
    episode_length_s=35.0,
  )
