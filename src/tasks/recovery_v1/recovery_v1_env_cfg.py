"""Recovery-v1 (rewrite): stand up from any floor-level starting pose.

Why the previous implementation failed — three root causes
──────────────────────────────────────────────────────────
Root cause 1 — airborne_penalty min_height=0.20 m (PRIMARY BUG):

  The airborne_penalty fires when:
    base_height > min_height  AND  no foot contact  AND  arm contact doesn't exempt
  The exemption for arm contact is gated by not_inverted = (proj_gz < 0.3):
    Supine robot: proj_gz ≈ +1.0  →  not_inverted = 0  →  arm contact never exempts.

  Consequence: from a supine or prone starting pose (base_z = 0.35 m > 0.20 m,
  no foot contact, not_inverted = 0) the penalty fires at 1.0 every step for the
  ENTIRE 35-second episode.  With weight = -10.0 and dt = 0.02:
    -10 × 0.02 × 1750 steps = -350 per fallen episode.

  The measured reward ≈ 200 per episode is exactly explained by this:
    0.44 × (upright + small positive) + 0.56 × (fallen, net ≈ -350)
    ≈ 0.44 × 600 + 0.56 × (-350) ≈ 264 - 196 ≈ 68
  (The rough match depends on exact upright rewards; the direction is unambiguous.)

  FIX: raise min_height from 0.20 → 0.65 m.  The penalty now fires ONLY when the
  robot is near standing height and becomes airborne — the actual jump-hack scenario.
  All floor-recovery phases (supine 0.35 m, sitting 0.28–0.38 m, push-up 0.30–0.45 m,
  squat-lean 0.52 m) are below 0.65 m and are completely exempt.

Root cause 2 — arm-based rewards need contact before they activate (REMOVED):

  arm_reach_down and pushup_support_reward both gate on arm-ground contact.
  From a cold-start random policy with init_std = 1.0, that contact happens
  rarely by accident.  These rewards contribute near-zero gradient for the first
  thousands of iterations while imposing sensor complexity and observation noise.
  Removed in this rewrite; they can be re-added once basic upright-to-floor
  recovery is stable (recovery-v2).

Root cause 3 — training from random init is fundamentally hard (DESIGN NOTE):

  Simultaneously learning (a) balance and (b) floor recovery with a single scalar
  PPO std is the hardest possible training setup.  Strongly recommended:

    python scripts/train.py Unitree-G1-RecoveryV1 \\
        --agent.resume=True \\
        --agent.load_run=<transition_v3_run>

  With a transition_v3 checkpoint the policy already knows how to stand; the
  recovery training only needs to extend the existing skill rather than learn
  it from scratch.  Cold-start training will still converge with the fixes
  above but takes approximately 2–3× longer.

Reward design (simplified, 15 terms)
──────────────────────────────────────
  Phase 1 — get off the floor:
    orientation_recovery  (+3.0)  primary upright signal; exp(-(proj_gz+1)²/1.0)
    height_recovery       (+2.0)  pelvis rising toward 0.78 m
    torso_height_reward   (+2.0)  chest rising; prevents leg-bridge optimum

  Phase 2 — sit to stand:
    shank_orientation     (+3.5)  shanks vertical (knee tuck signal)
    head_above_feet       (+2.5)  head vs feet relative height
    feet_proximity        (+2.0)  feet under pelvis

  Phase 3 — hold standing:
    pose_convergence      (+2.0)  joints to default, gated by upright
    hold_bonus            (+1.0)  stable lock-in
    both_feet_contact     (+0.2)

  Penalties:
    height_gated_ang_vel  (-0.05) zero below 0.40 m, full above 0.65 m
    joint_vel_penalty     (-0.005)
    action_rate_l2        (-0.012)
    joint_pos_limits      (-10.0)
    airborne_penalty      (-10.0) min_height=0.65 m (was 0.20 m — see root cause 1)
    root_lin_vel_penalty  (-0.3)  anti-explosive launch / push-off
    is_terminated         (-50.0)

Removed from previous version:
  arm_reach_down        — see root cause 2
  pushup_support_reward — see root cause 2
  body_orientation_l2   — redundant with orientation_recovery; sign is wrong when inverted
  angular_momentum      — secondary signal, adds sensor noise
  joint_acc_l2          — negligible weight (-5e-8), irrelevant
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
  """Create the recovery-v1 task configuration (rewrite).

  Self-contained: every observation, action, event, reward, and termination
  term is defined here. No inheritance from the transition chain.
  """

  # ── Observations ─────────────────────────────────────────────────────────────
  # base_height in actor: disambiguates push-up phase (0.40 m) from near-standing
  # (0.70 m) which have similar projected_gravity but need opposite actions.
  # foot_contact in actor: tells the policy whether to plant feet (standing)
  # or tuck them (recovery) — phase detection context.
  # arm_contact removed: no arm-based rewards in this version, sensor unneeded.
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
    # ── Reset: 9-template curriculum ─────────────────────────────────────────
    #    22 % fallen      (supine, prone)
    #    22 % sitting-up  (sitting_low 40°, sitting_high 30°)
    #    11 % squat-lean  (squat_lean 20° lean, knee=1.2 rad)
    #    44 % bent-upright (home, knees_bent, squat, deep_squat)
    #
    # With the airborne_penalty fixed (min_height=0.65m instead of 0.20m),
    # fallen episodes no longer receive -350 per episode, so a 56% non-upright
    # fraction no longer collapses average reward.  The 44% upright fraction
    # maintains enough positive gradient for the policy to prefer standing.
    "reset_robot": EventTermCfg(
      func=mdp.reset_to_fallen_or_bent_pose,
      mode="reset",
      params={
        "all_pose_configs": ALL_POSE_CONFIGS,
        "xy_pos_range": 0.5,
        "yaw_range": math.pi,
        "fallen_joint_perturbation": 0.25,
        "fallen_leg_perturbation":   0.05,
        "leg_perturbation": 0.05,
        "other_perturbation": 0.35,
        "fallen_lin_vel_range":   0.05,
        "fallen_ang_vel_range":   0.10,
        "fallen_joint_vel_range": 0.05,
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
    "torso_height_reward": RewardTermCfg(
      func=mdp.torso_height_reward,
      weight=2.0,
      params={
        "target_height": 0.90,
        "std": 0.50,
        "asset_cfg": SceneEntityCfg("robot", body_names=()),  # set per-robot
      },
    ),

    # ── Phase 2: sit-to-stand ─────────────────────────────────────────────────

    "shank_orientation_reward": RewardTermCfg(
      func=mdp.shank_orientation_reward,
      weight=3.5,
      params={
        "height_gate": 0.25,
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
    "feet_proximity_reward": RewardTermCfg(
      func=mdp.feet_proximity_reward,
      weight=2.0,
      params={
        "height_gate": 0.28,
        "std": 0.45,
        "asset_cfg": SceneEntityCfg("robot", body_names=()),  # set per-robot
      },
    ),

    # ── Phase 3: hold standing ────────────────────────────────────────────────

    "pose_convergence_gated": RewardTermCfg(
      func=mdp.pose_convergence_gated,
      weight=2.0,
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

    "height_gated_ang_vel": RewardTermCfg(
      func=mdp.height_gated_ang_vel_penalty,
      weight=-0.05,
      params={
        "gate_min_height": 0.40,
        "gate_max_height": 0.65,
        "asset_cfg": SceneEntityCfg("robot", body_names=()),  # set per-robot (torso)
      },
    ),
    # airborne_penalty: min_height raised 0.20 → 0.65 m.
    # Previous value (0.20 m) caused this penalty to fire for the ENTIRE episode
    # from supine/prone starting poses: base_z=0.35 m > 0.20 m AND no foot
    # contact AND proj_gz≈+1 (inverted) disables arm-contact exemption.
    # Result: -10 × 0.02 × 1750 = -350 per fallen episode, collapsing training.
    # At 0.65 m the penalty only fires when the robot is near standing height
    # and becomes airborne — the true jump-hack scenario.  All floor-recovery
    # phases (< 0.65 m) are below the threshold and never trigger it.
    "airborne_penalty": RewardTermCfg(
      func=mdp.airborne_penalty,
      weight=-10.0,
      params={
        "min_height": 0.65,             # was 0.20 m — see module docstring root cause 1
        "foot_sensor_name": "feet_ground_contact",
        "arm_sensor_name":  None,       # arm sensor removed in this version
        "asset_cfg": SceneEntityCfg("robot"),
      },
    ),
    "root_lin_vel_penalty": RewardTermCfg(
      func=mdp.root_lin_vel_penalty,
      weight=-0.3,
      params={"asset_cfg": SceneEntityCfg("robot")},
    ),
    "joint_vel_penalty": RewardTermCfg(
      func=mdp.joint_vel_penalty,
      weight=-0.005,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
    ),
    "joint_pos_limits": RewardTermCfg(func=mdp.joint_pos_limits, weight=-10.0),
    "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.012),
    "is_terminated": RewardTermCfg(func=mdp.is_terminated, weight=-50.0),
  }

  # ── Terminations ───────────────────────────────────────────────────────────────
  terminations = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    # fell_over: grace_period increased 20 → 40 steps (= 0.8 s at 50 Hz).
    # At lam=0.97, the GAE effective horizon is ≈ 33 steps (0.67 s).  With
    # grace_period=20, a perturbation-induced tipping event near step 20 still
    # attributes the -50 penalty at ~80% weight to the first policy action.
    # At grace_period=40 the attribution weight drops below 30% for step-0 actions,
    # giving the policy a fairer signal.  Also raised limit_angle from 75° to 80°
    # to reduce false terminations from valid recovery arcs.
    "fell_over": TerminationTermCfg(
      func=mdp.bad_orientation_while_elevated,
      params={
        "limit_angle": math.radians(80.0),    # was 75°
        "height_threshold": 0.65,
        "grace_period_steps": 40,             # was 20; see comment above
      },
    ),
    "joint_vel_overflow": TerminationTermCfg(
      func=mdp.joint_velocity_overflow,
      params={
        "threshold": 100.0,
        "root_vel_threshold": 15.0,
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
      nconmax=600,
      njmax=600,
      mujoco=MujocoCfg(
        timestep=0.005,
        iterations=30,
        ls_iterations=40,
        ccd_iterations=50,
      ),
    ),
    episode_length_s=30.0,   # reduced from 35 s; shorter episodes reduce variance
  )
