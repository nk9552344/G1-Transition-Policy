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
  Stand up from any fallen ground position (supine, prone, sitting, or
  partially tucked). Nine initial-state templates cover the full trajectory
  so every sub-skill gets gradient:
    Fallen (22 %):     supine, prone                          (base_z ~0.25 m)
    Sitting (22 %):    sitting_low (40° lean), sitting_high (30° lean)
    Squat-lean (11 %): squat_lean (20° lean, knee=1.2 rad)   (transition zone)
    Bent (44 %):       home, knees_bent, squat, deep_squat    (FK-verified)

Reward design summary
  Phase 1 — get off the floor:
    orientation_recovery (+3.0): pelvis gravity projection toward upright
    height_recovery (+2.0):      pelvis rising toward 0.78 m
    torso_height_reward (+3.0):  CHEST (not pelvis) rising — breaks leg-bridge
    arm_reach_down (+1.5):       hands toward floor when flat
    elbow_push_from_ground (+3.5): push-up only when arm is in contact
  Phase 2 — sit-to-stand:
    shank_orientation_reward (+3.5): shanks vertical (not forward-extended)
    head_above_feet_reward (+2.5):   head 1.15 m above feet (relative, not abs)
    feet_proximity_reward (+2.0):    feet under pelvis XY
  Phase 3 — hold standing:
    pose_convergence_gated (+1.5):   joints toward default, gated by upright
    hold_bonus (+1.0):               locked-in bonus
    both_feet_contact (+0.2):        both feet on ground

Key design lessons (do not revert):
  - All position-based (Gaussian). NO velocity-based rewards — they are
    gameable by oscillation: robot earns reward on every upward half-cycle
    and zero on the downward half, making waist-wiggling very profitable.
  - body_ang_vel weight -0.03 (NOT -0.10): recovery requires deliberate
    body rotation. -0.10 actively fights the roll/flip motion needed to
    get off the floor.
  - shank_orientation_reward std=0.50 (NOT 0.30): at std=0.30 the sitting
    range (cosine 0.55-0.70) gives near-zero reward → PPO can't see the
    gradient. At 0.50 the same range gives 0.44 → 4× clearer signal.
  - fallen_joint_perturbation=0.6: G1 push-up shoulder_pitch is 0.6-0.8 rad
    above default. At 0.3 rad perturbation the arm never starts near the
    needed position. At 0.6 it regularly does.
  - feet_proximity_reward std=0.45: at 0.30 the gradient at 0.60 m
    (typical sitting foot distance) is near-zero. At 0.45 it is meaningful.
  - height-gated termination at 0.65 m: the prone→bridge→squat trajectory
    crosses 0.50 m while still tilted — using 0.50 m created a punishment
    zone that taught the robot to AVOID raising past 0.50 m.
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
    "joint_pos": ObservationTermCfg(
      func=mdp.joint_pos_rel,
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    "joint_vel": ObservationTermCfg(
      func=mdp.joint_vel_rel,
      noise=Unoise(n_min=-1.5, n_max=1.5),
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
    "foot_contact": ObservationTermCfg(
      func=mdp.foot_contact,
      params={"sensor_name": "feet_ground_contact"},
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
    # ── Reset: 9-template curriculum (22% fallen / 22% sitting / 11% squat-lean / 44% bent)
    "reset_robot": EventTermCfg(
      func=mdp.reset_to_fallen_or_bent_pose,
      mode="reset",
      params={
        "all_pose_configs": ALL_POSE_CONFIGS,
        "xy_pos_range": 0.5,        # ±0.5 m XY scatter within env cell
        "yaw_range": math.pi,       # full 360° random heading
        # Joint noise for fallen/sitting/squat-lean templates.
        # 0.6 rad: G1 shoulder_pitch needs to reach 0.6-0.8 rad above default
        # for push-up position. At ±0.3 the arm never starts near that range.
        "fallen_joint_perturbation": 0.6,
        # Joint noise for bent (upright-squat) templates.
        "leg_perturbation": 0.10,
        "other_perturbation": 0.35,
        # Initial body/joint velocities.
        "joint_vel_range": 0.15,
        "lin_vel_range":   0.20,
        "ang_vel_range":   0.30,
        # Leg-joint IDs for bent and squat-lean templates (set per-robot via regex).
        "knee_cfg":      SceneEntityCfg("robot", joint_names=(".*_knee_joint",)),
        "hip_pitch_cfg": SceneEntityCfg("robot", joint_names=(".*_hip_pitch_joint",)),
        "ankle_cfg":     SceneEntityCfg("robot", joint_names=(".*_ankle_pitch_joint",)),
        "asset_cfg":     SceneEntityCfg("robot"),
      },
    ),
    # ── Disturbance pushes (keep recovery robust to perturbations).
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
    # ── Domain randomisation (startup, applied once per training run).
    "foot_friction": EventTermCfg(
      mode="startup",
      func=dr.geom_friction,
      params={
        "asset_cfg": SceneEntityCfg("robot", geom_names=()),  # set per-robot
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
        "asset_cfg": SceneEntityCfg("robot", body_names=()),  # set per-robot
        "operation": "add",
        "ranges": {0: (-0.05, 0.05), 1: (-0.05, 0.05), 2: (-0.05, 0.05)},
      },
    ),
  }

  # ── Rewards ───────────────────────────────────────────────────────────────────
  rewards = {

    # ── Phase 1: get off the floor ────────────────────────────────────────────
    #
    # orientation_recovery: primary upright signal. Gaussian on (proj_gz + 1)^2.
    #   Correctly distinguishes upright (0), flat (1), inverted (4).
    #   std=1.0 provides gradient across the full range.
    "orientation_recovery": RewardTermCfg(
      func=mdp.orientation_recovery,
      weight=3.0,
      params={
        "std": 1.0,
        "asset_cfg": SceneEntityCfg("robot"),
      },
    ),
    # height_recovery: pelvis rising. Gradient from 0.25 m (fallen) to 0.78 m.
    "height_recovery": RewardTermCfg(
      func=mdp.height_recovery,
      weight=2.0,
      params={
        "target_height": 0.78,   # m — G1 pelvis at standing
        "std": 0.65,             # m — wide enough to reach from fallen height
        "asset_cfg": SceneEntityCfg("robot"),
      },
    ),
    # torso_height_reward: CHEST (not pelvis) height. Position-based Gaussian —
    #   cannot be farmed by waist oscillation (oscillation earns reward only at
    #   the mean position). Body names set per-robot to the torso_link.
    "torso_height_reward": RewardTermCfg(
      func=mdp.torso_height_reward,
      weight=3.0,
      params={
        "target_height": 0.90,  # m — G1 torso_link at standing ~0.88-0.92 m
        "std": 0.50,            # m — gradient from 0.15 m (flat) to target
        "asset_cfg": SceneEntityCfg("robot", body_names=()),  # set per-robot
      },
    ),
    # arm_reach_down: pull hands toward floor level when robot is flat.
    #   Provides gradient BEFORE ground contact so the arm motion is
    #   discoverable without already knowing how to do push-ups.
    #   flat_gate_threshold=-0.85: active until 31° from upright (not 45°),
    #   so arm guidance continues through the sit-to-stand transition.
    "arm_reach_down": RewardTermCfg(
      func=mdp.arm_reach_down,
      weight=1.5,
      params={
        "height_gate": 0.60,          # m — suppress once hand is above mid-recovery
        "flat_gate_threshold": -0.85, # active until 31° from upright
        "asset_cfg": SceneEntityCfg("robot", body_names=()),  # set per-robot
      },
    ),
    # elbow_push_from_ground: elbow velocity ONLY when that arm is on the ground.
    #   Left arm contact gates left elbow; right arm contact gates right elbow.
    #   Forces the sequence: reach → plant → push → chest rises.
    #   sensor_name and body_names set per-robot in the G1 config.
    "elbow_push_from_ground": RewardTermCfg(
      func=mdp.elbow_push_from_ground,
      weight=3.5,
      params={
        "sensor_name": "arm_ground_contact",  # added in G1 config
        "height_gate": 0.70,
        "max_vel": 1.5,
        "flat_gate_threshold": -0.85,
        "asset_cfg": SceneEntityCfg("robot", body_names=()),  # set per-robot
      },
    ),

    # ── Phase 2: sit-to-stand ─────────────────────────────────────────────────
    #
    # shank_orientation_reward: ANTI-SITTING signal (HoST-inspired).
    #   In sitting with legs extended forward, knee→ankle direction is 30-50°
    #   from vertical (cosine ≈ 0.55-0.70). Squatting/standing requires
    #   cosine ≈ 1.0 (shank vertical). std=0.50 (NOT 0.30): at 0.30, sitting
    #   range gives near-zero reward (exp(-2.8)≈0.06) — gradient invisible to
    #   PPO. At 0.50 the same range gives exp(-0.81)≈0.44 — 4× clearer.
    #   Gate: pelvis > 0.30 m (suppress during flat push-up phase).
    #   body_names set per-robot to knee_link and ankle_roll_link bodies.
    "shank_orientation_reward": RewardTermCfg(
      func=mdp.shank_orientation_reward,
      weight=3.5,
      params={
        "height_gate": 0.30,  # m — active from floor clearance
        "std": 0.50,          # wider for clear gradient at sitting range
        "knee_asset_cfg":  SceneEntityCfg("robot", body_names=()),  # set per-robot
        "ankle_asset_cfg": SceneEntityCfg("robot", body_names=()),  # set per-robot
      },
    ),
    # head_above_feet_reward: head height RELATIVE to feet (HoST-inspired).
    #   Relative measurement gives 2.5× ratio sitting vs standing (0.88 m vs
    #   1.28 m), compared to 1.5× for the old absolute head_height_reward.
    #   target=1.15 m above feet, std=0.25 m.
    #   body_names set per-robot to torso_link and ankle_roll_link bodies.
    "head_above_feet_reward": RewardTermCfg(
      func=mdp.head_above_feet_reward,
      weight=2.5,
      params={
        "target_height": 1.15,  # m above feet — G1 standing ≈ 1.28 m
        "std": 0.25,
        "head_offset": 0.43,    # m — G1 head geom above torso_link origin
        "torso_asset_cfg": SceneEntityCfg("robot", body_names=()),  # set per-robot
        "foot_asset_cfg":  SceneEntityCfg("robot", body_names=()),  # set per-robot
      },
    ),
    # feet_proximity_reward: feet close to pelvis XY when pelvis is elevated.
    #   Sitting: feet 0.60 m from pelvis; squat: 0.10-0.20 m.
    #   std=0.45 (NOT 0.30): at 0.30, gradient at 0.60 m foot distance is
    #   near-zero. At 0.45 it is meaningful from the start of the tuck.
    "feet_proximity_reward": RewardTermCfg(
      func=mdp.feet_proximity_reward,
      weight=2.0,
      params={
        "height_gate": 0.35,  # m — active once pelvis clears prone height
        "std": 0.45,
        "asset_cfg": SceneEntityCfg("robot", body_names=()),  # set per-robot
      },
    ),

    # ── Phase 3: hold standing ────────────────────────────────────────────────
    #
    # pose_convergence_gated: joint positions toward default, gated by (-proj_gz).
    #   When flat (proj_gz≈0): gate≈0 — no pose reward (prevent "stay flat
    #   in default joints" local optimum). When upright (proj_gz≈-1): gate=1.
    "pose_convergence_gated": RewardTermCfg(
      func=mdp.pose_convergence_gated,
      weight=1.5,
      params={
        "std": 0.5,
        "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
      },
    ),
    # hold_bonus: fires when robot is simultaneously near-neutral AND near-zero
    #   body velocity. Incentivises holding once the other rewards guide to
    #   standing rather than drifting back down.
    "hold_bonus": RewardTermCfg(
      func=mdp.hold_bonus,
      weight=1.0,
      params={
        "pose_threshold":    0.08,   # mean |q - q_default| < 0.08 rad
        "ang_vel_threshold": 0.15,   # |ω_b| < 0.15 rad/s
        "lin_vel_threshold": 0.10,   # |v_b| < 0.10 m/s
        "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
      },
    ),
    # both_feet_contact: low weight (0.2, not 0.5) because rolling/flipping
    #   requires feet to leave the ground. 0.5 over-penalised rolling motions.
    "both_feet_contact": RewardTermCfg(
      func=mdp.both_feet_contact,
      weight=0.2,
      params={"sensor_name": "feet_ground_contact"},
    ),

    # ── Penalties ─────────────────────────────────────────────────────────────
    #
    # body_orientation_l2: penalise non-upright torso. Weight -3.0 (increased
    #   from v2's -2.0) to make the sitting/flat pose clearly unattractive.
    "body_orientation_l2": RewardTermCfg(
      func=mdp.body_orientation_l2,
      weight=-3.0,
      params={"asset_cfg": SceneEntityCfg("robot", body_names=())},  # set per-robot
    ),
    # body_ang_vel: penalise excessive body spin. Weight -0.03 (NOT -0.10):
    #   recovery REQUIRES deliberate body rotation (rolling from supine to prone,
    #   then from prone to upright). -0.10 actively fought the get-up motion.
    "body_ang_vel": RewardTermCfg(
      func=mdp.body_angular_velocity_penalty,
      weight=-0.03,
      params={"asset_cfg": SceneEntityCfg("robot", body_names=())},  # set per-robot
    ),
    # angular_momentum: same rationale as body_ang_vel. -0.008 (was -0.025).
    "angular_momentum": RewardTermCfg(
      func=mdp.angular_momentum_penalty,
      weight=-0.008,
      params={"sensor_name": "robot/root_angmom"},
    ),
    # joint_vel_penalty: -0.008 (was -0.02). Vigorous push-up and leg-swing
    #   motions require fast joint velocities. Over-penalising them slows recovery.
    "joint_vel_penalty": RewardTermCfg(
      func=mdp.joint_vel_penalty,
      weight=-0.008,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
    ),
    "joint_acc_l2": RewardTermCfg(func=mdp.joint_acc_l2, weight=-2.5e-7),
    "joint_pos_limits": RewardTermCfg(func=mdp.joint_pos_limits, weight=-10.0),
    # action_rate_l2: -0.02 (was -0.05). Recovery transitions rapidly between
    #   phases (push → tuck → stand); smoother penalty allows faster transitions.
    "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.02),
    "is_terminated": RewardTermCfg(func=mdp.is_terminated, weight=-200.0),
  }

  # ── Terminations ───────────────────────────────────────────────────────────────
  terminations = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    # fell_over with HEIGHT GATE at 0.65 m (not 0.50 m).
    #   At 0.50 m: the prone→bridge→squat trajectory crosses 0.50 m while
    #   still tilted → episode terminates → robot learns to AVOID rising past
    #   0.50 m → stuck in shallow leg-bridge. At 0.65 m the full mid-recovery
    #   path (0.15-0.65 m) is termination-free; unstable near-standing states
    #   (> 0.65 m + tilted 75°) are still terminated.
    "fell_over": TerminationTermCfg(
      func=mdp.bad_orientation_while_elevated,
      params={
        "limit_angle": math.radians(75.0),
        "height_threshold": 0.65,  # m — raised from 0.50
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
    decimation=4,  # 4 sim steps per policy step: 0.005 × 4 = 0.02 s control period (50 Hz)
    sim=SimulationCfg(
      nconmax=None,
      # njmax=600: G1 floor contact uses up to 454 constraint equations
      # (nefc). 600 gives safe headroom without excessive memory cost.
      # transition_v2 inherited njmax=300 — the root cause of the NaN crash
      # at 1.4k steps (nefc overflow corrupted physics → NaN observations).
      njmax=600,
      mujoco=MujocoCfg(
        timestep=0.005,
        iterations=10,
        ls_iterations=20,
        ccd_iterations=50,
      ),
    ),
    episode_length_s=35.0,
  )
