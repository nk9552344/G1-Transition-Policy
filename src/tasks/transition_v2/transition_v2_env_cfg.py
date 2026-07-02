"""Transition-v2: momentum-aware neutral-standing task configuration.

Extends the original transition task with two key upgrades:
  1. Initial momentum at reset — the robot starts with small random linear
     (±0.2 m/s) and angular (±0.3 rad/s) body velocity plus small random
     joint velocities (±0.15 rad/s), forcing the policy to first damp the
     momentum and then converge to neutral rather than starting from rest.
  2. Explicit momentum-damping rewards — angular_velocity_convergence and
     linear_velocity_convergence reward the policy for driving body velocity
     to zero; hold_bonus fires a step reward when the robot simultaneously
     holds neutral pose AND near-zero momentum (the "locked-in" state).

Episode length extended to 20 s (from 15 s) to give the policy enough time
to both damp momentum and converge to neutral.

All other structure (observations, action space, DR events, terminations)
mirrors the original transition policy.
"""

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

import src.tasks.transition_v2.mdp as mdp

# Random joint position offset at episode reset (radians).
JOINT_OFFSET_RANGE = 0.5

# Initial joint velocity range at reset (rad/s).
# Small nonzero value simulates residual motion from a prior locomotion phase.
JOINT_VEL_RANGE = 0.15

# Initial root body linear velocity range at reset (m/s, x and y axes).
# Simulates the robot decelerating to a stop from slow walking.
LINEAR_VEL_RANGE = 0.2

# Initial root body angular velocity range at reset (rad/s, all axes).
# Simulates the robot rocking or rotating as it halts.
ANGULAR_VEL_RANGE = 0.3


def make_transition_v2_env_cfg() -> ManagerBasedRlEnvCfg:
  """Create base transition-v2 task configuration."""

  ##
  # Observations — identical to v1; angular/linear velocity is already present
  # via base_ang_vel sensor and captured by the policy through the reward signal.
  ##

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

  observations = {
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
  }

  ##
  # Metrics
  ##

  metrics = {
    "mean_action_acc": MetricsTermCfg(func=mdp.mean_action_acc),
  }

  ##
  # Actions
  ##

  actions: dict[str, ActionTermCfg] = {
    "joint_pos": JointPositionActionCfg(
      entity_name="robot",
      actuator_names=(".*",),
      scale=0.25,  # Override per-robot via G1_ACTION_SCALE.
      use_default_offset=True,
    )
  }

  ##
  # Events
  ##

  events = {
    # Scatter robots across the flat plane.
    "reset_base": EventTermCfg(
      func=mdp.reset_root_state_uniform,
      mode="reset",
      params={
        "pose_range": {
          "x": (-0.5, 0.5),
          "y": (-0.5, 0.5),
          "z": (0.0, 0.0),
          "yaw": (-math.pi, math.pi),
        },
        # NEW vs v1: robot starts with small random body velocity.
        # Linear: simulates deceleration from slow walking.
        # Angular: simulates rocking / residual rotation as the robot stops.
        "velocity_range": {
          "x": (-LINEAR_VEL_RANGE, LINEAR_VEL_RANGE),
          "y": (-LINEAR_VEL_RANGE, LINEAR_VEL_RANGE),
          "roll": (-ANGULAR_VEL_RANGE, ANGULAR_VEL_RANGE),
          "pitch": (-ANGULAR_VEL_RANGE, ANGULAR_VEL_RANGE),
          "yaw": (-ANGULAR_VEL_RANGE * 0.5, ANGULAR_VEL_RANGE * 0.5),
        },
      },
    ),
    # Apply random joint offsets AND small initial joint velocities.
    # v1 used velocity_range=(0.0, 0.0). Here we add a small range to
    # simulate joints still settling from prior motion.
    "reset_robot_joints": EventTermCfg(
      func=mdp.reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": (-JOINT_OFFSET_RANGE, JOINT_OFFSET_RANGE),
        "velocity_range": (-JOINT_VEL_RANGE, JOINT_VEL_RANGE),
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      },
    ),
    # Occasional pushes — same as v1 since the policy must also handle
    # disturbances after it has reached neutral.
    "push_robot": EventTermCfg(
      func=mdp.push_by_setting_velocity,
      mode="interval",
      interval_range_s=(8.0, 10.0),
      params={
        "velocity_range": {
          "x": (-0.3, 0.3),
          "y": (-0.3, 0.3),
          "z": (-0.2, 0.2),
          "roll": (-0.3, 0.3),
          "pitch": (-0.3, 0.3),
          "yaw": (-0.5, 0.5),
        },
      },
    ),
    # Domain randomisation — same as v1.
    "foot_friction": EventTermCfg(
      mode="startup",
      func=dr.geom_friction,
      params={
        "asset_cfg": SceneEntityCfg("robot", geom_names=()),  # Set per-robot.
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
        "asset_cfg": SceneEntityCfg("robot", body_names=()),  # Set per-robot.
        "operation": "add",
        "ranges": {0: (-0.05, 0.05), 1: (-0.05, 0.05), 2: (-0.05, 0.05)},
      },
    ),
  }

  ##
  # Rewards
  #
  # Design rationale for the additions:
  #
  # pose_convergence (+2.0): Unchanged from v1 — primary signal for joint convergence.
  #
  # angular_velocity_convergence (+0.7): Explicitly rewards damping body angular
  #   velocity to zero. std=0.3 rad/s matches the initial ANGULAR_VEL_RANGE so the
  #   gradient is meaningful from the very first step. Without this, pose_convergence
  #   alone does not guarantee the robot stops rocking after reaching neutral.
  #
  # linear_velocity_convergence (+0.4): Same pattern for linear velocity. std=0.2 m/s
  #   matches LINEAR_VEL_RANGE. Smaller weight than angular because linear drift is
  #   less destabilising than rocking.
  #
  # hold_bonus (+1.0): Binary step reward that fires only when the robot is
  #   simultaneously near neutral (mean |q-q_default| < 0.08 rad), near-zero
  #   angular velocity (|ω| < 0.15 rad/s) and near-zero linear velocity
  #   (|v| < 0.1 m/s). This "locked in" bonus has no gradient direction of its
  #   own — it fires as an amplifier once the other rewards have guided the
  #   policy near the target, incentivising it to hold there rather than drift.
  #
  # both_feet_contact (+0.5): Unchanged from v1.
  #
  # body_orientation_l2 (-2.0): Unchanged from v1.
  #
  # joint_vel_penalty (-0.02): 2× heavier than v1 (-0.01). The robot now starts
  #   with nonzero joint velocity, so a stronger penalty is needed to discourage
  #   residual motion after convergence.
  #
  # body_ang_vel (-0.1): 2× heavier than v1 (-0.05). Rocking is more prevalent
  #   when the robot starts with angular momentum.
  #
  # angular_momentum, joint_acc_l2, joint_pos_limits, action_rate_l2,
  # is_terminated: Unchanged from v1.
  ##

  rewards = {
    "pose_convergence": RewardTermCfg(
      func=mdp.pose_convergence,
      weight=2.0,
      params={
        "std": 0.25,
        "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
      },
    ),
    # NEW: damp angular momentum explicitly
    "angular_velocity_convergence": RewardTermCfg(
      func=mdp.angular_velocity_convergence,
      weight=0.7,
      params={
        "std": ANGULAR_VEL_RANGE,  # 0.3 rad/s — gradient is nonzero at max initial ω
        "asset_cfg": SceneEntityCfg("robot"),
      },
    ),
    # NEW: damp linear momentum explicitly
    "linear_velocity_convergence": RewardTermCfg(
      func=mdp.linear_velocity_convergence,
      weight=0.4,
      params={
        "std": LINEAR_VEL_RANGE,  # 0.2 m/s — gradient is nonzero at max initial v
        "asset_cfg": SceneEntityCfg("robot"),
      },
    ),
    # NEW: bonus for being locked in to neutral with zero momentum
    "hold_bonus": RewardTermCfg(
      func=mdp.hold_bonus,
      weight=1.0,
      params={
        "pose_threshold": 0.08,        # mean |q - q_default| < 0.08 rad (~4.6°)
        "ang_vel_threshold": 0.15,     # |ω_b| < 0.15 rad/s
        "lin_vel_threshold": 0.10,     # |v_b| < 0.10 m/s
        "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
      },
    ),
    "both_feet_contact": RewardTermCfg(
      func=mdp.both_feet_contact,
      weight=0.5,
      params={"sensor_name": "feet_ground_contact"},
    ),
    "body_orientation_l2": RewardTermCfg(
      func=mdp.body_orientation_l2,
      weight=-2.0,
      params={"asset_cfg": SceneEntityCfg("robot", body_names=())},  # Set per-robot.
    ),
    "joint_vel_penalty": RewardTermCfg(
      func=mdp.joint_vel_penalty,
      weight=-0.02,  # 2× v1: stronger because robot starts with nonzero joint vel
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
    ),
    "body_ang_vel": RewardTermCfg(
      func=mdp.body_angular_velocity_penalty,
      weight=-0.1,  # 2× v1: rocking is more common given initial angular momentum
      params={"asset_cfg": SceneEntityCfg("robot", body_names=())},  # Set per-robot.
    ),
    "angular_momentum": RewardTermCfg(
      func=mdp.angular_momentum_penalty,
      weight=-0.025,
      params={"sensor_name": "robot/root_angmom"},
    ),
    "is_terminated": RewardTermCfg(func=mdp.is_terminated, weight=-200.0),
    "joint_acc_l2": RewardTermCfg(func=mdp.joint_acc_l2, weight=-2.5e-7),
    "joint_pos_limits": RewardTermCfg(func=mdp.joint_pos_limits, weight=-10.0),
    "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.05),
  }

  ##
  # Terminations — same as v1
  ##

  terminations = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    "fell_over": TerminationTermCfg(
      func=mdp.bad_orientation,
      params={"limit_angle": math.radians(70.0)},
    ),
  }

  ##
  # Assemble and return
  ##

  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane"),
      num_envs=1,
      extent=2.0,
    ),
    observations=observations,
    actions=actions,
    commands={},
    events=events,
    rewards=rewards,
    terminations=terminations,
    curriculum={},
    metrics=metrics,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="",  # Set per-robot.
      distance=3.0,
      elevation=-5.0,
      azimuth=90.0,
    ),
    sim=SimulationCfg(
      nconmax=None,
      njmax=300,
      mujoco=MujocoCfg(
        timestep=0.005,
        iterations=10,
        ls_iterations=20,
        ccd_iterations=50,
      ),
    ),
    decimation=4,
    episode_length_s=20.0,  # Extended from 15 s: more time to damp momentum
  )
