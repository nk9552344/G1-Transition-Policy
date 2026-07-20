"""Unitree G1 recovery-v1 environment configuration.

Applies G1-specific overrides on top of the base recovery-v1 environment:
  - Robot model and FK-verified joint defaults
  - Ground contact sensor for feet (with force, track_air_time)
  - Self-collision sensor
  - G1_ACTION_SCALE per joint
  - Body/joint name resolution for all reward terms that reference named links
  - Viewer camera attached to torso_link

Arm ground-contact sensor removed in this version: arm-based rewards
(arm_reach_down, pushup_support_reward) are not wired in the base config.
"""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg

from src.assets.robots import G1_ACTION_SCALE, get_g1_robot_cfg
from src.tasks.transition.mdp import self_collision_cost
from src.tasks.recovery_v1.recovery_v1_env_cfg import make_recovery_v1_env_cfg
import src.tasks.recovery_v1.mdp as mdp


def unitree_g1_recovery_v1_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 recovery-v1 configuration."""
  cfg = make_recovery_v1_env_cfg()

  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.njmax = 600

  cfg.scene.entities = {"robot": get_g1_robot_cfg()}

  geom_names = tuple(
    f"{side}_foot{i}_collision" for side in ("left", "right") for i in range(1, 8)
  )

  # Ground-contact sensor for feet.
  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
      mode="subtree",
      pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )

  # Self-collision sensor.
  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )

  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    feet_ground_cfg,
    self_collision_cfg,
  )

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = G1_ACTION_SCALE

  cfg.viewer.body_name = "torso_link"

  # ── Observation wiring ────────────────────────────────────────────────────────
  cfg.observations["actor"].terms["foot_contact"].params[
    "sensor_name"
  ] = feet_ground_cfg.name

  cfg.observations["critic"].terms["foot_contact"].params[
    "sensor_name"
  ] = feet_ground_cfg.name
  cfg.observations["critic"].terms["foot_contact_forces"].params[
    "sensor_name"
  ] = feet_ground_cfg.name

  # ── Domain randomisation targets ─────────────────────────────────────────────
  cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names
  cfg.events["base_com"].params["asset_cfg"].body_names = ("torso_link",)

  # ── Reward body references ────────────────────────────────────────────────────
  cfg.rewards["torso_height_reward"].params["asset_cfg"].body_names = ("torso_link",)
  cfg.rewards["height_gated_ang_vel"].params["asset_cfg"].body_names = ("torso_link",)

  cfg.rewards["shank_orientation_reward"].params["knee_asset_cfg"].body_names = (
    "left_knee_link",
    "right_knee_link",
  )
  cfg.rewards["shank_orientation_reward"].params["ankle_asset_cfg"].body_names = (
    "left_ankle_roll_link",
    "right_ankle_roll_link",
  )

  cfg.rewards["head_above_feet_reward"].params["torso_asset_cfg"].body_names = ("torso_link",)
  cfg.rewards["head_above_feet_reward"].params["foot_asset_cfg"].body_names = (
    "left_ankle_roll_link",
    "right_ankle_roll_link",
  )

  cfg.rewards["feet_proximity_reward"].params["asset_cfg"].body_names = (
    "left_ankle_roll_link",
    "right_ankle_roll_link",
  )

  # Self-collision penalty.
  cfg.rewards["self_collisions"] = RewardTermCfg(
    func=self_collision_cost,
    weight=-1.0,
    params={"sensor_name": self_collision_cfg.name, "force_threshold": 10.0},
  )

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.events["randomize_terrain"] = EventTermCfg(
      func=envs_mdp.randomize_terrain,
      mode="reset",
      params={},
    )

  return cfg
