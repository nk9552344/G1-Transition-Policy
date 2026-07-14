"""Unitree G1 recovery-v1 environment configuration.

Applies G1-specific overrides on top of the base recovery-v1 environment,
following the same pattern as the v3 G1 override.
"""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
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

  # Arm ground contact sensor.
  # Primary: subtree of left_elbow_link and right_elbow_link — covers all forearm
  # geoms (elbow_yaw, wrist, hand).  Two primary patterns → shape (B, 2).
  arm_ground_cfg = ContactSensorCfg(
    name="arm_ground_contact",
    primary=ContactMatch(
      mode="subtree",
      pattern=r"^(left_elbow_link|right_elbow_link)$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found",),
    reduce="netforce",
    num_slots=1,
  )

  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    feet_ground_cfg,
    self_collision_cfg,
    arm_ground_cfg,
  )

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = G1_ACTION_SCALE

  cfg.viewer.body_name = "torso_link"

  # ── Actor observation wiring ──────────────────────────────────────────────────
  # foot_contact in actor: same sensor as critic, no extra sensor needed.
  cfg.observations["actor"].terms["foot_contact"].params[
    "sensor_name"
  ] = feet_ground_cfg.name

  # arm_contact in actor: reads arm_ground_contact sensor's found field.
  # Uses mdp.foot_contact (same pattern — reads sensor.data.found.float()).
  cfg.observations["actor"].terms["arm_contact"].params[
    "sensor_name"
  ] = arm_ground_cfg.name

  # ── Critic observation wiring ─────────────────────────────────────────────────
  cfg.observations["critic"].terms["foot_contact"].params[
    "sensor_name"
  ] = feet_ground_cfg.name
  cfg.observations["critic"].terms["foot_contact_forces"].params[
    "sensor_name"
  ] = feet_ground_cfg.name
  cfg.observations["critic"].terms["arm_contact"].params[
    "sensor_name"
  ] = arm_ground_cfg.name

  # ── Domain randomisation targets ─────────────────────────────────────────────
  cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names
  cfg.events["base_com"].params["asset_cfg"].body_names = ("torso_link",)

  # ── Reward body references ────────────────────────────────────────────────────
  cfg.rewards["body_orientation_l2"].params["asset_cfg"].body_names = ("torso_link",)
  cfg.rewards["torso_height_reward"].params["asset_cfg"].body_names = ("torso_link",)

  # height_gated_ang_vel: same body as old body_ang_vel (torso_link).
  cfg.rewards["height_gated_ang_vel"].params["asset_cfg"].body_names = ("torso_link",)

  # arm_reach_down tracks wrist_yaw_link bodies.
  cfg.rewards["arm_reach_down"].params["asset_cfg"].body_names = (
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
  )

  # pushup_support_reward tracks elbow_link bodies (same as former elbow_push_from_ground).
  cfg.rewards["pushup_support_reward"].params["asset_cfg"].body_names = (
    "left_elbow_link",
    "right_elbow_link",
  )

  # shank_orientation_reward: knee_link → ankle_roll_link.
  cfg.rewards["shank_orientation_reward"].params["knee_asset_cfg"].body_names = (
    "left_knee_link",
    "right_knee_link",
  )
  cfg.rewards["shank_orientation_reward"].params["ankle_asset_cfg"].body_names = (
    "left_ankle_roll_link",
    "right_ankle_roll_link",
  )

  # head_above_feet_reward: head estimated from torso_link; feet from ankle_roll_links.
  cfg.rewards["head_above_feet_reward"].params["torso_asset_cfg"].body_names = ("torso_link",)
  cfg.rewards["head_above_feet_reward"].params["foot_asset_cfg"].body_names = (
    "left_ankle_roll_link",
    "right_ankle_roll_link",
  )

  # feet_proximity_reward tracks ankle_roll_links.
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
