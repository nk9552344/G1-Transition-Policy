"""Unitree G1 recovery-v1 environment configuration.

Applies G1-specific overrides on top of the base recovery-v1 environment,
following the same pattern as the v3 G1 override.
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


def unitree_g1_recovery_v1_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 recovery-v1 configuration."""
  cfg = make_recovery_v1_env_cfg()

  cfg.sim.contact_sensor_maxmatch = 64

  cfg.scene.entities = {"robot": get_g1_robot_cfg()}

  geom_names = tuple(
    f"{side}_foot{i}_collision" for side in ("left", "right") for i in range(1, 8)
  )

  # Ground-contact sensor for feet — same definition as v3.
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

  # Self-collision sensor — same definition as v3.
  # The robot may collide more during rolling/flipping motions, so history
  # length of 4 ensures transient contacts during get-up are penalised.
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
  # geoms (left_elbow_yaw_collision, left_wrist_collision, left_hand_collision and
  # the right-side equivalents).  This means the sensor fires whether the robot
  # presses its elbows, wrists, or palms into the terrain.
  # Secondary: terrain.
  # Two primary patterns -> sensor.data.found shape (B, 2) [left arm, right arm].
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

  # Wire foot-contact sensors for critic observations.
  cfg.observations["critic"].terms["foot_contact"].params[
    "sensor_name"
  ] = feet_ground_cfg.name
  cfg.observations["critic"].terms["foot_contact_forces"].params[
    "sensor_name"
  ] = feet_ground_cfg.name

  # Domain randomisation targets.
  cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names
  cfg.events["base_com"].params["asset_cfg"].body_names = ("torso_link",)

  # G1 reward body references.
  cfg.rewards["body_orientation_l2"].params["asset_cfg"].body_names = ("torso_link",)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("torso_link",)
  cfg.rewards["torso_height_reward"].params["asset_cfg"].body_names = ("torso_link",)

  # G1 arm reward body references.
  # arm_reach_down tracks the wrist_yaw_link bodies (palms / hand endpoint).
  cfg.rewards["arm_reach_down"].params["asset_cfg"].body_names = (
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
  )
  # elbow_push_from_ground tracks the elbow_link bodies.
  # The arm_ground_contact sensor (arm_ground_cfg) is still present in the scene
  # and is used internally by elbow_push_from_ground for its per-arm contact gate;
  # it is no longer registered as a standalone reward (removed: local-optimum trap).
  cfg.rewards["elbow_push_from_ground"].params["asset_cfg"].body_names = (
    "left_elbow_link",
    "right_elbow_link",
  )

  # G1 stand-up phase reward body references.
  #
  # shank_orientation_reward: shank = knee_link → ankle_roll_link segment.
  cfg.rewards["shank_orientation_reward"].params["knee_asset_cfg"].body_names = (
    "left_knee_link",
    "right_knee_link",
  )
  cfg.rewards["shank_orientation_reward"].params["ankle_asset_cfg"].body_names = (
    "left_ankle_roll_link",
    "right_ankle_roll_link",
  )
  #
  # head_above_feet_reward: head estimated from torso_link; feet from ankle_roll_links.
  cfg.rewards["head_above_feet_reward"].params["torso_asset_cfg"].body_names = ("torso_link",)
  cfg.rewards["head_above_feet_reward"].params["foot_asset_cfg"].body_names = (
    "left_ankle_roll_link",
    "right_ankle_roll_link",
  )
  #
  # feet_proximity_reward tracks ankle_roll_link as foot proxies.
  cfg.rewards["feet_proximity_reward"].params["asset_cfg"].body_names = (
    "left_ankle_roll_link",
    "right_ankle_roll_link",
  )

  # Self-collision penalty — same as v3.
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
