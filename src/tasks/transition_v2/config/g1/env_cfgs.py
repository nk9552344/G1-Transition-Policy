"""Unitree G1 transition-v2 environment configuration.

Applies G1-specific overrides on top of the base transition-v2 environment,
mirroring the pattern from the original transition task's G1 config.
"""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg

from src.assets.robots import G1_ACTION_SCALE, get_g1_robot_cfg
from src.tasks.transition.mdp import both_feet_contact, self_collision_cost
from src.tasks.transition_v2.transition_v2_env_cfg import make_transition_v2_env_cfg


def unitree_g1_transition_v2_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 transition-v2 configuration."""
  cfg = make_transition_v2_env_cfg()

  cfg.sim.contact_sensor_maxmatch = 64

  cfg.scene.entities = {"robot": get_g1_robot_cfg()}

  geom_names = tuple(
    f"{side}_foot{i}_collision" for side in ("left", "right") for i in range(1, 8)
  )

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

  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )

  cfg.scene.sensors = (cfg.scene.sensors or ()) + (feet_ground_cfg, self_collision_cfg)

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = G1_ACTION_SCALE

  cfg.viewer.body_name = "torso_link"

  # Wire contact sensors for critic observations.
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

  # Self-collision penalty — same as v1.
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
