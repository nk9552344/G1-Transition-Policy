"""Unitree G1 transition-to-neutral-standing environment configuration.

Applies G1-specific overrides on top of the base transition environment:
  - Correct action scale from motor dynamics (G1_ACTION_SCALE).
  - Contact sensors for feet (ground contact) and self-collision detection.
  - Body name references for orientation / angular-velocity rewards.
  - Self-collision penalty to prevent the limbs from driving into each other
    while the robot converges to the neutral pose.
"""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg

from src.assets.robots import G1_ACTION_SCALE, get_g1_robot_cfg
from src.tasks.transition.mdp import both_feet_contact, self_collision_cost
from src.tasks.transition.transition_env_cfg import make_transition_env_cfg


def unitree_g1_transition_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 transition-to-neutral-standing configuration."""
  cfg = make_transition_env_cfg()

  # Solver settings tuned for G1 (flat terrain, lower contact complexity).
  cfg.sim.contact_sensor_maxmatch = 64

  cfg.scene.entities = {"robot": get_g1_robot_cfg()}

  site_names = ("left_foot", "right_foot")
  geom_names = tuple(
    f"{side}_foot{i}_collision" for side in ("left", "right") for i in range(1, 8)
  )

  # Ground-contact sensor: tracks whether each foot subtree is in contact with
  # the terrain and the net contact force, used by both_feet_contact reward and
  # the critic foot observations.
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

  # Self-collision sensor rooted at pelvis; detects when limbs contact each
  # other while the robot is reconfiguring toward neutral.
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

  # Use per-joint action scale derived from G1 motor stiffness / effort limits.
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

  # Penalise self-collisions that can arise during joint reconfiguration.
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
