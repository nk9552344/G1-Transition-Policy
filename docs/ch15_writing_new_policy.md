# Chapter 15: Writing a New Policy From Scratch

This chapter walks through the complete process of creating a new policy using the transition
policy as a reference. We will build a hypothetical **"balance recovery"** policy — a policy
that drives the G1 from any tilted, near-fall configuration back to upright, even when the
initial tilt is more aggressive (up to 30°) and the robot starts with nonzero angular velocity.

This is different from the transition policy:
- Transition: offsets in joint space (joints far from home), zero body velocity
- Balance recovery: offsets in both joint space AND root angular velocity (robot is rocking)

---

## 15.1 Step 0: Decide What the Policy Should Do

Before writing any code, answer these questions:

**1. What is the goal?**
Drive the robot to neutral joint pose AND zero angular velocity simultaneously.

**2. What inputs does the policy need?**
- Joint error (current joints - home joints)
- Joint velocities
- Body angular velocity (how fast it is rocking)
- Projected gravity (which direction is up)
- Previous action (for smoothness)

Same as the transition policy — no new observations needed.

**3. What is different from the transition policy?**
- More aggressive initial angular velocity (up to ±1.0 rad/s at reset)
- Shorter episode length (8 seconds — balance recovery is faster than posture transition)
- `joint_vel_penalty` weight should be higher (the robot starts with momentum to stop)
- Maybe add a dedicated angular velocity convergence reward

**4. What terrain?**
Flat plane — same as transition policy.

**5. One new task or a modification of an existing one?**
New task. We will create `Unitree-G1-BalanceRecovery`.

---

## 15.2 Step 1: Create the Directory Structure

```
src/tasks/
└── balance_recovery/
    ├── __init__.py                  # Empty or "Balance recovery task"
    ├── balance_recovery_env_cfg.py  # Base config factory
    ├── config/
    │   ├── __init__.py              # Empty
    │   └── g1/
    │       ├── __init__.py          # Task registration
    │       ├── env_cfgs.py          # G1-specific overrides
    │       └── rl_cfg.py            # PPO hyperparameters
    ├── mdp/
    │   ├── __init__.py              # Re-exports
    │   └── rewards.py               # Custom reward functions
    └── rl/
        ├── __init__.py
        └── runner.py                # (Can reuse TransitionOnPolicyRunner or VelocityOnPolicyRunner)
```

---

## 15.3 Step 2: Write the MDP Rewards

Start with the custom reward functions. These are the most important piece and should be
written before the configuration.

```python
# src/tasks/balance_recovery/mdp/rewards.py

from __future__ import annotations
from typing import TYPE_CHECKING
import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def pose_convergence(
    env: ManagerBasedRlEnv,
    std: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward closeness to default joint configuration.

    Returns exp(-mean(error²) / std²) ∈ [0, 1].
    At std-sized mean error: reward ≈ 0.37.
    """
    asset = env.scene[asset_cfg.name]
    q = asset.data.joint_pos[:, asset_cfg.joint_ids]
    q_default = asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    mse = torch.mean(torch.square(q - q_default), dim=1)
    return torch.exp(-mse / std**2)


def angular_velocity_convergence(
    env: ManagerBasedRlEnv,
    std: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward low root body angular velocity (robot settling, not rocking).

    Returns exp(-|ω|² / std²) ∈ [0, 1].
    At std-sized angular velocity: reward ≈ 0.37.
    """
    asset = env.scene[asset_cfg.name]
    # Root link angular velocity in world frame [B, 3]
    ang_vel = asset.data.root_link_ang_vel_w
    ang_vel_sq = torch.sum(torch.square(ang_vel), dim=1)  # [B]
    return torch.exp(-ang_vel_sq / std**2)
```

**Reasoning for `angular_velocity_convergence`:**
- `std=0.5 rad/s`: At 0.5 rad/s angular velocity, reward = 0.37. At 0 rad/s, reward = 1.0.
  At the initial ±1.0 rad/s angular velocity, reward = exp(-1.0/0.25) = exp(-4) ≈ 0.018
  (nonzero gradient even from the initial state).
- This reward is on top of `pose_convergence` to explicitly penalize rocking motion even
  when the joints are momentarily near neutral.

---

## 15.4 Step 3: Write the MDP `__init__.py`

```python
# src/tasks/balance_recovery/mdp/__init__.py

from mjlab.envs.mdp import *  # noqa: F401, F403

# Reuse shared rewards and observations from the transition task
from src.tasks.velocity.mdp.rewards import (  # noqa: F401
    angular_momentum_penalty,
    body_angular_velocity_penalty,
    body_orientation_l2,
    self_collision_cost,
)
from src.tasks.velocity.mdp.observations import (  # noqa: F401
    foot_contact,
    foot_contact_forces,
)

# Also reuse the joint_vel_penalty and both_feet_contact from transition
from src.tasks.transition.mdp.rewards import (  # noqa: F401
    joint_vel_penalty,
    both_feet_contact,
)

from .rewards import *  # noqa: F401, F403  ← our new functions
```

---

## 15.5 Step 4: Write the Base Environment Config

```python
# src/tasks/balance_recovery/balance_recovery_env_cfg.py

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

import src.tasks.balance_recovery.mdp as mdp

JOINT_OFFSET_RANGE = 0.4      # rad — slightly less than transition task
ANGULAR_VEL_RANGE  = 1.0      # rad/s — new: robot starts with angular velocity


def make_balance_recovery_env_cfg() -> ManagerBasedRlEnvCfg:
    """Base configuration for the balance recovery task."""

    ##
    # Observations — same as transition policy (joint error is sufficient)
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
    # Actions
    ##
    actions: dict[str, ActionTermCfg] = {
        "joint_pos": JointPositionActionCfg(
            entity_name="robot",
            actuator_names=(".*",),
            scale=0.25,               # Placeholder — G1 override fills in G1_ACTION_SCALE
            use_default_offset=True,
        )
    }

    ##
    # Events
    ##
    events = {
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
                "velocity_range": {
                    # ← NEW: start with angular velocity (robot rocking)
                    "roll":  (-ANGULAR_VEL_RANGE, ANGULAR_VEL_RANGE),
                    "pitch": (-ANGULAR_VEL_RANGE, ANGULAR_VEL_RANGE),
                    "yaw":   (-0.5, 0.5),  # Smaller yaw rate
                },
            },
        ),
        "reset_robot_joints": EventTermCfg(
            func=mdp.reset_joints_by_offset,
            mode="reset",
            params={
                "position_range": (-JOINT_OFFSET_RANGE, JOINT_OFFSET_RANGE),
                "velocity_range": (-0.3, 0.3),  # ← NEW: joints also start with velocity
                "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
            },
        ),
        "push_robot": EventTermCfg(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(5.0, 7.0),     # More frequent than transition task
            params={
                "velocity_range": {
                    "x": (-0.3, 0.3),
                    "y": (-0.3, 0.3),
                    "z": (-0.1, 0.1),
                    "roll":  (-0.5, 0.5),   # Stronger angular push
                    "pitch": (-0.5, 0.5),
                    "yaw":   (-0.3, 0.3),
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

    ##
    # Rewards
    ##
    rewards = {
        # Primary: converge to neutral pose
        "pose_convergence": RewardTermCfg(
            func=mdp.pose_convergence,
            weight=2.0,
            params={"std": 0.25, "asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
        ),
        # NEW: converge angular velocity to zero
        # Weight 1.0 — secondary but important. std=0.5 rad/s — gradient from ±1 rad/s initial.
        "angular_velocity_convergence": RewardTermCfg(
            func=mdp.angular_velocity_convergence,
            weight=1.0,
            params={"std": 0.5, "asset_cfg": SceneEntityCfg("robot")},
        ),
        "both_feet_contact": RewardTermCfg(
            func=mdp.both_feet_contact,
            weight=0.5,
            params={"sensor_name": "feet_ground_contact"},
        ),
        "body_orientation_l2": RewardTermCfg(
            func=mdp.body_orientation_l2,
            weight=-2.0,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=())},
        ),
        # Higher weight than transition because the robot starts with momentum
        "joint_vel_penalty": RewardTermCfg(
            func=mdp.joint_vel_penalty,
            weight=-0.02,   # 2× the transition task
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
        ),
        "body_ang_vel": RewardTermCfg(
            func=mdp.body_angular_velocity_penalty,
            weight=-0.1,    # 2× the transition task (rocking is more dangerous here)
            params={"asset_cfg": SceneEntityCfg("robot", body_names=())},
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
    # Terminations
    ##
    terminations = {
        "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
        "fell_over": TerminationTermCfg(
            func=mdp.bad_orientation,
            params={"limit_angle": math.radians(65.0)},  # Tighter than transition (65° vs 70°)
        ),
    }

    ##
    # Metrics
    ##
    metrics = {
        "mean_action_acc": MetricsTermCfg(func=mdp.mean_action_acc),
    }

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
            body_name="",       # Set per-robot
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
        episode_length_s=8.0,    # Shorter: balance recovery is faster
    )
```

---

## 15.6 Step 5: Write the G1 Override Config

```python
# src/tasks/balance_recovery/config/g1/env_cfgs.py

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg

from src.assets.robots import G1_ACTION_SCALE, get_g1_robot_cfg
from src.tasks.velocity.mdp.rewards import self_collision_cost
from src.tasks.balance_recovery.balance_recovery_env_cfg import make_balance_recovery_env_cfg


def unitree_g1_balance_recovery_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = make_balance_recovery_env_cfg()

    cfg.sim.contact_sensor_maxmatch = 64

    cfg.scene.entities = {"robot": get_g1_robot_cfg()}

    geom_names = tuple(
        f"{side}_foot{i}_collision"
        for side in ("left", "right")
        for i in range(1, 8)
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

    # Wire contact sensors
    cfg.observations["critic"].terms["foot_contact"].params["sensor_name"] = "feet_ground_contact"
    cfg.observations["critic"].terms["foot_contact_forces"].params["sensor_name"] = "feet_ground_contact"
    cfg.rewards["both_feet_contact"].params["sensor_name"] = "feet_ground_contact"

    # Domain randomization targets
    cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names
    cfg.events["base_com"].params["asset_cfg"].body_names = ("torso_link",)

    # Body references for rewards
    cfg.rewards["body_orientation_l2"].params["asset_cfg"].body_names = ("torso_link",)
    cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("torso_link",)

    # Self-collision penalty
    cfg.rewards["self_collisions"] = RewardTermCfg(
        func=self_collision_cost,
        weight=-1.0,
        params={"sensor_name": "self_collision", "force_threshold": 10.0},
    )

    if play:
        cfg.episode_length_s = int(1e9)
        cfg.observations["actor"].enable_corruption = False
        cfg.events.pop("push_robot", None)

    return cfg
```

---

## 15.7 Step 6: Write the RL Config

```python
# src/tasks/balance_recovery/config/g1/rl_cfg.py

from mjlab.rl import (
    RslRlModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)


def unitree_g1_balance_recovery_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    return RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=True,
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        ),
        critic=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=True,
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.01,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1.0e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
        experiment_name="g1_balance_recovery",
        save_interval=100,
        num_steps_per_env=24,
        max_iterations=10001,
    )
```

---

## 15.8 Step 7: Register the Task

```python
# src/tasks/balance_recovery/config/g1/__init__.py

from mjlab.tasks.registry import register_mjlab_task
from src.tasks.transition.rl import TransitionOnPolicyRunner  # Reuse this runner

from .env_cfgs import unitree_g1_balance_recovery_env_cfg
from .rl_cfg import unitree_g1_balance_recovery_ppo_runner_cfg

register_mjlab_task(
    task_id="Unitree-G1-BalanceRecovery",
    env_cfg=unitree_g1_balance_recovery_env_cfg(),
    play_env_cfg=unitree_g1_balance_recovery_env_cfg(play=True),
    rl_cfg=unitree_g1_balance_recovery_ppo_runner_cfg(),
    runner_cls=TransitionOnPolicyRunner,
)
```

---

## 15.9 Step 8: Add Missing `__init__.py` Files

Create empty or minimal `__init__.py` files:

```python
# src/tasks/balance_recovery/__init__.py
"""Balance recovery task for legged robots."""

# src/tasks/balance_recovery/config/__init__.py
(empty)

# src/tasks/balance_recovery/config/g1/__init__.py
# (already written above — contains registration)

# src/tasks/balance_recovery/rl/__init__.py
# (not needed if we reuse TransitionOnPolicyRunner — just import directly in config/__init__.py)

# src/tasks/balance_recovery/mdp/__init__.py
# (written in step 4)
```

---

## 15.10 Step 9: Verify the Task is Discovered

```bash
uv run python scripts/list_envs.py
```

Expected output should include `Unitree-G1-BalanceRecovery`.

If it does not appear:
1. Check `src/tasks/balance_recovery/config/g1/__init__.py` has `register_mjlab_task`.
2. Check that no directory in the path is named `utils` or `mdp` (blacklisted).
3. Check for import errors: `python -c "import src.tasks"` and read the traceback.

---

## 15.11 Step 10: Train

```bash
uv run python scripts/train.py Unitree-G1-BalanceRecovery \
    --env.scene.num-envs 4096 \
    --agent.gpu-ids 0
```

---

## 15.12 Debugging Checklist

**Reward is always zero:**
- Print `env.scene.keys()` to see what sensors/entities exist.
- Print `asset_cfg.joint_ids` to see which joints are selected.
- Add `print` statements inside the reward function and run with `num_envs=1`.

**Robot falls immediately:**
- Reduce `ANGULAR_VEL_RANGE` in the reset (start with less aggressive initial velocity).
- Check action scale (print `cfg.actions["joint_pos"].scale` — should be a dict with values ≈0.2-0.5).
- Reduce `JOINT_OFFSET_RANGE`.

**Policy learns to not move:**
- The `joint_vel_penalty` weight may be too large. Reduce from -0.02 to -0.005.
- The `body_ang_vel` weight may be overwhelming the primary `pose_convergence`.
- Check reward magnitudes in training logs — they should be roughly comparable.

**Slow convergence after 5000 iterations:**
- Is `pose_convergence` still < 0.5? The policy is struggling to reduce joint error.
- Increase `std` from 0.25 to 0.4 (provides gradient signal from farther away).
- Or add a curriculum: start with `JOINT_OFFSET_RANGE=0.2` and increase to 0.4.

**ONNX export fails:**
- Make sure you are using `TransitionOnPolicyRunner` or another runner with ONNX export.
- Check that `attach_metadata_to_onnx` is called after `export_policy_to_onnx`.

---

## 15.13 Quick Reference: Configuration Decisions

| Decision | Where to change it | Notes |
|----------|-------------------|-------|
| Initial joint offset range | `JOINT_OFFSET_RANGE` in base cfg | ±0.5 rad = hard, ±0.25 = medium |
| Initial angular velocity | `velocity_range` in `reset_base` event | New for balance recovery |
| Episode length | `episode_length_s` in `ManagerBasedRlEnvCfg` | Shorter = more resets |
| Primary reward std | `std` in `pose_convergence` params | 0.25 for strict, 0.4 for forgiving |
| Termination angle | `limit_angle` in `bad_orientation` params | Degrees to radians via math.radians |
| Action scale | `G1_ACTION_SCALE` or manual dict | Must match motor specs |
| Network size | `hidden_dims` in `RslRlModelCfg` | (512,256,128) for full body |
| Number of envs | `--env.scene.num-envs` CLI flag | 4096 for GPU training |
| PPO clip | `clip_param` in `RslRlPpoAlgorithmCfg` | 0.2 standard, 0.1 if unstable |
| Body for viewer | `cfg.viewer.body_name` | Must be a body name in the MJCF |
| Contact sensor pattern | `pattern` in `ContactMatch` | Regex matched against body names |
