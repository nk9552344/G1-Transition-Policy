# Chapter 3: The mjlab Framework and Every Import Explained

Every `import` in this codebase is load-bearing. If you do not know what a class does, you
cannot know what parameters it accepts, and you will spend hours chasing runtime errors.
This chapter explains every import used in the transition policy, grouped by module.

---

## 3.1 `from mjlab.envs import ManagerBasedRlEnvCfg`

This is the **root configuration object** for the entire environment. It is a pure dataclass —
it holds references to sub-configurations but does not create any simulation state. The
environment is only materialized when `ManagerBasedRlEnv(cfg=..., device=...)` is called.

```python
@dataclass
class ManagerBasedRlEnvCfg:
    scene:        SceneCfg                      # What entities and sensors exist
    observations: dict[str, ObservationGroupCfg]# What the policy sees
    actions:      dict[str, ActionTermCfg]       # How the policy acts
    commands:     dict[str, CommandTermCfg]      # Command generators (velocity, etc.)
    events:       dict[str, EventTermCfg]        # What fires on reset/interval
    rewards:      dict[str, RewardTermCfg]       # Scalar rewards summed each step
    terminations: dict[str, TerminationTermCfg]  # When episodes end
    curriculum:   dict[str, CurriculumTermCfg]   # Adaptive training difficulty
    metrics:      dict[str, MetricsTermCfg]      # Logged performance metrics
    viewer:       ViewerConfig                   # Camera position and tracking target
    sim:          SimulationCfg                  # Physics parameters
    decimation:   int                            # Physics sub-steps per policy step
    episode_length_s: float                      # Episode time in seconds
    seed:         int = 0                        # RNG seed
```

The key insight: this is a **schema**, not code. You declare what you want; the framework
instantiates it. This makes it easy to mix and match components.

---

## 3.2 `from mjlab.envs import mdp as envs_mdp`

The `mjlab.envs.mdp` module is a library of pre-built MDP components. It covers the most
common operations you will need:

**Observation functions in `mjlab.envs.mdp`:**
- `builtin_sensor(env, sensor_name)` — reads any sensor registered on the scene by its `name` field
- `projected_gravity(env)` — returns gravity vector rotated into the robot's body frame `[B, 3]`
- `joint_pos_rel(env)` — returns `current_pos - default_pos` for all controlled joints `[B, N]`
- `joint_vel_rel(env)` — returns `current_vel - default_vel` (usually zero default) `[B, N]`
- `last_action(env)` — returns the previous action tensor `[B, N]`
- `generated_commands(env, command_name)` — returns the current command from the command manager

**Event functions in `mjlab.envs.mdp`:**
- `reset_root_state_uniform(env, pose_range, velocity_range)` — randomize root position/velocity
- `reset_joints_by_offset(env, position_range, velocity_range, asset_cfg)` — add noise to joints
- `push_by_setting_velocity(env, velocity_range)` — apply an instantaneous velocity impulse
- `randomize_terrain(env)` — sample a new terrain tile on reset (used in play mode)

**Reward functions in `mjlab.envs.mdp`:**
- `is_terminated(env)` — returns 1.0 for environments that terminated this step
- `joint_acc_l2(env)` — L2 norm of joint accelerations
- `joint_pos_limits(env)` — penalty for approaching hardware joint limits
- `action_rate_l2(env)` — L2 norm of (current_action - previous_action)
- `time_out(env)` — fires when episode_length_buf >= max_episode_length
- `bad_orientation(env, limit_angle)` — fires when the torso tilts beyond limit_angle radians

The star-import in `transition/mdp/__init__.py`:
```python
from mjlab.envs.mdp import *
```
brings all of these into the `mdp` namespace so `transition_env_cfg.py` can write
`mdp.is_terminated`, `mdp.joint_acc_l2`, etc.

---

## 3.3 `from mjlab.envs.mdp.actions import JointPositionActionCfg`

This configures how the policy's output is interpreted as a command to the robot's joints.

```python
@dataclass
class JointPositionActionCfg(ActionTermCfg):
    entity_name:        str          # Which entity in the scene to control ("robot")
    actuator_names:     tuple[str]   # Regex patterns to select actuators (".*" = all)
    scale:              float | dict # Multiplier applied to network output
    use_default_offset: bool         # If True: target = default_pos + action * scale
                                     # If False: target = action * scale (absolute)
```

**Critical detail — `use_default_offset=True`:**
The network outputs values in roughly `[-1, 1]`. These are multiplied by `scale`, then added
to `default_joint_pos`. The result is a **desired joint position**. The PD controller then
drives the robot toward that position.

If `use_default_offset=False`, the network would be outputting absolute joint positions,
which is much harder to learn because the output space would need to cover the full joint
range (which varies per joint).

With `use_default_offset=True`, the network only needs to learn **deviations from the natural
standing pose**, which is a much simpler function to approximate.

---

## 3.4 `from mjlab.managers.action_manager import ActionTermCfg`

The base class for all action configurations. It defines the minimal interface:
- `func` — the function that applies the action to the simulation
- `params` — additional parameters passed to the function

You rarely need to use `ActionTermCfg` directly; use `JointPositionActionCfg` instead.

---

## 3.5 `from mjlab.managers.event_manager import EventTermCfg`

Events are functions that execute at specific moments in the episode lifecycle.

```python
@dataclass
class EventTermCfg:
    func:           Callable          # The event function to call
    mode:           str               # "startup" | "reset" | "interval"
    interval_range_s: tuple[float,float] | None  # Only for mode="interval"
    params:         dict              # Keyword arguments passed to func
```

**Modes:**
- `"startup"` — fires once when the environment is first created. Used for domain
  randomization that should be applied once per training run (but see below).
- `"reset"` — fires every time an environment episode resets. Used for joint/base
  randomization that varies per episode.
- `"interval"` — fires at a random interval within `interval_range_s` seconds during an
  episode. Used for disturbance injection (push_robot).

**Warning about `"startup"` mode:** In practice, startup events may fire once per worker
process in a multi-GPU setup, and friction/CoM randomization may not persist across the
training run as you might expect. If you need per-environment randomization that changes
every episode, use `"reset"` mode instead.

---

## 3.6 `from mjlab.managers.metrics_manager import MetricsTermCfg`

Metrics are computed alongside rewards but their values are only logged — they do not
affect the policy gradient.

```python
@dataclass
class MetricsTermCfg:
    func: Callable  # Returns a scalar tensor [B] or [B, 1]
    params: dict    # Additional parameters
```

Example: `mean_action_acc` measures how jerky the actions are. If this metric grows during
training, the policy is becoming more erratic — a warning sign even if the reward is
increasing.

---

## 3.7 `from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg`

```python
@dataclass
class ObservationTermCfg:
    func:   Callable           # Returns tensor [B, D]
    params: dict               # Additional keyword args to func
    noise:  NoiseCfg | None    # Optional noise to add during training
    clip:   tuple | None       # Optional clamp range after noise
    scale:  float | None       # Optional scaling after noise
```

```python
@dataclass
class ObservationGroupCfg:
    terms:            dict[str, ObservationTermCfg]
    concatenate_terms: bool   # True: all terms concatenated into [B, sum_D]
    enable_corruption: bool   # True: apply noise during training
    history_length:   int     # > 1: stack N past observations (creates [B, N*D])
```

**What happens at each step (if enable_corruption=True):**
1. Call each term's `func` to get clean observations
2. Add noise sampled from the term's `noise` distribution
3. Optionally clip to the `clip` range
4. Optionally multiply by `scale`
5. Concatenate all terms along dim=1

The actor group has `enable_corruption=True` to simulate real-world sensor noise.
The critic group has `enable_corruption=False` because we want a clean value estimate.

---

## 3.8 `from mjlab.managers.reward_manager import RewardTermCfg`

```python
@dataclass
class RewardTermCfg:
    func:   Callable  # Returns tensor [B] (raw value before weighting)
    weight: float     # Multiplied by func's output at each step
    params: dict      # Additional keyword args to func
```

The total reward at each step is:
```
total_reward[b] = sum(term.weight * term.func(env, **term.params) for term in rewards)
```

The `weight` has units of "reward per unit of what the function returns." For example,
`pose_convergence` returns a value in `[0, 1]`, so `weight=2.0` means the maximum possible
contribution of this term is `+2.0` per step.

Choosing weights is covered thoroughly in Chapter 9.

---

## 3.9 `from mjlab.managers.scene_entity_config import SceneEntityCfg`

`SceneEntityCfg` is used to pass **entity and joint/body selection** to MDP functions.
It is not a full configuration — it is a selector.

```python
@dataclass
class SceneEntityCfg:
    name:        str              # Entity name in the scene dict ("robot")
    joint_names: str | list[str]  # Regex patterns for joints (".*" = all)
    body_names:  str | list[str]  # Regex patterns for bodies ("torso_link")
    site_names:  str | list[str]  # Regex patterns for MuJoCo sites
    geom_names:  str | list[str]  # Regex patterns for collision geoms
```

At runtime, `SceneEntityCfg.joint_ids` is resolved: the framework looks up which joint
indices match the `joint_names` pattern and caches the result. You never set `joint_ids`
manually — it is populated by the framework.

**The placeholder pattern:** In `transition_env_cfg.py`, several `SceneEntityCfg` instances
have empty `body_names=()` or `geom_names=()`. This is intentional — they are placeholders
that the robot-specific override fills in:

```python
# In transition_env_cfg.py (base layer):
cfg.rewards["body_orientation_l2"].params["asset_cfg"].body_names = ()  # placeholder

# In config/g1/env_cfgs.py (G1 override):
cfg.rewards["body_orientation_l2"].params["asset_cfg"].body_names = ("torso_link",)
```

---

## 3.10 `from mjlab.managers.termination_manager import TerminationTermCfg`

```python
@dataclass
class TerminationTermCfg:
    func:     Callable  # Returns bool tensor [B]
    time_out: bool      # If True: this termination is a timeout (not a failure)
    params:   dict      # Additional keyword args to func
```

Terminations that are timeouts (`time_out=True`) are treated differently by PPO: the episode
ends but the bootstrapped value at the terminal state is **not penalized**. This is important
— without this flag, PPO would treat an episode that ran out of time identically to one
where the robot fell, which would create a bias against longer episodes.

---

## 3.11 `from mjlab.scene import SceneCfg`

```python
@dataclass
class SceneCfg:
    terrain:  TerrainEntityCfg        # The ground mesh or plane
    entities: dict[str, EntityCfg]    # Physical entities (robot, obstacles)
    sensors:  tuple[SensorCfg]        # Sensors not attached to a specific entity
    num_envs: int                     # Number of parallel environments
    extent:   float                   # Spacing between environments (meters)
```

`extent` controls how far apart the parallel environments are spawned. For the transition
task, `extent=2.0` means each environment occupies a 2m × 2m tile. With 4096 environments,
the total scene is roughly 82m × 82m.

---

## 3.12 `from mjlab.sim import MujocoCfg, SimulationCfg`

```python
@dataclass
class SimulationCfg:
    mujoco:                MujocoCfg  # MuJoCo solver parameters
    nconmax:               int | None # Max number of contact pairs
    njmax:                 int        # Max number of contact constraints
    contact_sensor_maxmatch: int      # Max contacts per sensor slot
    nan_guard:             NanGuardCfg# Debug option for NaN detection

@dataclass
class MujocoCfg:
    timestep:    float  # Simulation timestep in seconds (0.005 = 200 Hz)
    iterations:  int    # Number of constraint solver iterations
    ls_iterations: int  # Number of line-search iterations in solver
    ccd_iterations: int # Continuous collision detection iterations
```

**Why these values matter:**
- `timestep=0.005`: Too large → physics becomes unstable. Too small → training is slow.
  5 ms is well-validated for G1-class humanoids.
- `iterations=10`: The constraint solver converges faster with more iterations but each step
  takes longer. 10 is a balanced default for flat terrain.
- `ccd_iterations=50`: Continuous collision detection catches fast-moving contacts that would
  otherwise tunnel through geometry. 50 is sufficient for walking/standing speeds.
- `njmax=300`: Maximum simultaneous contact constraints. The G1 with 7 foot geoms per foot
  can generate up to ~14 contacts at once. 300 is conservative — reduce if memory is tight.

---

## 3.13 `from mjlab.terrains import TerrainEntityCfg`

```python
TerrainEntityCfg(terrain_type="plane")  # Flat infinite plane
TerrainEntityCfg(terrain_type="generator", cfg=ROUGH_TERRAINS_CFG)  # Heightfield
```

For the transition policy, `terrain_type="plane"` is correct because:
- The robot is not walking — it is standing up.
- Rough terrain would add unnecessary complexity to the reward signal.
- Flat terrain makes both_feet_contact reliable (no tilted surfaces).

---

## 3.14 `from mjlab.utils.noise import UniformNoiseCfg as Unoise`

```python
@dataclass
class UniformNoiseCfg:
    n_min: float  # Lower bound of uniform noise
    n_max: float  # Upper bound of uniform noise
```

Noise is added to each element of the observation independently. The noise draws are
fresh at every step. This means:
- The policy must be robust to bounded sensor errors.
- The policy cannot rely on precise cancellation of noise across consecutive steps
  (because the noise is i.i.d., not correlated).

Common noise magnitudes used in this codebase:

| Sensor | Noise | Rationale |
|--------|-------|-----------|
| IMU angular velocity | ±0.2 rad/s | Real IMU gyro noise floor |
| Projected gravity | ±0.05 | Attitude estimation error |
| Joint position | ±0.01 rad | Encoder quantization |
| Joint velocity | ±1.5 rad/s | Velocity computed from finite difference of position |
| IMU linear velocity | ±0.5 m/s | Linear velocity is not directly sensed, estimated |

---

## 3.15 `from mjlab.viewer import ViewerConfig`

```python
@dataclass
class ViewerConfig:
    origin_type: OriginType  # ASSET_BODY: camera follows a specific body
    entity_name: str         # Which entity to track ("robot")
    body_name:   str         # Which body on the entity to track ("torso_link")
    distance:    float       # Camera distance from target (meters)
    elevation:   float       # Camera elevation angle (degrees, negative = looking up)
    azimuth:     float       # Camera azimuth angle (degrees from front)
```

The `body_name=""` placeholder in `transition_env_cfg.py` is filled in by the robot-specific
override: `cfg.viewer.body_name = "torso_link"`.

---

## 3.16 `from mjlab.sensor import ContactSensorCfg, ContactMatch`

These are covered in full detail in Chapter 10. The key types:

```python
ContactSensorCfg  # Declares a contact sensor (which bodies, which contacts to track)
ContactMatch      # Specifies how to match bodies for contact detection
ContactSensor     # The runtime object (accessed via env.scene["sensor_name"])
```

---

## 3.17 `from mjlab.tasks.registry import register_mjlab_task`

```python
register_mjlab_task(
    task_id:    str,                    # Unique string identifier
    env_cfg:    ManagerBasedRlEnvCfg,   # Training configuration
    play_env_cfg: ManagerBasedRlEnvCfg, # Evaluation configuration
    rl_cfg:     RslRlOnPolicyRunnerCfg, # PPO configuration
    runner_cls: type,                   # Runner class for training
)
```

This function stores the task in a global dict keyed by `task_id`. When you run
`scripts/train.py Unitree-G1-Transition`, the registry looks up `"Unitree-G1-Transition"`,
retrieves the stored `env_cfg` and `rl_cfg`, and passes them to the runner.

---

## 3.18 `from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg`

These configure the PPO learning algorithm. See Chapter 12 for full parameter explanations.

```python
RslRlModelCfg          # Neural network architecture (hidden dims, activation)
RslRlPpoAlgorithmCfg   # PPO hyperparameters (clip, learning rate, etc.)
RslRlOnPolicyRunnerCfg # Bundles actor, critic, algorithm configs
```

---

## 3.19 `from src.assets.robots import G1_ACTION_SCALE, get_g1_robot_cfg`

- `G1_ACTION_SCALE`: A `dict[str, float]` mapping actuator regex patterns to scale values.
  The keys are the same patterns used in `BuiltinPositionActuatorCfg.target_names_expr`.
  The values are derived from motor specs. See Chapter 4.

- `get_g1_robot_cfg()`: Returns a fresh `EntityCfg` instance each call. Important: you must
  call this function (not cache its return value) to avoid mutation issues across configs.

---

## 3.20 The `if TYPE_CHECKING:` Pattern

Throughout the reward functions you will see:

```python
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv
```

This is a standard Python pattern to avoid circular imports. `TYPE_CHECKING` is `False` at
runtime, so `ManagerBasedRlEnv` is never actually imported. The string annotation
`"ManagerBasedRlEnv"` in function signatures is resolved lazily by type checkers only.

The reason it's needed: `rewards.py` is in the `mdp` subpackage, and `ManagerBasedRlEnv`
imports the managers which import the MDP. Importing `ManagerBasedRlEnv` directly in
`rewards.py` would create a circular dependency chain.
