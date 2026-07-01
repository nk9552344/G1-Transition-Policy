# Chapter 8: Events and Domain Randomization

Events are the mechanism for injecting variability into training. They include:
1. **Episode resets** — randomize the initial state every episode
2. **Interval disturbances** — apply forces or perturbations mid-episode
3. **Domain randomization** — change physics parameters to train robust policies

---

## 8.1 EventTermCfg Modes in Detail

### `mode="startup"`

Fires once when the environment is created (the very first call to any environment function).
Used for physics parameter randomization that should vary across training runs but not across
episodes.

```python
"foot_friction": EventTermCfg(
    mode="startup",
    func=dr.geom_friction,
    params={...},
)
```

**Important caveat:** "Startup" in a multi-GPU training context means "once per worker
process." With 4 GPU workers, there will be 4 independent randomizations (one per process).
This is usually fine — it creates diverse physics across the parallel workers.

### `mode="reset"`

Fires every time an environment episode ends (the robot fell or timed out).

```python
"reset_robot_joints": EventTermCfg(
    mode="reset",
    func=mdp.reset_joints_by_offset,
    params={"position_range": (-0.5, 0.5), ...},
)
```

Reset events are the primary source of training diversity. Every episode starts from a
different random state, forcing the policy to generalize.

**Order of reset events:** Multiple reset events may fire simultaneously. The framework
processes them in dict insertion order. The `reset_base` fires before `reset_robot_joints`
in the transition config, which is correct: base position is set first, then joint positions.

### `mode="interval"`

Fires at random intervals during the episode, independently per environment. The interval
is sampled uniformly from `interval_range_s` at the start of each interval.

```python
"push_robot": EventTermCfg(
    mode="interval",
    interval_range_s=(8.0, 10.0),
    func=mdp.push_by_setting_velocity,
    params={...},
)
```

This means: sometime between 8 and 10 seconds after the last push (or since episode start),
apply a velocity impulse. Not all environments receive the push at the same time — the
interval is sampled independently, creating temporal diversity.

---

## 8.2 `reset_root_state_uniform`

```python
"reset_base": EventTermCfg(
    func=mdp.reset_root_state_uniform,
    mode="reset",
    params={
        "pose_range": {
            "x": (-0.5, 0.5),   # ±0.5 m in x
            "y": (-0.5, 0.5),   # ±0.5 m in y
            "z": (0.0, 0.0),    # Fixed height (0 relative to terrain)
            "yaw": (-math.pi, math.pi),  # Full 360° yaw randomization
        },
        "velocity_range": {},   # Empty: zero linear and angular velocity at reset
    },
)
```

**`pose_range`:** Specifies uniform bounds for each pose component relative to the
terrain height. `"z": (0.0, 0.0)` means the base Z is always at terrain height + the
robot's natural height (from HOME_KEYFRAME's `pos=(0, 0, 0.8)`).

**`velocity_range`: empty dict** means zero velocities. The robot starts standing still.

**Why randomize position:** In multi-environment training, the robots are already spread
across a grid. But within each grid cell, the robot might drift over episodes. Position
randomization also prevents the policy from learning position-dependent behaviors (since
the terrain is always the same flat plane).

**Why randomize yaw:** The policy must be invariant to heading direction. By randomizing
yaw ∈ [-π, π], the policy sees the robot facing every direction equally during training.
If yaw were fixed, the policy might learn to exploit specific heading-gravity interactions.

**Pitch and roll:** These are not randomized. The robot always starts upright. Randomizing
initial pitch/roll would require careful handling to ensure the robot is not immediately
falling, which would make the `is_terminated` reward fire constantly early in training.

---

## 8.3 `reset_joints_by_offset` — The Core of the Transition Task

```python
"reset_robot_joints": EventTermCfg(
    func=mdp.reset_joints_by_offset,
    mode="reset",
    params={
        "position_range": (-JOINT_OFFSET_RANGE, JOINT_OFFSET_RANGE),  # (-0.5, 0.5) rad
        "velocity_range": (0.0, 0.0),                                  # zero velocity
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),     # all joints
    },
)
```

**What this does:**
```
for each joint j in selected joints:
    offset_j = uniform(-0.5, 0.5)      # Independent per joint
    joint_pos_j = q_default_j + offset_j  # Start from home + random offset
    joint_vel_j = 0.0                  # Always start at rest
```

**Why independent offsets matter:** Each joint independently draws its offset. With 29
joints, the probability that all joints simultaneously draw values near zero (a "trivial
reset") is `(2 * 0.05 / 1.0)^29 ≈ 0` — essentially impossible. Every episode starts with
a meaningfully challenging initial configuration.

**`JOINT_OFFSET_RANGE = 0.5 rad`:** This value controls the difficulty of the task.

| Offset Range | Difficulty | Considerations |
|-------------|------------|----------------|
| 0.1 rad | Easy | Robot barely displaced, fast convergence, may not generalize |
| 0.25 rad | Medium | Good balance for initial training |
| 0.5 rad | Hard | Max displacement close to hardware limits for wrists/fingers |
| 1.0 rad | Too Hard | Some joints may violate limits immediately |

At 0.5 rad, the worst-case initial pose has every joint offset by 0.5 rad from home.
For the knee (default 0.3 rad), this means the knee could be fully extended (−0.2 rad)
or deeply bent (0.8 rad). The policy must handle both extremes.

**Why velocity=0:** Starting with zero velocity means the robot is in a static (but
displaced) standing configuration. If we also randomized velocity, the robot might start
with it already falling — this would create an immediate termination before the policy
can do anything, producing zero-signal gradients at the start of training.

---

## 8.4 `push_by_setting_velocity`

```python
"push_robot": EventTermCfg(
    func=mdp.push_by_setting_velocity,
    mode="interval",
    interval_range_s=(8.0, 10.0),
    params={
        "velocity_range": {
            "x": (-0.3, 0.3),    # ±0.3 m/s linear push in x
            "y": (-0.3, 0.3),    # ±0.3 m/s linear push in y
            "z": (-0.2, 0.2),    # Small vertical component
            "roll": (-0.3, 0.3), # Angular push components
            "pitch": (-0.3, 0.3),
            "yaw": (-0.5, 0.5),  # Larger yaw disturbance
        },
    },
)
```

**Mechanism:** Rather than applying a force, this event **directly sets the root body
velocity** to a value within the specified ranges. This is equivalent to an instantaneous
impulse — the robot suddenly has a new velocity and must recover.

**Why set velocity instead of applying force:** Applying a force would require knowing the
robot's mass and the desired impulse magnitude precisely. Setting velocity directly gives
predictable disturbance magnitudes regardless of the robot's inertia.

**Transition vs Velocity task push magnitude:**
- Transition task: ±0.3 m/s, ±0.3 rad/s
- Velocity task: ±0.5 m/s, ±0.52 rad/s

The transition task uses lighter pushes because:
1. The robot is stationary — it has no angular momentum to help it recover.
2. The primary goal is convergence to neutral, not disturbance rejection.
3. A heavy push at a late episode time (8-10 s in) can make the robot fall just before
   a successful episode would have ended, wasting the learning signal.

**Timing at 8-10 s (out of 15 s episodes):** This means the push fires roughly in the
second half of the episode. By then, a learning policy should have mostly converged to
neutral. The push tests whether the policy can recover from a disturbance at neutral —
a more realistic deployment scenario than just converging from initial offset.

---

## 8.5 Domain Randomization: `foot_friction`

```python
"foot_friction": EventTermCfg(
    mode="startup",
    func=dr.geom_friction,
    params={
        "asset_cfg": SceneEntityCfg("robot", geom_names=()),  # Set per-robot!
        "operation": "abs",        # Set friction to exactly the drawn value
        "ranges": (0.3, 1.6),      # Friction coefficient range
        "shared_random": True,     # All geoms in the asset get the same value
    },
)
```

In the G1 override:
```python
cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names
# where: geom_names = ("left_foot1_collision", ..., "right_foot7_collision")
```

**`operation="abs"`:** The drawn friction coefficient replaces the MJCF default. Alternative:
`operation="add"` would add to the default; `operation="mul"` would multiply.

**`ranges=(0.3, 1.6)`:** The friction coefficient is drawn from `U(0.3, 1.6)`.
- 0.3 = slippery floor (wet tile, polished wood)
- 0.8 = typical indoor floor
- 1.6 = rubberized floor or carpet

A policy trained across this range learns to work on any surface. If you only train at 0.8,
the policy may oscillate on slippery floors (insufficient friction) or move inefficiently on
sticky floors (excessive friction compensation).

**`shared_random=True`:** All 14 foot geoms get the same friction value (drawn once per
robot). This matches reality: the floor surface is the same under both feet. If
`shared_random=False`, each geom would get an independent random value, creating unrealistic
per-geom friction variation.

---

## 8.6 Domain Randomization: `encoder_bias`

```python
"encoder_bias": EventTermCfg(
    mode="startup",
    func=dr.encoder_bias,
    params={
        "asset_cfg": SceneEntityCfg("robot"),
        "bias_range": (-0.015, 0.015),  # ±0.015 rad constant bias per joint
    },
)
```

The encoder bias is a **fixed offset** added to every joint position reading for the duration
of training (startup mode). On a real robot, joint encoders have calibration offsets that
cause the reported position to be `true_position + bias`.

The effect: the policy receives `joint_pos_rel = (q + bias) - q_default` instead of the
true error. The policy must learn to be robust to small constant biases in its position
feedback.

**±0.015 rad (≈ 0.86°):** This is consistent with typical BLDC motor encoder calibration
accuracy on the G1.

---

## 8.7 Domain Randomization: `base_com`

```python
"base_com": EventTermCfg(
    mode="startup",
    func=dr.body_com_offset,
    params={
        "asset_cfg": SceneEntityCfg("robot", body_names=()),  # Set per-robot
        "operation": "add",
        "ranges": {0: (-0.05, 0.05), 1: (-0.05, 0.05), 2: (-0.05, 0.05)},
    },
)
```

In the G1 override:
```python
cfg.events["base_com"].params["asset_cfg"].body_names = ("torso_link",)
```

This shifts the center of mass (CoM) of the torso link by a random offset in each axis
(±5 cm). This simulates:
- Payload uncertainty (carrying objects)
- Internal component placement variation
- Mass model inaccuracies in the MJCF

A ±5 cm CoM shift changes the balance point significantly. A policy that is not robust to
this will lean or fall when the real robot's mass distribution differs from the simulation.

**`body_names=("torso_link",)`:** Only the torso CoM is randomized. Randomizing every link
independently would make the total body mass distribution wildly unrealistic. The torso is
chosen because it has the largest mass and contributes most to the balance dynamics.

---

## 8.8 Writing a Custom Event Function

```python
def my_custom_reset(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,    # Shape [K] — which environments are being reset
    my_param: float = 0.1,
) -> None:                    # Must return None
    asset: Entity = env.scene["robot"]
    # ... modify asset.data or other simulation state ...
    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
```

**Critical — `env_ids`:** Event functions receive `env_ids`, a tensor of environment indices
that are being reset on this step. Only modify state for those environments. If you write
state for all environments, you will corrupt environments that are mid-episode.

**No return value:** Event functions must return `None`. They modify simulation state as a
side effect via the entity's write methods.

**Registering it:**
```python
events["my_reset"] = EventTermCfg(
    func=my_custom_reset,
    mode="reset",
    params={"my_param": 0.2},
)
```
