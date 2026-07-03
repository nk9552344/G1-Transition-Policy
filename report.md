# Technical Report: Transition Policy for the Unitree G1 Humanoid Robot

**Repository:** `unitree_rl_mjlab`  
**Author:** Neeraj (neeraj11.trn@infosys.com)  
**Date:** July 2, 2026  

---

## Abstract

This report presents the complete technical design of a reinforcement learning-based transition policy for the Unitree G1 humanoid robot. The policy trains the robot to recover from arbitrary standing perturbations—random joint displacements of up to ±0.5 radians and, in the extended version, residual body momentum—and return to a stable neutral upright stance. We describe the problem formulation, environment architecture, observation and action spaces, reward function design, domain randomization strategy, and neural network configuration. We compare the transition task against the existing locomotion (velocity-tracking) policy, explain how the transition policy draws inspiration from locomotion policy design, and document the extended transition-v2 formulation, which introduces momentum-awareness. All implementations use the MuJoCo physics engine, the `mjlab` manager-based RL environment framework, and the RSL-RL PPO algorithm.

---

## 1. Introduction and Motivation

A humanoid robot operating in the real world must handle more than forward locomotion. After stopping, after being pushed, or after executing a motion that displaces its configuration, the robot needs a dedicated controller to settle back into a safe, stable neutral posture. This problem—transitioning from an arbitrary upright-but-displaced standing configuration back to a default neutral stance—is distinct from locomotion and requires its own policy.

### 1.1 Problem Statement

The transition task is formally defined as follows: Given the G1 humanoid robot in a standing-upright configuration with all joints displaced from their default positions by independent random offsets drawn uniformly from U(−0.5, +0.5) radians, train a neural network policy that drives all joints back to the robot's home keyframe (HOME\_KEYFRAME) and holds them there for the remainder of the episode, while maintaining a stable two-foot stance and an upright torso.

### 1.2 Why a Separate Policy?

The existing locomotion policy (velocity-tracking task, `Unitree-G1-Rough`) handles walking and running on rough terrain. It produces commanded velocities and uses gait rewards to produce a walking motion. It is not designed to:

- Bring the robot to a stationary rest at a specific joint configuration
- Converge to a precise target pose from a displaced start
- Handle the standing/holding problem without a gait command

The transition policy fills this gap. It is intended to run after the locomotion policy decelerates to a stop, or after any motion sequence that leaves the robot in a non-neutral standing configuration.

---

## 2. Technical Foundation: Simulation Framework

### 2.1 Physics Simulation — MuJoCo

All training and validation uses the MuJoCo physics engine (via the `mjlab` framework). The key simulation parameters are:

| Parameter | Value | Effect |
|-----------|-------|--------|
| `timestep` | 0.005 s (200 Hz) | Physics integration rate |
| `decimation` | 4 | Policy runs at 50 Hz (every 4 substeps) |
| `iterations` | 10 | Newton solver iterations per physics step |
| `ls_iterations` | 20 | Line-search iterations |
| `ccd_iterations` | 50 | Continuous collision detection iterations |
| `njmax` | 300 | Maximum contact pairs |
| `integrator` | `implicitfast` | Semi-implicit Euler for stability |

The control loop runs at 50 Hz (policy step dt = 20 ms), while the physics solver runs at 200 Hz internally. This 4× decimation allows the neural network to run cheaply while MuJoCo handles high-frequency contact dynamics.

### 2.2 The mjlab Environment Framework

`mjlab` is a manager-based RL environment framework modeled after Isaac Lab but built on MuJoCo. It organizes the RL environment into six managers:

- **ObservationManager**: computes and noise-corrupts actor/critic observations
- **ActionManager**: converts policy outputs to joint position targets via PD controllers
- **EventManager**: fires reset and domain-randomization events on schedule
- **RewardManager**: evaluates all reward terms and sums them
- **TerminationManager**: evaluates termination conditions
- **CurriculumManager**: adjusts task difficulty based on agent progress

Each manager is configured with dataclasses (`ObservationTermCfg`, `RewardTermCfg`, etc.) that declare function pointers and parameters. The environment is assembled by the factory function `make_transition_env_cfg()`, which returns a `ManagerBasedRlEnvCfg` object.

### 2.3 The Inner Loop (One Environment Step)

Each call to `env.step(action)` executes the following sequence:

1. **Apply Action**: the policy output is converted to a joint position target via the PD controller (see Section 4)
2. **Step Physics**: MuJoCo integrates 4 sub-steps at 200 Hz
3. **Fire Interval Events**: periodic disturbances such as random pushes
4. **Compute Observations**: all sensor terms are evaluated with optional noise corruption
5. **Compute Rewards**: all reward terms are evaluated and multiplied by their weights
6. **Check Terminations**: if any termination condition fires, the episode ends
7. **Return**: (obs, reward, done, info) tuple returned to the RL runner

### 2.4 Parallel Training

During training, `num_envs = 4096` environments run simultaneously on the GPU. All tensors carry a leading batch dimension `B = 4096`. The policy processes all 4096 environments in a single forward pass, and MuJoCo simulates all 4096 in parallel. This massively accelerates data collection.

---

## 3. The Unitree G1 Robot Model

### 3.1 Hardware Anatomy

The Unitree G1 is a full-body humanoid robot. In the simulation model used here, it has **29 actively controlled degrees of freedom**:

| Body Region | Joints |
|-------------|--------|
| Left/Right Hip | Hip pitch, hip roll, hip yaw (6 joints) |
| Left/Right Knee | Knee pitch (2 joints) |
| Left/Right Ankle | Ankle pitch, ankle roll (4 joints, parallel linkage) |
| Left/Right Shoulder | Shoulder pitch, shoulder roll, shoulder yaw (6 joints) |
| Left/Right Elbow | Elbow pitch (2 joints) |
| Left/Right Wrist | Wrist roll, wrist pitch, wrist yaw (6 joints) |
| Torso | Waist yaw, waist pitch, waist roll (3 joints, partial parallel linkage) |

The robot stands at approximately 0.8 m center-of-mass height.

### 3.2 Motor Families and Actuator Dynamics

The G1 uses four distinct brushless DC motor families. Each motor family is characterized by its rotor inertia, planetary gear ratios, velocity limit, and torque limit:

| Family | Assigned Joints | Vel Limit | Torque Limit |
|--------|----------------|-----------|--------------|
| 5020 | Elbows, shoulder pitch/roll/yaw, wrist roll | 37 rad/s | 25 N·m |
| 7520-14 | Hip pitch/yaw, waist yaw | 32 rad/s | 88 N·m |
| 7520-22 | Hip roll, knee | 20 rad/s | 139 N·m |
| 4010 | Wrist pitch/yaw | 22 rad/s | 5 N·m |
| WAIST (2× 5020) | Waist pitch/roll | 37 rad/s | 50 N·m |
| ANKLE (2× 5020) | Ankle pitch/roll | 37 rad/s | 50 N·m |

Hip roll and knee joints use the most powerful motors (139 N·m) because they must support full body weight during stance. Wrists use the weakest motors (5 N·m).

### 3.3 Reflected Inertia Calculation

The effective rotational inertia at the joint output (armature) is computed using the two-stage planetary gearbox formula. For a two-stage planetary with three rotor stages I₁, I₂, I₃ and gear ratios r₁, r₂, r₃:

```
I_reflected = I₁ × r₁² + I₂ × (r₁ × r₂)² + I₃ × (r₁ × r₂ × r₃)²
```

For the 5020 motor:
- Rotor inertias: (0.139e-4, 0.017e-4, 0.169e-4) kg·m²  
- Gear ratios: (1, 1 + 46/18, 1 + 56/16) = (1, 3.556, 4.5)
- Armature ≈ 0.003610 kg·m²

This armature is used directly in MuJoCo's actuator model to correctly represent the joint's effective inertia.

### 3.4 PD Controller Parameters

The PD controller gains are derived from motor dynamics rather than tuned empirically. For each motor family, the stiffness and damping are computed from a target natural frequency of 10 Hz and a damping ratio of 2.0 (critically overdamped):

```
ω_n   = 10 × 2π = 62.83 rad/s   (natural frequency)
ζ     = 2.0                       (damping ratio)
K     = I_armature × ω_n²         (stiffness, N·m/rad)
D     = 2 × ζ × I_armature × ω_n  (damping, N·m·s/rad)
```

For the 5020 motor: K ≈ 14.25 N·m/rad, D ≈ 0.907 N·m·s/rad  
For the 7520-22 motor: K ≈ 99.10 N·m/rad, D ≈ 6.309 N·m·s/rad

This principled derivation ensures that the simulated actuator dynamics closely match the real hardware behavior.

### 3.5 The Home Keyframe (Neutral Standing Pose)

The target configuration for the transition policy is the `HOME_KEYFRAME`:

```python
HOME_KEYFRAME = EntityCfg.InitialStateCfg(
    pos=(0, 0, 0.8),                         # CoM at 0.8 m
    joint_pos={
        ".*_hip_pitch_joint":    -0.1,        # Hips slightly back
        ".*_knee_joint":          0.3,        # Knees slightly bent
        ".*_ankle_pitch_joint":  -0.2,        # Ankles compensate for knee bend
        ".*_shoulder_pitch_joint": 0.35,      # Arms slightly forward
        ".*_elbow_joint":         0.87,       # Elbows near 90°
        "left_shoulder_roll_joint":   0.18,   # Left arm slightly out
        "right_shoulder_roll_joint": -0.18,   # Right arm slightly out (mirror)
    },
    joint_vel={".*": 0.0},
)
```

All joints not listed default to 0.0 radians. This defines a stable, slightly crouched standing posture—the target the policy must learn to reach and maintain.

---

## 4. Action Space

### 4.1 Joint Position Actions

The policy outputs a joint position residual via the `JointPositionActionCfg`. The joint position target is:

```
target_q = default_joint_pos + action × scale
```

where `default_joint_pos` is the HOME\_KEYFRAME configuration. This means the policy outputs **deviations from the home position**, not absolute angles.

### 4.2 Per-Joint Action Scale (Derived from Motor Dynamics)

The transition policy uses a **per-joint action scale** derived from motor dynamics, not a single global scalar. The formula is:

```
scale[joint] = 0.25 × effort_limit / stiffness
```

This ensures that a maximum action magnitude of 1.0 corresponds to a torque at the effort limit of the motor divided by the PD stiffness. Physically, this is the maximum joint displacement the stiffest motor can produce at its torque limit.

The resulting action scales (from the logged configuration) are:

| Joint Group | Scale |
|-------------|-------|
| Elbow/shoulder pitch/roll/yaw, wrist roll | ~0.439 rad |
| Hip pitch/yaw, waist yaw | ~0.548 rad |
| Hip roll, knee | ~0.351 rad |
| Wrist pitch/yaw | ~0.075 rad |
| Waist pitch/roll | ~0.439 rad |
| Ankle pitch/roll | ~0.439 rad |

**Why per-joint scales matter:** Without motor-specific scaling, the policy would need to discover different effective ranges for each joint through trial and error. With physics-informed scaling, a policy action of 1.0 has a physically consistent meaning across all joints—it saturates the motor.

**Contrast with locomotion policy:** The velocity task initially used a uniform `scale=0.25` and then overrides it per-robot. The transition policy adopts the per-joint `G1_ACTION_SCALE` from the start, ensuring more physical realism.

### 4.3 Actuator Count

All 29 joints are simultaneously controlled. The action vector has dimension 29.

---

## 5. Observation Space

The transition policy implements **asymmetric actor-critic** observations: the actor (deployed on hardware) sees a minimal noisy observation; the critic (training only) sees a privileged richer observation.

### 5.1 Actor Observations (Deployed Policy)

| Observation Term | Source | Noise Level | Dimension |
|-----------------|--------|-------------|-----------|
| `base_ang_vel` | IMU angular velocity sensor | ±0.2 rad/s (uniform) | 3 |
| `projected_gravity` | Gravity vector in body frame | ±0.05 (uniform) | 3 |
| `joint_pos` | Joint positions relative to HOME\_KEYFRAME (q − q\_default) | ±0.01 rad (uniform) | 29 |
| `joint_vel` | Joint velocities | ±1.5 rad/s (uniform) | 29 |
| `actions` | Last action (policy output from t−1) | none | 29 |

**Total actor observation dimension: 93**

The `joint_pos` observation (`joint_pos_rel`, returning q − q\_default) is the direct error signal for the primary task. When all joints are at HOME\_KEYFRAME, `joint_pos` is the zero vector.

No velocity command, no terrain scan, no phase signal—the transition task has a fixed target and does not require these.

### 5.2 Critic Observations (Training Only)

The critic receives all actor observations plus:

| Additional Term | Source | Noise | Dimension |
|----------------|--------|-------|-----------|
| `base_lin_vel` | IMU linear velocity | ±0.5 m/s | 3 |
| `foot_contact` | Binary contact state (both feet) | none | 2 |
| `foot_contact_forces` | Net contact force magnitude per foot | none | 2 |

**Total critic observation dimension: 100**

The critic uses privileged contact information to learn a more accurate value function. At deployment, only the actor network runs on hardware—the critic is discarded.

### 5.3 Noise Corruption Rationale

Observation noise during training serves as **sim-to-real domain randomization for sensors**. Real robot sensors have measurement noise:

- IMU angular velocity: ±0.2 rad/s matches typical MEMS IMU noise on the G1
- Joint position encoders: ±0.01 rad matches typical encoder resolution/backlash
- Joint velocity (estimated from encoder differentiation): ±1.5 rad/s is conservative, accounting for velocity estimation latency

Training with noise forces the policy to be robust to sensor imperfections, reducing the sim-to-real gap at deployment.

### 5.4 Comparison with Locomotion Policy Observations

The locomotion (velocity-tracking) policy actor observation includes additional terms not present in the transition policy:

| Term | Locomotion | Transition | Reason |
|------|-----------|-----------|--------|
| `command` (velocity) | Yes | No | No velocity target in transition |
| `phase` (gait clock) | Yes | No | No gait required; robot must stand still |
| `height_scan` (terrain) | Yes | No | Flat terrain only |
| `joint_pos_rel` | Yes | Yes | Shared: error signal |
| `base_ang_vel` | Yes | Yes | Shared: balance feedback |
| `projected_gravity` | Yes | Yes | Shared: orientation |

The transition policy is thus a **strict subset** of the locomotion observation space, reflecting the simpler nature of the task.

---

## 6. Reward Function Design

The reward function is the most important design decision in the policy. This section explains every reward term, its mathematical form, the reason for its inclusion, and the rationale for its weight.

### 6.1 Design Principles

The reward system follows these principles inherited from locomotion policy engineering practice:

1. **Gaussian exponential pattern** for continuous convergence objectives: `exp(−error²/σ²)` provides a nonzero learning gradient even when the robot is far from the target.
2. **Quadratic penalties** (L2) for smoothness and safety constraints.
3. **Binary rewards** (0/1) for discrete contact conditions where gradient direction is implicit.
4. **Catastrophic penalty** for termination to ensure the policy strongly prefers survival.

### 6.2 The Gaussian Exponential Pattern

Many reward terms in this codebase use:
```
reward = exp(−MSE / σ²)
```

This bell curve is centered at the target (MSE = 0):
- At perfect alignment (MSE = 0): reward = 1.0
- At MSE = σ²: reward = exp(−1) ≈ 0.37
- At MSE = 4σ²: reward ≈ 0.018

**Why not a linear reward?** A linear penalty `max(0, 1 − error/range)` has a cliff at `error = range` and constant gradient everywhere. The exponential provides a larger gradient magnitude near the target (encouraging precision) and a meaningful—though small—gradient far from the target (providing initial learning signal when the robot starts with large offsets).

**Choosing σ:** σ is set to match the expected operating range. For `pose_convergence` with initial offset ±0.5 rad and σ = 0.25 rad:
- At 0.5 rad mean error (initial state): reward ≈ 0.018 (nonzero gradient exists)
- At 0.1 rad error: reward ≈ 0.85 (strong positive reinforcement)
- At 0.0 rad error: reward = 1.0

Using `mean` over joints (not `sum`) ensures the reward magnitude is independent of the number of joints, making σ interpretable as "per-joint RMS error at which reward = 0.37."

### 6.3 Reward Terms — Transition Policy (v1)

#### `pose_convergence` (weight: +2.0)

```python
def pose_convergence(env, std, asset_cfg):
    q         = asset.data.joint_pos[:, asset_cfg.joint_ids]          # [B, 29]
    q_default = asset.data.default_joint_pos[:, asset_cfg.joint_ids]  # [B, 29]
    mse       = torch.mean(torch.square(q - q_default), dim=1)        # [B]
    return torch.exp(-mse / std**2)                                    # [B] ∈ [0,1]
```

**Purpose:** Primary learning signal. Drives all 29 joints to HOME\_KEYFRAME.  
**std = 0.25 rad:** Gradient is meaningful from the initial ±0.5 rad offset.  
**Weight = +2.0:** Anchor weight for the entire reward function. All other terms are calibrated relative to this.

#### `both_feet_contact` (weight: +0.5)

```python
def both_feet_contact(env, sensor_name):
    in_contact = sensor.data.found > 0   # [B, 2] — left and right foot
    return in_contact.all(dim=1).float() # [B] — 1.0 iff both feet down
```

**Purpose:** Prevents the policy from achieving `pose_convergence` through unstable means (hopping, single-leg stance). Without this reward, the policy may discover that it can reach neutral by lifting a foot temporarily, which is unstable on real hardware.  
**Weight = +0.5:** Secondary incentive. Maximum contribution is 0.5/step.

#### `body_orientation_l2` (weight: −2.0)

**Purpose:** Penalizes the torso tilting. Measures the x and y components of gravity projected into the body frame—these are zero when upright and grow as sin(θ) as the robot tilts by angle θ.  
**Weight = −2.0:** Strong penalty. At 30° tilt, penalty ≈ −1.0/step.

#### `joint_vel_penalty` (weight: −0.01)

```python
def joint_vel_penalty(env, asset_cfg):
    vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    return torch.sum(torch.square(vel), dim=1)
```

**Purpose:** (1) Discourages flailing during convergence. (2) Penalizes oscillation around the target—the policy learns to *arrive and stop*, not oscillate.  
**Weight = −0.01:** Small; typical penalty ≈ −0.29/step (29 joints × 1 rad/s² → 29 × 0.01 = 0.29).

#### `body_ang_vel` (weight: −0.05)

**Purpose:** Penalizes roll and pitch angular velocity of the torso (but not yaw, since spinning in place does not threaten balance).  
**Weight = −0.05:** Moderate penalty.

#### `angular_momentum` (weight: −0.025)

**Purpose:** Penalizes whole-body angular momentum (sum of all link momenta about a common reference point). Captures limb motion that body_ang_vel misses—arms swinging can create large total angular momentum even when the torso is stationary.

#### `is_terminated` (weight: −200.0)

```python
def is_terminated(env):
    return env.termination_manager.terminated.float()  # [B], fires once per fall
```

**Purpose:** Catastrophic penalty for falling. Fires once when the `bad_orientation` termination condition (torso angle > 70°) triggers.  
**Calibration:** Maximum cumulative positive reward over 15 s at 50 Hz (750 steps, γ=0.99):
- `pose_convergence`: 2.0 × Σ_{k=0}^{749} 0.99^k ≈ 2.0 × 74.5 = 149
- `is_terminated` penalty: −200 (larger in magnitude than 149)

This ensures the policy always prefers survival over any amount of positive reward from a single episode.

#### `joint_acc_l2` (weight: −2.5e-7)

**Purpose:** Penalizes jerky joint trajectories (rapid changes in joint velocity). Extremely small weight because accelerations can be very large (hundreds of rad/s²) during fast transitions.

#### `joint_pos_limits` (weight: −10.0)

**Purpose:** Penalizes joint positions approaching or exceeding 90% of hardware joint limits. Prevents the policy from driving joints into their mechanical stops during convergence.

#### `action_rate_l2` (weight: −0.05)

**Purpose:** Penalizes rapid changes in the policy output (action) between consecutive steps. Encourages smooth joint position commands, reducing PD controller chatter and hardware wear.

#### `self_collisions` (weight: −1.0, G1-specific)

**Purpose:** Penalizes arm or leg collisions with the body during reconfiguration. Particularly important for the transition task because joints start up to ±0.5 rad from neutral—arms may swing into the torso during convergence.  
**Sensor:** `ContactSensorCfg` monitors the subtree rooted at the pelvis for self-contact.  
**`force_threshold = 10 N`:** Micro-contacts from resting gravity are not penalized; only impacts above 10 N are counted.

### 6.4 Reward Weight Comparison: Transition vs. Locomotion

The following table highlights reward term differences between the transition policy and the velocity locomotion policy, showing how the transition policy was derived from the locomotion policy foundation while replacing locomotion-specific terms with posture-specific ones:

| Term | Transition v1 | Locomotion | Notes |
|------|--------------|-----------|-------|
| `pose_convergence` | +2.0 | N/A | **New**: primary for transition |
| `track_linear_velocity` | N/A | +1.0 | Removed: no velocity command |
| `track_angular_velocity` | N/A | +1.0 | Removed: no velocity command |
| `variable_posture` | N/A | +1.0 | Replaced by `pose_convergence` |
| `both_feet_contact` | +0.5 | N/A | **New**: bilateral stance requirement |
| `body_orientation_l2` | −2.0 | −1.0 | Weight doubled (no gait motion) |
| `joint_vel_penalty` | −0.01 | N/A | **New**: settling signal |
| `body_ang_vel` | −0.05 | −0.05 | Unchanged |
| `angular_momentum` | −0.025 | −0.025 | Unchanged |
| `is_terminated` | −200.0 | −200.0 | Unchanged |
| `joint_acc_l2` | −2.5e-7 | −2.5e-7 | Unchanged |
| `joint_pos_limits` | −10.0 | −10.0 | Unchanged |
| `action_rate_l2` | −0.05 | −0.05 | Unchanged |
| `foot_gait` | N/A | +0.5 | Removed: no gait |
| `foot_clearance` | N/A | −1.0 | Removed: no stepping |
| `foot_slip` | N/A | −0.25 | Removed: no walking |
| `soft_landing` | N/A | −1e-3 | Removed: no stepping |
| `stand_still` | N/A | −1.0 | Replaced by `pose_convergence` |
| `self_collisions` | −1.0 | −1.0 | Carried over |

The transition policy inherits the safety/smoothness penalty infrastructure directly from the locomotion policy (orientation, angular momentum, joint limits, action rate), and replaces locomotion-specific gait rewards with posture convergence terms.

---

## 7. Episode Reset and Domain Randomization

### 7.1 Episode Reset (per-episode, mode="reset")

Every time an episode terminates (fall or time-out):

**`reset_base`:** Scatters the robot randomly across the flat terrain plane:
```
x, y ∈ [−0.5, 0.5] m    (avoids crowding)
yaw ∈ [−π, π] rad        (random heading)
velocities: zero
```

**`reset_robot_joints`:** Applies independent random offsets to every joint:
```
q_init = q_HOME_KEYFRAME + U(−0.5, +0.5) rad  (independent per joint)
q_dot  = 0.0 rad/s
```

The ±0.5 radian offset range is the core training perturbation. The policy must recover from any combination of 29 independent random joint offsets. This ensures the learned policy is a general recovery strategy, not a specific motion sequence.

### 7.2 Domain Randomization (once at startup, mode="startup")

**`foot_friction`:** Randomizes the foot collision geom friction coefficients:
```
friction ∈ [0.3, 1.6]   (absolute value, shared across all 14 foot geoms)
```
This covers slippery floors (μ=0.3) to rough carpets (μ=1.6), forcing the policy to be robust to varying ground conditions.

**`encoder_bias`:** Adds a persistent per-joint sensor offset:
```
bias_range ∈ [−0.015, +0.015] rad (per joint)
```
Simulates real encoder bias that remains constant within an episode. The policy must learn to recover despite small systematic measurement errors.

**`base_com`:** Offsets the torso center-of-mass:
```
x, y, z offset ∈ [−0.05, +0.05] m
```
Simulates payload uncertainty or manufacturing variation in the robot's mass distribution.

### 7.3 Periodic Disturbance (mode="interval")

**`push_robot`:** Every 8–10 seconds, randomly sets the robot's velocity:
```
v_x, v_y ∈ [−0.3, 0.3] m/s     (lighter than locomotion: ±0.5 m/s)
v_z ∈ [−0.2, 0.2] m/s
ω_roll, ω_pitch ∈ [−0.3, 0.3] rad/s
ω_yaw ∈ [−0.5, 0.5] rad/s
```

These pushes test whether the policy can recover from disturbances *after* reaching the neutral pose—ensuring the policy holds, not just reaches, the target.

The push magnitudes are lighter than the locomotion task (±0.5 m/s for locomotion vs. ±0.3 m/s for transition) because the robot is stationary and more vulnerable to sudden velocity changes.

---

## 8. Contact Sensing

### 8.1 Foot–Ground Contact Sensor

```python
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
```

This sensor tracks whether each foot subtree is in contact with the terrain. `fields=("found", "force")` returns both a binary contact flag and the net contact force vector. The sensor is used by:
- `both_feet_contact` reward (binary: both feet down?)
- Critic observations (`foot_contact`, `foot_contact_forces`)

### 8.2 Self-Collision Sensor

```python
self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
)
```

Self-contact is detected by matching the robot's own pelvis subtree against itself. `history_length=4` tracks contacts over the last 4 substeps (20 ms window). The `self_collision_cost` reward function counts substeps in this window where contact force exceeds the 10 N threshold.

---

## 9. Termination Conditions

| Condition | Function | Threshold | time_out |
|-----------|----------|-----------|----------|
| Episode time limit | `time_out` | 15 s (750 steps at 50 Hz) | True |
| Fallen over | `bad_orientation` | Torso tilt > 70° | False |

The `time_out=True` flag is critical for PPO: when an episode ends due to time-out (not failure), PPO bootstraps the value function from the terminal state rather than treating it as a failure. This prevents the algorithm from penalizing episodes that simply ran their full duration.

The `bad_orientation` termination fires when the torso exceeds 70° from vertical. This triggers the `is_terminated` penalty (−200.0) and forces a new episode.

---

## 10. Neural Network Architecture

Both actor and critic use the same feedforward architecture with observation normalization:

```
Input → LayerNorm → Linear(in, 512) → ELU
                  → Linear(512, 256) → ELU
                  → Linear(256, 128) → ELU
                  → Linear(128, out)
```

| Component | Configuration | Output Dimension |
|-----------|--------------|-----------------|
| Actor | (512, 256, 128), ELU, obs normalization | 29 (joint residuals) |
| Critic | (512, 256, 128), ELU, obs normalization | 1 (value estimate) |

**Action distribution:** Gaussian with shared scalar standard deviation, initialized at 1.0. The policy outputs the mean; the standard deviation is a trainable scalar parameter.

**Observation normalization (`obs_normalization=True`):** Running mean and variance are maintained for each observation dimension. This is critical because observations span very different scales (angular velocities in rad/s vs. joint positions in rad vs. contact forces in N).

**ELU activation:** Exponential Linear Unit provides smooth gradients and avoids dead neurons (compared to ReLU), which is important for the continuous joint position outputs.

---

## 11. PPO Training Configuration

### 11.1 Algorithm Hyperparameters

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `num_steps_per_env` | 24 | Rollout steps per environment per iteration |
| `num_learning_epochs` | 5 | Number of passes over the collected buffer |
| `num_mini_batches` | 4 | Buffer is split into 4 mini-batches per epoch |
| `learning_rate` | 1e-3 | Adam optimizer initial learning rate |
| `schedule` | adaptive | LR is scaled up/down based on KL divergence |
| `gamma` | 0.99 | Discount factor |
| `lam` | 0.95 | GAE lambda (bias-variance trade-off) |
| `clip_param` | 0.2 | PPO clipping parameter ε |
| `value_loss_coef` | 1.0 | Weight for critic loss |
| `entropy_coef` | 0.01 | Entropy bonus (encourages exploration) |
| `desired_kl` | 0.01 | Target KL divergence for adaptive LR |
| `max_grad_norm` | 1.0 | Gradient clipping threshold |
| `max_iterations` | 10,001 | Total training iterations |

### 11.2 Batch Size Calculation

```
Total environments: 4096
Steps per env per iteration: 24
Total samples per iteration: 4096 × 24 = 98,304
Mini-batch size: 98,304 / 4 = 24,576
```

With 5 epochs per iteration, each sample is trained on 5 times per collection cycle.

### 11.3 Adaptive Learning Rate

When `schedule="adaptive"`:
- If KL divergence between old and new policy exceeds `desired_kl = 0.01`: learning rate × 1/1.5 (decrease)
- If KL divergence falls below `desired_kl / 2 = 0.005`: learning rate × 1.5 (increase)

This automatically prevents destructively large policy updates during training.

### 11.4 GAE Advantage Estimation

Generalized Advantage Estimation (GAE) with λ=0.95 computes per-step advantages:
```
δ_t = r_t + γ V(s_{t+1}) - V(s_t)          (TD residual)
A_t = δ_t + (γλ)δ_{t+1} + (γλ)²δ_{t+2} + ...
```

With γ=0.99 and λ=0.95, the effective discount for advantages is γλ = 0.9405. This gives a bias-variance trade-off: lower λ would produce lower-variance but higher-bias estimates; higher λ approaches Monte Carlo returns (low bias, high variance).

---

## 12. ONNX Export and Deployment

The `TransitionOnPolicyRunner` extends the base runner to export deployment-ready ONNX models on every checkpoint save:

```python
class TransitionOnPolicyRunner(MjlabOnPolicyRunner):
    def save(self, path, infos=None):
        super().save(path, infos)
        self.export_policy_to_onnx(policy_path, "policy.onnx")
        metadata = get_base_metadata(self.env.unwrapped, run_name)
        attach_metadata_to_onnx(onnx_path, metadata)
```

Every 100 training iterations (`save_interval=100`), a checkpoint is saved and the actor network is exported to ONNX format with deployment metadata (environment dimensions, action scale, observation normalization statistics).

The deployment stack runs on the G1's onboard computer (ARM64 architecture) using ONNX Runtime 1.22.0 for AArch64, which is included in the `deploy/` directory.

The C++ deployment code implements:
- The same observation computation as the Python training environment
- The same action mapping (target = default + action × scale)
- An FSM (Finite State Machine) to switch between standing, locomotion, and RL policies

---

## 13. Transition-v2: Momentum-Aware Extension

### 13.1 Motivation

The original transition policy (v1) assumes the robot starts from rest (zero body velocity, zero joint velocity). In practice, the robot transitions from locomotion, where it has residual body momentum. The v1 policy, never trained with initial momentum, would struggle to first damp the momentum before converging to neutral.

Transition-v2 extends the task to cover this more realistic scenario: the robot begins each episode with small random body velocities and joint velocities, simulating the robot decelerating from slow walking.

### 13.2 Initial Momentum at Reset

The v2 episode reset adds velocity ranges to the `reset_base` event:

| Velocity Component | v1 | v2 |
|-------------------|----|----|
| Body linear x, y | 0 m/s | ±0.2 m/s |
| Body angular roll, pitch | 0 rad/s | ±0.3 rad/s |
| Body angular yaw | 0 rad/s | ±0.15 rad/s |
| Joint velocities | 0 rad/s | ±0.15 rad/s |

The velocity ranges are chosen to match plausible end-of-locomotion states:
- ±0.2 m/s linear: the robot was walking at slow pace and has begun to decelerate
- ±0.3 rad/s angular: rocking from the last walking step
- ±0.15 rad/s joint: joints still settling from the prior motion

### 13.3 New Reward Terms for v2

#### `angular_velocity_convergence` (weight: +0.7)

```python
def angular_velocity_convergence(env, std, asset_cfg):
    ang_vel_b = asset.data.root_link_ang_vel_b  # [B, 3]
    ang_vel_sq = torch.sum(torch.square(ang_vel_b), dim=1)
    return torch.exp(-ang_vel_sq / std**2)
```

**σ = 0.3 rad/s** (matches `ANGULAR_VEL_RANGE`): The gradient is meaningful from the very first step of an episode where the robot starts rocking at ±0.3 rad/s.  
**Purpose:** Explicitly rewards damping body angular velocity to zero. Without this, `pose_convergence` alone does not guarantee the robot stops rocking after reaching neutral—it could oscillate around the target while satisfying the pose condition intermittently.

#### `linear_velocity_convergence` (weight: +0.4)

```python
def linear_velocity_convergence(env, std, asset_cfg):
    lin_vel_b = asset.data.root_link_lin_vel_b  # [B, 3]
    lin_vel_sq = torch.sum(torch.square(lin_vel_b), dim=1)
    return torch.exp(-lin_vel_sq / std**2)
```

**σ = 0.2 m/s** (matches `LINEAR_VEL_RANGE`).  
**Weight = +0.4** (smaller than angular because linear drift is less destabilizing than rocking).

#### `hold_bonus` (weight: +1.0)

```python
def hold_bonus(env, pose_threshold, ang_vel_threshold, lin_vel_threshold, asset_cfg):
    pose_ok = torch.mean(torch.abs(q - q_default), dim=1) < pose_threshold  # [B]
    ang_ok  = torch.norm(ang_vel_b, dim=1) < ang_vel_threshold               # [B]
    lin_ok  = torch.norm(lin_vel_b, dim=1) < lin_vel_threshold               # [B]
    return (pose_ok & ang_ok & lin_ok).float()
```

**Thresholds:**
- Pose: mean |q − q\_default| < 0.08 rad (~4.6°)
- Angular velocity: |ω\_b| < 0.15 rad/s
- Linear velocity: |v\_b| < 0.10 m/s

**Purpose:** Binary bonus for the "locked-in" state. Fires only when all three conditions hold simultaneously. This has no gradient direction of its own—the gradient comes from the three convergence reward terms. The `hold_bonus` fires as an amplifier once the policy has learned to satisfy all conditions, incentivizing it to hold the locked state rather than drift.

### 13.4 Weight Adjustments in v2

| Term | v1 Weight | v2 Weight | Reason |
|------|-----------|-----------|--------|
| `joint_vel_penalty` | −0.01 | −0.02 | Robot starts with nonzero joint velocity; stronger damping needed |
| `body_ang_vel` | −0.05 | −0.10 | Rocking more prevalent with initial angular momentum |

### 13.5 Episode Length and Training Schedule

| Parameter | v1 | v2 |
|-----------|----|----|
| `episode_length_s` | 15 s (750 steps) | 20 s (1000 steps) |
| `num_steps_per_env` | 24 | 32 |
| `max_iterations` | 10,001 | 15,001 |

The episode is extended from 15 s to 20 s to give the policy enough time to both damp initial momentum and converge to neutral. The `num_steps_per_env` is increased proportionally to keep the per-iteration batch size comparable. More training iterations (15,001 vs. 10,001) are allocated because the v2 task is harder.

### 13.6 v1 vs. v2 Summary Comparison

| Dimension | Transition v1 | Transition v2 |
|-----------|--------------|---------------|
| Initial joint offset | ±0.5 rad | ±0.5 rad |
| Initial joint velocity | 0 rad/s | ±0.15 rad/s |
| Initial body linear vel | 0 m/s | ±0.2 m/s |
| Initial body angular vel | 0 rad/s | ±0.3 rad/s |
| Episode length | 15 s | 20 s |
| Primary reward | `pose_convergence` | `pose_convergence` |
| Momentum damping rewards | None | `angular_velocity_convergence`, `linear_velocity_convergence` |
| Hold-state bonus | None | `hold_bonus` |
| `joint_vel_penalty` weight | −0.01 | −0.02 |
| `body_ang_vel` weight | −0.05 | −0.10 |
| Max training iterations | 10,001 | 15,001 |

---

## 14. Relationship to the Locomotion Policy

### 14.1 Architectural Inheritance

The transition policy was directly built on the same framework and conventions as the velocity locomotion policy (`Unitree-G1-Rough`). The following design elements are carried over without modification:

- **Same PD actuator model** with physics-derived gains (stiffness, damping, armature)
- **Same per-joint action scale formula** (`0.25 × effort_limit / stiffness`)
- **Same simulation parameters** (timestep, decimation, integrator, solver iterations)
- **Same observation noise values** (IMU noise ±0.2, encoder ±0.01, velocity ±1.5)
- **Same domain randomization** (foot friction [0.3, 1.6], encoder bias ±0.015, CoM offset ±0.05)
- **Same safety penalties** (joint limits, action smoothness, angular momentum, orientation)
- **Same neural network architecture** (512-256-128, ELU, obs normalization)
- **Same PPO hyperparameters** (clip_param=0.2, gamma=0.99, lam=0.95, lr=1e-3)
- **Same catastrophic termination penalty** (is_terminated: −200.0)
- **Same ONNX export pipeline**
- **Same contact sensor configuration** (foot–ground and self-collision sensors)

### 14.2 Key Differences: Task-Specific Design

| Aspect | Locomotion Policy | Transition Policy |
|--------|------------------|------------------|
| **Goal** | Track velocity commands on rough terrain | Return to HOME\_KEYFRAME from any upright pose |
| **Terrain** | Procedurally generated rough terrain | Flat plane only |
| **Commands** | Velocity (lin_x, lin_y, ang_z), resampled 3–8 s | None |
| **Gait** | Required (foot_gait reward, phase clock) | Explicitly unwanted |
| **Primary reward** | `track_linear_velocity`, `track_angular_velocity` | `pose_convergence` |
| **Reset displacement** | 0 (starts at default) | ±0.5 rad per joint |
| **Terrain scan** | Yes (160-point height map) | No |
| **Curriculum** | Yes (terrain difficulty, velocity range staged) | No |
| **Foot clearance** | Yes | No |
| **Foot slip** | Yes | No |
| **Self-collision penalty** | Yes (inherited) | Yes (more important here) |
| **Both feet contact** | No explicit reward | Yes (prevents single-leg balance) |

### 14.3 Why the Transition Policy Needed to Diverge

The locomotion policy cannot be used as a transition controller because:

1. **Goal mismatch:** The locomotion policy tracks velocity commands. Without a command, it continues to generate gait motions rather than converging to a static pose.

2. **Gait rewards interfere:** `foot_gait` rewards alternate foot contacts at a fixed period (0.6 s). This would actively discourage the robot from standing still, as the reward function expects a walking gait.

3. **No convergence signal:** The locomotion policy's `variable_posture` reward is speed-dependent. At zero speed (standing), it rewards the standing posture. But it provides no signal for recovering from a perturbed initial configuration.

4. **Reset distribution mismatch:** The locomotion policy resets joints at zero offset (default position), so it is never trained on recovering from large joint displacements.

5. **Episode structure:** The locomotion policy's curriculum progressively introduces harder terrain. The transition task needs flat terrain only—terrain variation would introduce unnecessary complexity.

---

## 15. Validation and Evaluation

### 15.1 Play Mode

The `play.py` script enables policy visualization and validation:
```bash
uv run python scripts/play.py Unitree-G1-Transition \
    --checkpoint-file logs/rsl_rl/g1_transition/<run>/model_<iter>.pt \
    --viewer viser       # browser-based viewer, no display required
```

Options:
- `--no-terminations`: Disables termination conditions to observe full episode trajectory without early stops
- `--viewer native`: Native MuJoCo viewer (requires `$DISPLAY`)

### 15.2 Qualitative Validation Criteria

During play mode, a successfully trained policy should exhibit:

1. **Convergence:** From any of the ±0.5 rad initial offsets, all joints reach the HOME\_KEYFRAME within 3–5 seconds
2. **Stability:** The robot maintains a bilateral stance (both feet in contact) throughout
3. **Upright posture:** The torso remains upright (< 15° from vertical) during convergence
4. **Smooth motion:** No jerky or oscillatory motion; the `action_rate_l2` penalty ensures smooth commands
5. **Robustness to pushes:** When the `push_robot` event fires (every 8–10 s), the robot recovers
6. **No self-collision:** Arms do not swing into the torso during reconfiguration

### 15.3 Logged Metrics

The training runner logs per-step and episode-averaged values for each reward term to Weights & Biases (wandb). Key metrics to monitor:

| Metric | Healthy Range | Indicates |
|--------|--------------|-----------|
| `pose_convergence` mean | Rising from ~0.02 to > 0.8 | Convergence learning |
| `both_feet_contact` mean | Rising to > 0.9 | Stable bilateral stance |
| `body_orientation_l2` mean | Falling toward 0 | Upright posture maintained |
| `is_terminated` total | Falling rapidly | Fewer falls over time |
| Mean episode length | Rising toward 750 steps | Episodes running to timeout, not falling |

### 15.4 Simulation Environment Validation

The environment configuration was validated to be internally consistent through:

1. **Action scale validation:** Per-joint scales computed programmatically from actuator specs and verified to be in physically meaningful ranges (0.075–0.548 rad)
2. **Reward calibration:** Maximum cumulative positive return (≈149) is less than the termination penalty magnitude (200), ensuring correct incentive structure
3. **Noise level validation:** Noise values match typical G1 hardware sensor specifications
4. **Domain randomization coverage:** Friction range [0.3, 1.6] covers plausible indoor and outdoor floor conditions; CoM offset ±0.05 m covers reasonable payload scenarios

---

## 16. Training Infrastructure

### 16.1 Software Stack

| Component | Version/Details |
|-----------|----------------|
| Physics engine | MuJoCo (via `mjlab`) |
| RL framework | RSL-RL (PPO implementation) |
| Python | 3.10 |
| Package manager | `uv` |
| Deep learning | PyTorch |
| Multi-GPU | `torchrunx` |
| Deployment inference | ONNX Runtime 1.22.0 (AArch64) |
| Experiment tracking | Weights & Biases (wandb) |

### 16.2 Training Commands

```bash
# Single GPU (default)
uv run python scripts/train.py Unitree-G1-Transition --env.scene.num-envs 4096

# CPU only
uv run python scripts/train.py Unitree-G1-Transition --gpu-ids null

# Specific GPU
uv run python scripts/train.py Unitree-G1-Transition --agent.gpu-ids 1

# Multi-GPU
uv run python scripts/train.py Unitree-G1-Transition --gpu-ids all
```

### 16.3 Checkpoint Structure

Checkpoints are saved to:
```
logs/rsl_rl/g1_transition/<timestamp>/
├── params/
│   ├── agent.yaml    # Full PPO configuration snapshot
│   └── env.yaml      # Full environment configuration snapshot
├── model_<N>.pt      # PyTorch checkpoint (actor + critic weights, running stats)
└── policy.onnx       # Exported deployment-ready actor network
```

---

## 17. Task Registration

The task is registered via `mjlab`'s task registry system. Each task registers three functions under a string ID:

| Task ID | env\_cfg function | rl\_cfg function | runner\_cls |
|---------|------------------|-----------------|------------|
| `Unitree-G1-Transition` | `unitree_g1_transition_env_cfg` | `unitree_g1_transition_ppo_runner_cfg` | `TransitionOnPolicyRunner` |
| `Unitree-G1-Transition-v2` | `unitree_g1_transition_v2_env_cfg` | `unitree_g1_transition_v2_ppo_runner_cfg` | `TransitionOnPolicyRunner` |

The `train.py` script discovers all registered tasks by importing `src.tasks`, then uses `tyro` to parse CLI arguments and override any configuration field.

---

## 18. Complete Hyperparameter Reference

### 18.1 Simulation

| Parameter | Value |
|-----------|-------|
| Physics timestep | 0.005 s |
| Policy timestep | 0.020 s (decimation = 4) |
| Episode length | 15 s (v1) / 20 s (v2) |
| Gravity | [0, 0, −9.81] m/s² |
| Integrator | `implicitfast` |
| Solver iterations | 10 |
| Line-search iterations | 20 |
| CCD iterations | 50 |
| Max contact pairs | 300 |
| Contact sensor slots | 64 |

### 18.2 Reset Distribution

| Parameter | Value |
|-----------|-------|
| Base x, y | ±0.5 m |
| Base yaw | ±π rad |
| Joint position offset | ±0.5 rad |
| Joint velocity (v1) | 0 rad/s |
| Joint velocity (v2) | ±0.15 rad/s |
| Body linear velocity (v2) | ±0.2 m/s (x, y) |
| Body angular velocity (v2) | ±0.3 rad/s (roll, pitch), ±0.15 (yaw) |

### 18.3 Domain Randomization

| Parameter | Range |
|-----------|-------|
| Foot friction | [0.3, 1.6] |
| Encoder bias | ±0.015 rad |
| CoM offset | ±0.05 m (x, y, z) |
| Push velocity x, y | ±0.3 m/s |
| Push interval | 8–10 s |

### 18.4 Reward Weights — Transition v1

| Term | Weight |
|------|--------|
| `pose_convergence` (std=0.25) | +2.0 |
| `both_feet_contact` | +0.5 |
| `body_orientation_l2` | −2.0 |
| `joint_vel_penalty` | −0.01 |
| `body_ang_vel` | −0.05 |
| `angular_momentum` | −0.025 |
| `is_terminated` | −200.0 |
| `joint_acc_l2` | −2.5e-7 |
| `joint_pos_limits` | −10.0 |
| `action_rate_l2` | −0.05 |
| `self_collisions` (threshold=10 N) | −1.0 |

### 18.5 Reward Weights — Transition v2 (additions/changes)

| Term | Weight |
|------|--------|
| `angular_velocity_convergence` (std=0.3 rad/s) | +0.7 |
| `linear_velocity_convergence` (std=0.2 m/s) | +0.4 |
| `hold_bonus` (pose < 0.08 rad, ω < 0.15, v < 0.10) | +1.0 |
| `joint_vel_penalty` | −0.02 (doubled from v1) |
| `body_ang_vel` | −0.10 (doubled from v1) |

### 18.6 PPO Hyperparameters

| Parameter | v1 | v2 |
|-----------|----|----|
| `num_steps_per_env` | 24 | 32 |
| `max_iterations` | 10,001 | 15,001 |
| `num_learning_epochs` | 5 | 5 |
| `num_mini_batches` | 4 | 4 |
| `learning_rate` | 1e-3 | 1e-3 |
| `schedule` | adaptive | adaptive |
| `gamma` | 0.99 | 0.99 |
| `lam` | 0.95 | 0.95 |
| `clip_param` | 0.2 | 0.2 |
| `entropy_coef` | 0.01 | 0.01 |
| `desired_kl` | 0.01 | 0.01 |
| `max_grad_norm` | 1.0 | 1.0 |
| `save_interval` | 100 | 100 |

---

## 19. Discussion

### 19.1 Design Choices and Their Rationale

**Flat terrain only:** Rough terrain would require a height-scan sensor and terrain curriculum, adding complexity without contributing to the transition problem. The policy must generalize across different floor friction values (handled by domain randomization), not different terrain geometries.

**No gait rewards:** The locomotion policy's `foot_gait` reward enforces an alternating contact pattern at 0.6 s period. Including this would prevent the robot from standing still—which is exactly what the transition policy requires.

**Per-joint action scale from motor dynamics:** Rather than empirically tuning a single global scale, the action scale is computed directly from motor specifications. This ensures physical consistency and reduces the number of hyperparameters to tune.

**Gaussian convergence rewards instead of linear:** The Gaussian exponential form provides a smooth, non-zero gradient from the maximum initial offset (0.5 rad) all the way to the target. A linear penalty would provide no gradient once the robot is further away than the clamp range.

**Catastrophic termination penalty:** The −200 `is_terminated` penalty is calibrated to be slightly larger than the maximum achievable positive return over a full episode (~149). This ensures the policy strongly prefers survival without becoming paralyzingly conservative.

**Momentum-aware v2 formulation:** Rather than implicitly hoping that the v1 policy would handle initial momentum, v2 explicitly adds momentum-damping rewards with σ values matched to the initial velocity ranges. This ensures gradient signals are present from the first training step for the new difficulty dimension.

### 19.2 Limitations

1. **Static transition only:** The policy recovers from standing-but-displaced configurations. It does not handle getting up from a fall (from-ground recovery) or transitions between sitting and standing.

2. **No locomotion handoff protocol:** The current design assumes the transition policy is engaged after the locomotion policy has already decelerated. A smooth handoff between the two policies remains an open engineering problem.

3. **Single robot:** The task is designed for the G1 with 29 DOF. Extending to other robot morphologies (G1-23DOF, H1-2, etc.) requires new per-robot configuration files but the base reward infrastructure is reusable.

4. **No terrain generalization:** The policy is trained on flat terrain only. Transitioning on uneven surfaces would require extending the observation space and training curriculum.

---

## 20. Conclusion

This report documents the complete technical design of the Unitree G1 transition policy—a reinforcement learning controller that drives the robot from arbitrary upright-standing configurations back to a stable neutral posture. The design makes the following key contributions:

1. **A well-motivated task formulation** that is distinct from locomotion and necessary for real-world deployment: a robot that can recover its neutral posture after any standing disturbance.

2. **A physics-informed action parameterization** using per-joint action scales derived from motor stiffness and effort limits, ensuring physical consistency without manual hyperparameter tuning.

3. **A reward function** based on the Gaussian exponential convergence pattern, calibrated so that: (a) the primary `pose_convergence` reward provides a learning signal from the largest initial perturbations, (b) safety penalties ensure smooth and safe convergence, and (c) the termination penalty correctly outweighs all achievable positive returns.

4. **A systematic extension (v2)** that adds initial momentum to the episode distribution and introduces explicit momentum-damping reward terms, enabling the policy to handle more realistic end-of-locomotion transitions.

5. **Architectural consistency with the locomotion policy**, allowing the transition policy to reuse the same actuator model, domain randomization configuration, safety penalties, neural network architecture, and deployment pipeline established for the locomotion controller.

Together, these design choices produce a policy suitable for deployment on real G1 hardware as a safety recovery behavior that can be engaged whenever the robot needs to return to its default standing posture.

---

## References and Source Files

| Concept | File |
|---------|------|
| Base transition environment | [src/tasks/transition/transition_env_cfg.py](src/tasks/transition/transition_env_cfg.py) |
| Transition reward functions | [src/tasks/transition/mdp/rewards.py](src/tasks/transition/mdp/rewards.py) |
| G1-specific overrides | [src/tasks/transition/config/g1/env_cfgs.py](src/tasks/transition/config/g1/env_cfgs.py) |
| PPO configuration | [src/tasks/transition/config/g1/rl_cfg.py](src/tasks/transition/config/g1/rl_cfg.py) |
| Transition-v2 environment | [src/tasks/transition_v2/transition_v2_env_cfg.py](src/tasks/transition_v2/transition_v2_env_cfg.py) |
| Transition-v2 rewards | [src/tasks/transition_v2/mdp/rewards.py](src/tasks/transition_v2/mdp/rewards.py) |
| G1 motor constants | [src/assets/robots/unitree_g1/g1_constants.py](src/assets/robots/unitree_g1/g1_constants.py) |
| Locomotion reference | [src/tasks/velocity/velocity_env_cfg.py](src/tasks/velocity/velocity_env_cfg.py) |
| Training script | [scripts/train.py](scripts/train.py) |
| Play/evaluation script | [scripts/play.py](scripts/play.py) |
| ONNX export runner | [src/tasks/transition/rl/runner.py](src/tasks/transition/rl/runner.py) |
| Logged PPO config (v1) | [logs/rsl_rl/g1_transition/2026-06-30_14-26-17/params/agent.yaml](logs/rsl_rl/g1_transition/2026-06-30_14-26-17/params/agent.yaml) |
| Logged env config (v1) | [logs/rsl_rl/g1_transition/2026-06-30_14-26-17/params/env.yaml](logs/rsl_rl/g1_transition/2026-06-30_14-26-17/params/env.yaml) |
