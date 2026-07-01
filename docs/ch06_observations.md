# Chapter 6: Observations — What the Policy Sees

The observation design is the most consequential decision after the reward design. If the
policy cannot observe the information it needs to solve the task, it will fail regardless of
how good the reward is. If it observes too much, it may overfit to training conditions and
fail on the real robot.

---

## 6.1 The Actor/Critic Split (Asymmetric Actor-Critic)

This policy uses **asymmetric actor-critic**, also called **privileged learning** or
**student-teacher asymmetry**:

- The **actor** (the policy that acts in the environment) sees only sensor-accessible,
  noise-corrupted observations. This matches what a real robot could observe.
- The **critic** (the value function, used only during training) sees a richer observation
  including ground-truth physics quantities (e.g., actual contact forces, true linear
  velocity) that would require additional hardware or estimation to access on a real robot.

The actor learns to act well by optimizing against the advantage estimates provided by the
better-informed critic. The result is a policy that makes good decisions from limited sensor
data, guided during training by privileged information it does not have at deployment time.

---

## 6.2 Actor Observations

```python
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
```

### `base_ang_vel` — IMU Angular Velocity (3 values)

**Shape:** `[B, 3]` — (roll rate, pitch rate, yaw rate) in the robot's body frame.

**Source:** The G1 has an onboard IMU (Inertial Measurement Unit). The angular velocity
is measured by a gyroscope in the IMU. `sensor_name="robot/imu_ang_vel"` is the name of
the builtin sensor registered on the robot entity.

**Why include it:** The angular velocity tells the policy whether the torso is currently
rotating. If the robot is starting to fall (high roll or pitch rate), the policy must act
quickly to arrest the motion. Without this, the policy only knows its current tilt angle
(from projected gravity), not how fast it is changing.

**Noise ±0.2 rad/s:** Real MEMS gyroscopes have drift and noise at this level. The policy
must be robust to this magnitude of angular velocity error.

### `projected_gravity` — Tilt Direction (3 values)

**Shape:** `[B, 3]` — the unit gravity vector `[0, 0, -1]` expressed in the robot's body frame.

**Source:** `mdp.projected_gravity` computes this from the root body's quaternion. On a real
robot, this is estimated from the IMU (a combination of gyroscope integration and
accelerometer readings).

**What it encodes:** When the robot is perfectly upright, this vector is `[0, 0, -1]` in
body coordinates. When tilted forward, the z-component decreases and the x-component
becomes nonzero. The policy uses this to know which direction "down" is relative to the body.

**Why not use Euler angles:** Euler angles have singularities (gimbal lock). The gravity
vector as a 3D unit vector is continuous everywhere and has no ambiguity.

**Noise ±0.05:** Small noise on gravity, corresponding to ±0.05 rad (≈3°) of attitude error.

### `joint_pos` — Joint Angles Relative to Default (N values)

**Shape:** `[B, N]` where N is the number of controlled joints.

**Source:** `mdp.joint_pos_rel` computes `q - q_default` for every controlled joint.
`q_default` is the HOME_KEYFRAME joint positions.

**Why relative, not absolute:** The absolute joint position is less useful because the
policy's goal is "reach default" not "reach some absolute angle." By returning the error
`q - q_default`, the observation directly encodes the task signal. When all entries are
near zero, the task is done.

**Noise ±0.01 rad:** Joint encoders on the G1 have approximately 12-14 bit resolution over
a ~π rad range, giving ≈0.001 rad quantization. The ±0.01 rad noise is larger and
also captures calibration errors and MJCF vs. real-robot kinematic mismatch.

### `joint_vel` — Joint Velocities (N values)

**Shape:** `[B, N]` — velocity of each controlled joint in rad/s.

**Source:** `mdp.joint_vel_rel` returns `vel - vel_default`. Since `vel_default=0.0`, this
is just the current velocity. (The `_rel` suffix implies relative to default, but the default
velocity is zero.)

**Why include velocity:** The policy is approximately computing a second-order controller.
If it only sees position error (`joint_pos`), it cannot distinguish between:
- "Error is 0.3 rad but decreasing rapidly" → do nothing, it will converge
- "Error is 0.3 rad and stationary" → apply more force

Joint velocity is essential for damping: the policy must apply less torque when the joint is
already moving toward the target.

**Noise ±1.5 rad/s:** This is the largest relative noise magnitude in the actor observation.
Velocity is not directly measured on the G1 — it is computed from finite differences of
encoder readings (and filtered). Finite differencing amplifies encoder noise. The large noise
reflects this reality.

### `actions` — Previous Action (N values)

**Shape:** `[B, N]` — the action sent to the robot on the previous policy step.

**Source:** `mdp.last_action` returns the previous action tensor (no noise).

**Why include it:** Adding the previous action to the observation effectively gives the
policy a one-step memory without using recurrent networks. The policy can compute
action-rate terms in its head (even though they also appear in the reward) and can condition
on "what did I just do" to smooth its behavior.

**Total actor observation dimension:**
```
base_ang_vel:    3
projected_gravity: 3
joint_pos:       N (e.g., 29 for G1 full body)
joint_vel:       N
actions:         N
Total:           6 + 3N = 6 + 87 = 93 dimensions
```

---

## 6.3 Critic Observations

```python
critic_terms = {
    **actor_terms,          # Everything the actor sees
    "base_lin_vel": ...,    # Additional privileged terms
    "foot_contact": ...,
    "foot_contact_forces": ...,
}
```

### `base_lin_vel` — Linear Velocity (3 values)

**Shape:** `[B, 3]` — (vx, vy, vz) in the robot's body frame.

**Why privileged:** Real-world linear velocity estimation requires either:
- An expensive external localization system (GPS, mocap)
- A state estimator that fuses IMU + leg kinematics (imprecise, drift-prone)
- Direct MuJoCo readout (perfect, but only in simulation)

The critic gets the true simulated linear velocity, which gives it a much better value
estimate. The actor must estimate behavior without knowing its exact speed.

**Noise ±0.5 m/s:** Even for the critic, we add some noise to prevent it from memorizing
exact velocity values.

### `foot_contact` — Binary Contact State (2 values)

**Shape:** `[B, 2]` — 0.0 or 1.0 for each foot.

**Source:** `mdp.foot_contact(env, sensor_name)` returns 1.0 if the sensor found a contact
for that foot.

**Why privileged:** The actor could in principle infer contact from joint forces and velocity
changes, but this requires complex inference. The critic is told directly.

### `foot_contact_forces` — Contact Force Magnitude (6 values, flattened)

**Shape:** `[B, 2*3]` — the 3D contact force vector for each foot, log-transformed.

**Source:**
```python
def foot_contact_forces(env, sensor_name):
    forces = sensor_data.force           # [B, N_contacts, 3]
    flat = forces.flatten(start_dim=1)   # [B, N*3]
    return torch.sign(flat) * torch.log1p(torch.abs(flat))
```

**Why log-transform:** Contact forces can range from 0 to several hundred Newtons (the G1
weighs ~60 kg, so normal stance force ≈ 300 N). A linear observation would be dominated
by large forces. `log1p(x) = log(1 + x)` compresses the range: 0→0, 100→4.6, 1000→6.9,
making the network see a more uniform distribution.

**Total critic observation dimension:**
```
All actor terms:       93
base_lin_vel:          3
foot_contact:          2
foot_contact_forces:   6
Total:                 104 dimensions
```

---

## 6.4 `ObservationGroupCfg` Parameters

```python
ObservationGroupCfg(
    terms=actor_terms,
    concatenate_terms=True,   # Flatten all terms into one vector
    enable_corruption=True,   # Apply noise (actor=True, critic=False)
    history_length=1,         # No temporal stacking
)
```

### `concatenate_terms=True`

All terms are concatenated along dim=1 into a single flat tensor `[B, D_total]`. This is
what the neural network expects.

If `concatenate_terms=False`, the manager would return a dict of tensors, one per term.
This is used in some implementations where different network heads process different
observation subsets, but not in this codebase.

### `enable_corruption=True/False`

- `True` for the actor: noise is applied during training. At evaluation time in play mode,
  the config sets `enable_corruption=False` explicitly.
- `False` for the critic: the critic always sees clean observations (the noise advantage
  would be meaningless since the critic is only used for value estimation, not deployment).

### `history_length=1`

With `history_length=1`, only the current observation is used (no temporal stacking).

With `history_length=N`, the last N observations are stacked: `[B, N*D]`. This gives the
policy a form of memory without recurrent networks. For tasks that require tracking dynamics
over time (e.g., estimating velocity from position changes), history_length > 1 is useful.

For the transition task, `history_length=1` is sufficient because:
- Joint position error and velocity are already in the observation (no need to estimate them)
- The policy does not need to remember the trajectory, only the current state

---

## 6.5 What Is NOT in the Actor Observations (and Why)

### No velocity command

The transition policy has no velocity command because the goal is always the same: reach
the default pose. Adding a "zero command" observation would waste input dimensions and add
noise for no benefit.

### No foot contact

The actor does not observe foot contact. The policy can infer approximate contact state from
the joint positions (if knees are extended and ankles at default, feet are probably on the
ground). In practice, the `both_feet_contact` reward guides the policy toward keeping both
feet down without requiring contact in the observation.

### No terrain scan / height field

Flat terrain only. No height scan is needed because the ground is always at z=0.

### No gait phase signal

Gait phase (a sinusoidal clock signal) is used in walking policies to coordinate left and
right leg swing. For a standing policy, there is no cyclic gait, so no phase signal is needed.

---

## 6.6 Writing a Custom Observation Function

Observation functions follow a strict signature:

```python
def my_observation(
    env: ManagerBasedRlEnv,
    param1: float,          # Optional additional params
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),  # Standard pattern
) -> torch.Tensor:          # Must return [B, D] tensor
    asset: Entity = env.scene[asset_cfg.name]
    # ... compute something from asset.data ...
    return result  # shape [B, D]
```

**Rules:**
1. First argument must always be `env: ManagerBasedRlEnv`.
2. Return shape must be `[B, D]` where D ≥ 1. If your quantity is scalar per environment,
   return `[B, 1]` not `[B]`.
3. The function must be **deterministic** given the env state. Do not sample random numbers
   inside an observation function.
4. The function must not modify the env state. Observations are read-only.

**Registering it:**
```python
actor_terms["my_obs"] = ObservationTermCfg(
    func=my_observation,
    params={"param1": 0.5},
    noise=Unoise(n_min=-0.1, n_max=0.1),
)
```

The framework calls `my_observation(env, param1=0.5)` and adds the result to the
concatenated observation vector.

---

## 6.7 The `builtin_sensor` Pattern

```python
ObservationTermCfg(
    func=mdp.builtin_sensor,
    params={"sensor_name": "robot/imu_ang_vel"},
)
```

`builtin_sensor` is a generic observation function that reads any sensor registered on the
scene by name. The sensor name format is `"entity_name/sensor_type"`. The G1's IMU angular
velocity sensor is registered under `"robot/imu_ang_vel"`.

This pattern avoids writing a custom observation function for every sensor. The returned
tensor shape depends on the sensor: `imu_ang_vel` returns `[B, 3]`.

If the sensor name is wrong, the function will raise a `KeyError` at runtime. When debugging
"why is my observation size wrong," check that all `sensor_name` strings match actual
registered sensors in the entity's MJCF.
