# Chapter 9: Rewards — Theory, Design, and Every Function Explained

Reward design is the hardest part of RL policy engineering. Getting it wrong leads to:
- **Reward hacking**: The policy maximizes the number without achieving the goal
- **Gradient starvation**: The reward is always near zero, providing no learning signal
- **Conflicting rewards**: Terms fight each other, causing oscillation or suboptimal behavior

This chapter covers the theory behind every reward function used in the transition policy,
how to compute the right weight and std values, and how to write new reward functions.

---

## 9.1 The Structure of a Reward Term

Every reward term has the form:
```
contribution_per_step = weight × func(env, **params)
```

The total reward at step t is:
```
R_t = Σ (weight_i × func_i(env))
```

PPO optimizes the **discounted sum of future rewards**:
```
G_t = Σ_{k=0}^{∞} γ^k × R_{t+k}   (γ = 0.99)
```

For a 15 s episode at 50 Hz (750 steps), and with γ=0.99:
```
Σ_{k=0}^{749} 0.99^k ≈ 74.5  (geometric sum)
```

So the cumulative return from a constant reward of 1.0/step is approximately 74.5. The
maximum cumulative return from `pose_convergence` (weight=2.0, max value=1.0) is 2.0 × 74.5
= 149. This is the expected return for a policy that instantly reaches and holds neutral.

The `is_terminated` penalty (-200.0) fires once per episode for failed episodes. Its
effective cumulative value is approximately -200. This is slightly larger than the maximum
achievable positive return (149), which correctly incentivizes survival over the episode.

---

## 9.2 The Exponential Gaussian Pattern

Many reward functions in this codebase use the form:
```
reward = exp(-error² / σ²)
```

This is a Gaussian bell curve centered at `error=0`:
- At `error=0`: reward = 1.0 (maximum, perfectly at target)
- At `error=σ`: reward = exp(-1) ≈ 0.37 (one std from target)
- At `error=2σ`: reward = exp(-4) ≈ 0.018 (two stds from target)

### Why Exponential, Not Linear

A linear reward (`reward = max(0, 1 - |error| / range)`) has:
- A sharp cliff at `error = range` (reward drops to zero)
- A constant gradient everywhere else (same encouragement near and far)

The exponential has:
- A smooth, dense gradient everywhere
- Larger gradient magnitude when close to the target (encourages precision)
- Nonzero gradient even far from the target (provides signal when the robot starts far away)

At `error = σ`, the gradient is `d/d(error²) × exp(-error²/σ²) = -1/σ² × exp(-1) ≈ -0.37/σ²`.
This gradient is meaningful even when the robot is not close to neutral.

### Choosing σ (std)

The `std` parameter controls the **effective learning radius**:
- Larger `std` → reward is nonzero even when far from target → easier early learning
- Smaller `std` → reward requires precision → harder to achieve initially, but more precise

For `pose_convergence` with `std=0.25`:
```
At 0.5 rad error (initial):  reward = exp(-0.5²/0.25²) = exp(-4) ≈ 0.018  (small but nonzero)
At 0.1 rad error:            reward = exp(-0.1²/0.25²) = exp(-0.16) ≈ 0.85  (strong signal)
At 0.0 rad error (perfect):  reward = 1.0
```

This is well-designed: the robot gets some reward signal even at the initial 0.5 rad
offset, and a strong reward gradient when it is close to neutral. If `std=0.05`, the reward
at 0.5 rad would be `exp(-100) ≈ 0`, providing no gradient for a far-away robot.

### When to Use `mean` vs `sum` in the Error

```python
# pose_convergence uses mean:
mse = torch.mean(torch.square(q - q_default), dim=1)
return torch.exp(-mse / std**2)
```

Using `mean` ensures the reward magnitude does not scale with the number of joints. If you
use `sum`, a robot with more joints would get a lower reward for the same per-joint error.
This makes `std` easier to reason about: "std is the typical per-joint RMS error at which
the reward is 0.37."

---

## 9.3 Every Reward Function Explained

### `pose_convergence` (+2.0, weight=2.0)

```python
def pose_convergence(env, std, asset_cfg):
    q = asset.data.joint_pos[:, asset_cfg.joint_ids]       # Current positions
    q_default = asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    mse = torch.mean(torch.square(q - q_default), dim=1)   # Per-env mean sq error
    return torch.exp(-mse / std**2)                         # [B], range [0, 1]
```

**Purpose:** Primary learning signal. Drives all joints toward HOME_KEYFRAME.

**Weight choice (2.0):** This is the primary positive reward. Its maximum contribution
per step is 2.0. The other penalties should not dominate this in magnitude during good
behavior. If you increase penalties too much, the policy may find it better to just fall
(collecting the -200 once) than to stay standing and accumulate small penalties.

**std choice (0.25):** See analysis in 9.2.

**Debugging:** If `mean(pose_convergence)` in logs is stuck near 0.02 (the value at 0.5 rad
initial error), the policy is not converging. If stuck near 0.85 (value at 0.1 rad), the
policy is doing well but cannot achieve the final 0.1 rad precision — consider reducing std.

---

### `both_feet_contact` (+0.5)

```python
def both_feet_contact(env, sensor_name):
    sensor: ContactSensor = env.scene[sensor_name]
    in_contact = sensor.data.found > 0   # [B, 2] bool tensor
    return in_contact.all(dim=1).float() # [B] — 1.0 iff both feet in contact
```

**Purpose:** Encourages maintaining a stable two-foot stance. Without this, the policy might
discover that it can achieve high `pose_convergence` by hopping or shifting all weight to
one foot, which is unstable.

**Weight choice (0.5):** Secondary positive reward. Smaller than `pose_convergence` so the
primary goal remains joint convergence. The maximum contribution is 0.5/step.

**Important:** This reward returns either 0.0 or 1.0 (binary). It provides no gradient
information about "which direction to move" — only "are both feet down or not." The gradient
comes from the fact that episodes where both feet are consistently down receive 0.5 more
reward per step (×750 steps = 375 cumulative reward), while episodes where one foot lifts
frequently receive less. PPO's advantage estimates capture this difference.

---

### `body_orientation_l2` (-2.0)

```python
def body_orientation_l2(env, asset_cfg):
    body_quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :]  # [B, 1, 4]
    projected_gravity_b = quat_apply_inverse(body_quat_w, gravity_w)      # [B, 3]
    xy_squared = torch.sum(torch.square(projected_gravity_b[:, :2]), dim=1)
    return xy_squared  # [B], range [0, 2] approximately
```

**Purpose:** Penalize the torso tilting. A tilted torso means the robot is about to fall.

**What it measures:** `projected_gravity_b[:, :2]` gives the x and y components of gravity
in the body frame. When upright, gravity is purely in the z direction: `[0, 0, -g]`, so
the x and y components are zero. When tilted at angle θ, the xy magnitude is `sin(θ)`.

For small angles: `sin(θ) ≈ θ`, so the penalty is approximately `θ²` (radians squared).

**Weight choice (-2.0):** Strong penalty on tilting. For a tilt of 30° (0.52 rad):
- `sin(30°) = 0.5`
- Penalty magnitude = `0.5² = 0.25` per component → `sum = 0.5`
- Weighted = `-2.0 × 0.5 = -1.0` per step

At -1.0/step over 750 steps, that's -750 cumulative penalty. The policy strongly prefers
to stay upright.

**Note:** Only xy components are penalized, not z. The z component of projected gravity
(`-cos(θ) ≈ -1` for small θ) does not indicate tilting — it is just gravity's magnitude.

---

### `joint_vel_penalty` (-0.01)

```python
def joint_vel_penalty(env, asset_cfg):
    vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    return torch.sum(torch.square(vel), dim=1)  # [B], range [0, ∞)
```

**Purpose:** Penalize joint velocities. This serves two functions:
1. During convergence: discourages flailing and fast movements that overshoot the target.
2. After convergence: penalizes oscillation around the target — the policy learns to arrive
   and *stay* at neutral, not continuously oscillate.

**Weight choice (-0.01):** Very small weight. Typical joint velocities during convergence
are 0.5-2 rad/s per joint, and there are 29 joints. The sum of squared velocities might be
`29 × 1.0² = 29`, so the penalty is `-0.01 × 29 = -0.29/step`. This is small compared to
`pose_convergence` (+2.0) but accumulates over time.

**`sum` not `mean`:** Using `sum` makes the penalty larger for robots that are moving many
joints simultaneously. This is correct behavior: a robot flailing all 29 joints should be
penalized more than one moving only the hips.

---

### `body_ang_vel` (-0.05)

```python
def body_angular_velocity_penalty(env, asset_cfg):
    ang_vel = asset.data.body_link_ang_vel_w[:, asset_cfg.body_ids, :]  # [B, 1, 3]
    ang_vel_xy = ang_vel[:, :, :2]  # Only roll/pitch angular velocity
    return torch.sum(torch.square(ang_vel_xy), dim=(-1, -2))  # [B]
```

**Purpose:** Penalize the torso spinning or rocking. High angular velocity of the torso
means the robot is losing balance.

**Why only xy (roll/pitch rates, not yaw rate):** Yaw rotation (spinning in place) is not
particularly dangerous for balance. The robot can spin freely without falling. Roll and pitch
rates, however, directly threaten balance — a high pitch rate means the robot is tipping.

**Weight (-0.05):** Moderate penalty. At 1.0 rad/s roll+pitch combined: `-0.05 × 1.0 = -0.05/step`.

---

### `angular_momentum` (-0.025)

```python
def angular_momentum_penalty(env, sensor_name):
    angmom = env.scene[sensor_name].data  # [B, 3] whole-body angular momentum
    return torch.sum(torch.square(angmom), dim=-1)  # [B]
```

**Purpose:** Penalize whole-body angular momentum. Angular momentum is the sum of all body
link angular momenta about a common reference point. High angular momentum means the robot
as a system is rotating.

**Difference from `body_ang_vel`:** `body_ang_vel` measures only the torso's angular
velocity. `angular_momentum` accounts for the angular momentum of all limbs — arms swinging,
legs moving. A robot can have high total angular momentum even if the torso is momentarily
stationary (e.g., during arm swinging).

**Sensor name `"robot/root_angmom"`:** This is a builtin sensor that returns the
whole-body angular momentum vector.

**Weight (-0.025):** Small weight because angular momentum is naturally larger than other
penalty signals. The sensor returns SI units (kg·m²/s). A heavy robot moving fast will have
large angular momentum — we only penalize excesses.

---

### `is_terminated` (-200.0)

```python
# From mjlab.envs.mdp
def is_terminated(env) -> torch.Tensor:
    return env.termination_manager.terminated.float()  # [B], 1.0 if terminated
```

**Purpose:** Large penalty for falling over (triggering `bad_orientation` termination).
This is the most important penalty — it makes falling catastrophic compared to any temporary
reduction in other rewards.

**Weight (-200.0):** This fires once per episode for environments that fall. Its cumulative
impact is approximately -200 per episode. Compare to the maximum cumulative positive reward
≈ 2.0 × 74.5 = 149. The math: falling costs more than the entire positive reward, so the
policy is strongly incentivized to never fall.

**Critical distinction:** This fires when `bad_orientation` terminates the episode, NOT when
the `time_out` termination fires. Time-out terminations are marked with `time_out=True` and
PPO treats them differently (no penalty, value bootstrap). Falling terminations are actual
failures.

**Warning:** If this weight is too large (-1000), the policy may become overly conservative
and refuse to move at all (staying at any non-zero error to avoid the tiny risk of falling).
-200 is calibrated to be "falling is bad but not so bad that the policy freezes."

---

### `joint_acc_l2` (-2.5e-7)

```python
# From mjlab.envs.mdp
# Computes mean squared joint acceleration
```

**Purpose:** Penalize jerky joint trajectories. Joint acceleration is the rate of change
of joint velocity.

**Weight (-2.5e-7):** Extremely small weight. Joint accelerations during fast movements
can be very large (hundreds of rad/s²). The tiny weight ensures this penalty is noticeable
over many steps without dominating the reward signal.

---

### `joint_pos_limits` (-10.0)

```python
# From mjlab.envs.mdp
# Returns penalty for joint positions approaching or exceeding soft limits
```

**Purpose:** Prevent joint limit violations. The soft limit factor (0.9) means this penalty
activates at 90% of the hardware limit.

**Weight (-10.0):** Strong penalty. The penalty value is small when joints are well within
limits (near zero), but if a joint hits the soft limit, the contribution is significant.
This prevents the policy from casually violating limits during convergence.

---

### `action_rate_l2` (-0.05)

```python
# From mjlab.envs.mdp
# Returns sum squared difference between current and previous actions
```

**Purpose:** Penalize rapid changes in the action commands (the policy output). This
encourages smooth joint trajectory commands, which:
- Reduces wear on real hardware
- Prevents the PD controller from oscillating due to rapidly changing targets
- Makes the policy more stable under observation noise

**Weight (-0.05):** Moderate. Action changes are bounded by the clip to [-1, 1] per joint.
Maximum change per step per joint is 2.0, so max penalty is `-0.05 × N × 4 = -0.05 × 29 × 4
= -5.8/step` (if all joints reverse direction every step). In practice, policy actions
are much smoother — typical penalty is `-0.05 × ~0.1 = -0.005/step` per joint.

---

### `self_collisions` (-1.0, G1-specific)

```python
def self_collision_cost(env, sensor_name, force_threshold=10.0):
    data = sensor.data
    force_mag = torch.norm(data.force_history, dim=-1)   # [B, N, H]
    hit = (force_mag > force_threshold).any(dim=1)       # [B, H]
    return hit.sum(dim=-1).float()                       # [B]
```

**Purpose:** Penalize the arms or legs colliding with each other during the reconfiguration.
This is particularly relevant for the transition policy because the arms start at large
offsets (up to 0.5 rad) and may swing into the torso during convergence.

**`force_threshold=10.0 N`:** Small contact forces (gravity resting on a surface) are not
penalized — only contacts above 10 N. This prevents penalizing micro-contacts that are
unavoidable.

**`history_length=4`:** The sensor tracks contact over the last 4 physics substeps (20 ms).
The penalty counts how many substeps in this window had a collision exceeding threshold.
Maximum return = 4 (collision every substep). Weight -1.0: maximum penalty = -4/step.

**Weight (-1.0):** Light. Self-collision is undesirable but not catastrophic. The policy
should minimize it without it dominating the reward.

---

## 9.4 How to Choose Reward Weights

The key principle: **reward weights should reflect the relative importance of each objective,
expressed in comparable units**.

**Step 1: Identify the primary objective.** For the transition task, it is `pose_convergence`.
Set its weight to 1.0 or 2.0 as the anchor.

**Step 2: Estimate the typical magnitude of each term.** Run the environment with a zero-policy
or random policy for a few episodes and log each term's raw value. This tells you the scale.

**Step 3: Decide relative importance.** If `pose_convergence` = 0.5 (typical during early
training) and `body_orientation_l2` = 0.1 (small tilts), you want orientation to have less
impact than convergence. Set orientation weight such that `-weight × 0.1 << weight_pc × 0.5`.

**Step 4: Verify the termination penalty is dominant.** The cumulative sum of all positive
rewards over a successful episode should be less than the absolute value of `is_terminated`.

**Step 5: Iterate.** Most reward weight tuning is empirical. Watch the per-term contributions
in your training logs. If a penalty term has much larger magnitude than the primary reward,
the policy will optimize against it at the expense of the primary objective.

---

## 9.5 Writing a Custom Reward Function

```python
from __future__ import annotations
from typing import TYPE_CHECKING
import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

def my_reward(
    env: ManagerBasedRlEnv,
    std: float,                                    # Custom parameters
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Short description. Document the return range."""
    asset = env.scene[asset_cfg.name]
    # ... compute something ...
    return result  # shape [B], reward before weighting
```

**Rules:**
1. First argument must be `env: ManagerBasedRlEnv`.
2. Return a `torch.Tensor` of shape `[B]`.
3. **Do not apply the weight inside the function.** The framework does that. Return the raw
   value and set the weight in `RewardTermCfg`.
4. Keep the return value in a predictable range. For exponential rewards, `[0, 1]`. For
   L2 penalties, document the typical magnitude.
5. You can write to `env.extras["log"]` to log metrics from within reward functions:
   ```python
   env.extras["log"]["Metrics/my_metric"] = torch.mean(result)
   ```

**Registering it:**
```python
rewards["my_reward"] = RewardTermCfg(
    func=my_reward,
    weight=-0.1,
    params={"std": 0.5, "asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
)
```

**Stateful reward functions (class-based):**
Some rewards need to track state across steps (e.g., `feet_swing_height` tracks peak foot
height). Use a class with `__init__` and `__call__`:

```python
class MyStatefulReward:
    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
        # Initialize state tensors here (shape [B] or [B, N])
        self.state = torch.zeros(env.num_envs, device=env.device)

    def __call__(self, env: ManagerBasedRlEnv, **params) -> torch.Tensor:
        # Use self.state
        return result
```

The `__init__` receives the full `RewardTermCfg` so you can read all params.
