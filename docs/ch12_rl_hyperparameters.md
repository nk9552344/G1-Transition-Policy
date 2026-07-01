# Chapter 12: RL Hyperparameters — PPO Theory and Tuning Guide

This chapter explains what PPO (Proximal Policy Optimization) is, what every hyperparameter
in `RslRlOnPolicyRunnerCfg` does, and how to diagnose and fix common training problems.

---

## 12.1 PPO in One Page

PPO is an **on-policy** actor-critic algorithm. At each iteration:

1. **Collect rollout:** Run the current policy for `num_steps_per_env` steps across all
   environments. Store (obs, action, reward, done, value, log_prob) tuples.

2. **Compute advantages (GAE):**
   ```
   δ_t = r_t + γ * V(s_{t+1}) - V(s_t)     (TD error)
   A_t = δ_t + (γλ)δ_{t+1} + (γλ)²δ_{t+2} + ...    (GAE)
   ```
   Where γ=0.99 and λ=0.95. This is a weighted average of n-step returns.

3. **Update network (multiple passes):** For `num_learning_epochs` passes over the buffer,
   split into `num_mini_batches` batches. For each batch:
   ```
   ratio r_t = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)     (probability ratio)
   actor loss = -E[min(r_t * A_t, clip(r_t, 1-ε, 1+ε) * A_t)]
   critic loss = E[(V_θ(s_t) - R_t)²]
   entropy_bonus = H[π_θ(·|s_t)]
   total_loss = actor_loss + coef * critic_loss - coef * entropy_bonus
   ```

4. **Adaptive learning rate** (when `schedule="adaptive"`):
   If the KL divergence between old and new policy exceeds `desired_kl`, reduce `lr`.
   If KL is below `desired_kl / 2`, increase `lr`.

---

## 12.2 The `RslRlOnPolicyRunnerCfg`

```python
def unitree_g1_transition_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
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
        experiment_name="g1_transition",
        save_interval=100,
        num_steps_per_env=24,
        max_iterations=10001,
    )
```

---

## 12.3 Actor Architecture: `RslRlModelCfg`

### `hidden_dims=(512, 256, 128)`

A 3-layer MLP:
```
input (93 dims) → Dense(512) → ELU → Dense(256) → ELU → Dense(128) → ELU → Dense(N_actions=29) → tanh(log_std)
```

**Why these sizes:** The input is ~93 dimensional (for the full G1). The network needs
enough capacity to represent the inverse-kinematics-like function from joint errors to
corrective actions, but not so large that it overfits or trains slowly.

- 512 is wide enough to mix all input signals freely.
- Tapering (512→256→128) creates a bottleneck that forces the network to learn
  compressed representations.
- The last layer (128→N) is small, which prevents overfitting to specific observations.

**For simpler tasks** (fewer joints, fewer obs): You can reduce to (256, 128, 64).
**For more complex tasks** (whole-body motion, terrain awareness): Consider (512, 512, 256).

### `activation="elu"`

ELU (Exponential Linear Unit): `f(x) = x if x > 0, else α(e^x - 1)`.

ELU is preferred over ReLU for this task because:
- **No dead neurons:** ReLU neurons with negative inputs output exactly 0 and produce zero
  gradient. ELU has negative-regime gradient, keeping all neurons active.
- **Centered outputs:** ELU outputs are centered near zero (mean ≈ 0 for standard inputs),
  which helps with batch normalization and layer outputs.
- **Smooth gradient:** No discontinuity at x=0 (unlike ReLU), which produces smoother
  learning.

### `obs_normalization=True`

Maintains a running mean and variance of the observation. Each observation is
normalized: `obs_normalized = (obs - mean) / std`.

This is **critical** for this task because observations have very different scales:
- `projected_gravity` is always in `[-1, 1]` (normalized 3D vector)
- `joint_vel` can be in `[-10, 10]` rad/s
- `joint_pos_rel` is in `[-0.5, 0.5]` for most joints

Without normalization, the network's first layers would need to learn wildly different
scales for each input, which is very slow. With normalization, every input is approximately
`N(0, 1)` and the network receives balanced gradients.

**Important:** The running mean/variance is stored in the network checkpoint. When loading
a checkpoint for play or deployment, the normalization statistics are loaded automatically.
If you change the observation space (add/remove terms), you **must** retrain from scratch —
the saved statistics from a previous run will be wrong.

### `distribution_cfg`

The actor outputs the mean of a Gaussian distribution over actions:
```python
{
    "class_name": "GaussianDistribution",
    "init_std": 1.0,       # Initial standard deviation of the action distribution
    "std_type": "scalar",  # One std shared across all action dimensions
}
```

**`init_std=1.0`:** At the start of training, actions are drawn from `N(mean, 1.0)`.
Combined with the action scale (~0.3 rad per unit), this gives initial actions of
±0.3 rad (approximately matching the joint offset range, which is a good exploration scale).

**`std_type="scalar"`:** A single learnable log_std parameter is shared across all N action
dimensions. Alternative: `"diagonal"` would use a per-dimension std.

The std decreases during training as the policy becomes more confident. This is desirable:
early training needs wide exploration; late training needs precise control. The entropy
coefficient (0.01) prevents the std from collapsing to zero.

---

## 12.4 Algorithm Hyperparameters

### `clip_param=0.2` (ε)

The PPO clipping threshold. The actor loss clamps the probability ratio to `[1-ε, 1+ε]`:
```
ratio = π_new(a|s) / π_old(a|s)
clipped_ratio = clip(ratio, 0.8, 1.2)
actor_loss = -min(ratio * A, clipped_ratio * A)
```

**Effect:** Limits how much the policy can change in one update step. Prevents large,
destabilizing updates.

**0.2 is the standard value** for most PPO applications. If training is unstable (policy
collapses after a few updates), reduce to 0.1. If training is slow, try 0.3.

### `entropy_coef=0.01`

Encourages exploration by adding a bonus for high-entropy (uncertain) policies:
```
total_loss += entropy_coef × H[π]
```

At `entropy_coef=0.01`, the entropy bonus is small — the policy is allowed to become
fairly confident in its actions while retaining some exploration. If you reduce this to
zero, the policy may exploit specific trajectories and fail to explore better solutions.

**For the transition task:** 0.01 works well because the task has a single clear solution
(reach neutral pose). The entropy bonus prevents premature convergence to suboptimal
standing poses.

### `num_learning_epochs=5`

Number of full passes over the rollout buffer during each training iteration. Each pass
shuffles the data into `num_mini_batches` mini-batches.

**More epochs = more gradient updates per rollout = more sample efficiency**, but at the
risk of overfitting to the rollout data (leading to clipping everywhere and no learning).
5 epochs is the sweet spot for this class of tasks.

**If training diverges** (reward suddenly collapses): Reduce to 3-4 epochs. The policy may
be updating too aggressively.

### `num_mini_batches=4`

The rollout is split into 4 mini-batches, each processed independently with its own
gradient update. Total samples per iteration: `num_envs × num_steps_per_env`.
Mini-batch size: `num_envs × num_steps_per_env / 4`.

For `num_envs=4096` and `num_steps_per_env=24`:
- Total samples: 98,304
- Mini-batch size: 24,576

Larger mini-batches give more stable gradient estimates but require more memory.
Smaller mini-batches introduce noise that can help escape local optima.

### `learning_rate=1e-3` with `schedule="adaptive"`

**`schedule="adaptive"`:** The learning rate is adjusted based on the KL divergence between
the old and new policy at each update:
- If KL > `desired_kl` (0.01): reduce lr by multiplying by 0.9
- If KL < `desired_kl / 2` (0.005): increase lr by multiplying by 1.1

This prevents large learning rate steps when the policy is making big changes (which would
violate the PPO trust region), and allows faster learning when changes are small.

**Initial `lr=1e-3`:** Good for the first iterations. Will be adapted automatically.

### `gamma=0.99`

The discount factor. Future rewards at distance k are weighted by `0.99^k`.

At `γ=0.99` and 750-step episodes:
- `0.99^750 ≈ 0.00055` — rewards at the very end of the episode have minimal impact.
- `0.99^100 ≈ 0.37` — rewards 100 steps away have 37% weight.

**Effect of choice:**
- `γ=0.99`: Good for 15 s episodes. The discount is tight enough that the policy focuses
  on the near future while still being aware of the full episode.
- `γ=0.995` or higher: Appropriate for longer episodes (walking tasks).
- `γ=0.95`: Too aggressive for long episodes — the policy ignores rewards more than ~60
  steps ahead.

### `lam=0.95`

The GAE lambda parameter. Controls the bias-variance tradeoff in advantage estimation:
- `λ=1.0`: High variance, unbiased (equivalent to Monte Carlo returns)
- `λ=0.0`: Low variance, high bias (equivalent to 1-step TD)
- `λ=0.95`: Standard balance

**Technical:** With `λ=0.95` and `γ=0.99`, the effective horizon of advantage estimates is
`1 / (1 - γλ) = 1 / (1 - 0.9405) = ~17 steps`. The policy is trained to look ~17 steps
ahead when computing how good an action is.

### `desired_kl=0.01`

The target KL divergence between policy updates. The adaptive learning rate maintains
`KL(π_old || π_new) ≈ 0.01`.

A KL of 0.01 means the old and new policies are "close" — the average change in log
probability per action is small. This prevents catastrophic updates.

**If training is unstable:** Reduce to 0.005. **If training is slow:** Try 0.02.

### `max_grad_norm=1.0`

Gradient clipping threshold. If the total gradient norm exceeds 1.0, it is normalized to 1.0.

This prevents exploding gradients, which can occur when:
- The loss is very large (e.g., `is_terminated` fires for many environments simultaneously)
- The network is in a bad state (after exploring an unusual part of the observation space)

---

## 12.5 Runner-Level Parameters

### `num_steps_per_env=24`

Each iteration collects 24 steps of experience per environment before updating the network.

For `num_envs=4096`: 4096 × 24 = 98,304 samples per update.

**At 50 Hz (0.02 s/step):** 24 steps = 0.48 s of simulated experience per update.
This is shorter than one gait cycle (0.6 s), which means each update has incomplete gait
information. For walking tasks, increasing to 48 (one gait cycle) can help. For the
transition task (no gait), 24 is fine.

**Tradeoff:**
- More steps per env = more data per update = more stable gradients = slower training
- Fewer steps per env = more frequent updates = potentially faster learning = more variance

### `max_iterations=10001`

Total number of training iterations. Each iteration processes `num_envs × num_steps_per_env`
environment steps.

Total simulated steps: `10001 × 4096 × 24 ≈ 1 billion steps`
Simulated time: `1 billion × 0.02 s = 20 million seconds ≈ 231 days` (of simulated time)

This sounds large, but each GPU step processes all 4096 environments in parallel, so the
wall-clock time is much shorter. With a high-end GPU and 4096 environments, 10,000 iterations
takes approximately 2-8 hours.

### `save_interval=100`

Saves a checkpoint every 100 iterations. Also triggers ONNX export (via the custom
`TransitionOnPolicyRunner.save` method). Checkpoints are saved to:
```
logs/rsl_rl/g1_transition/{datetime}/model_{iteration}.pt
```

### `experiment_name="g1_transition"`

The top-level log directory name. All runs for this task go to:
```
logs/rsl_rl/g1_transition/{run_datetime}/
```

---

## 12.6 Diagnosing Training Problems

### Problem: Policy collapses (reward suddenly drops)

**Possible causes:**
1. Learning rate too high → reduce `desired_kl` to 0.005
2. Too many learning epochs → reduce `num_learning_epochs` to 3
3. Conflicting reward terms → check that penalties don't dominate positive rewards

### Problem: Slow convergence (reward barely increases)

**Possible causes:**
1. std too small for the initial error → increase `pose_convergence` std
2. Learning rate too low → increase `desired_kl` to 0.02
3. Too few environments → increase `num_envs`
4. Observation not informative enough → check that `joint_pos_rel` has the correct joints

### Problem: Policy learns to fall immediately

**Possible causes:**
1. `is_terminated` weight not large enough to deter falling
2. Physics timestep too large → robot starts exploding
3. Initial action scale too large → robot jerks violently at episode start

### Problem: Policy oscillates around the target

**Possible causes:**
1. `joint_vel_penalty` weight too small → increase from -0.01 to -0.05
2. `action_rate_l2` weight too small → increase from -0.05 to -0.2
3. PD damping too low → increase `DAMPING_RATIO` in motor constants
4. `std` in `pose_convergence` too small → the policy sees a very sharp gradient near
   zero and oscillates across it
