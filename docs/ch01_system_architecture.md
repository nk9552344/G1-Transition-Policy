# Chapter 1: System Architecture — The Complete Data Flow

Understanding the data flow is the most important prerequisite for debugging any policy.
If a reward is not working, you need to know exactly where in the pipeline it is computed.
If the policy is not learning, you need to know what the policy actually sees as input.

---

## 1.1 The Three-Level Hierarchy

The system has three distinct levels of abstraction:

```
Level 1: Physics (MuJoCo)
   └── Simulates the robot body in continuous time at 5 ms per step

Level 2: Manager-Based Environment (mjlab)
   └── Wraps the physics into a structured RL interface:
       observations, actions, rewards, resets, terminations

Level 3: RL Runner (RSL-RL via mjlab)
   └── Runs PPO on top of the environment:
       collects rollouts, computes advantages, updates neural network weights
```

Each level knows nothing about the levels above it. MuJoCo does not know about rewards.
The environment does not know about PPO. The runner does not know about contact geometry.

---

## 1.2 The Inner Loop (one environment step)

Every call to `env.step(action)` does the following in order:

```
1. APPLY ACTION
   action (tensor, shape [B, num_joints]) arrives from the policy network.
   JointPositionActionCfg converts it to a joint position target:
     target = default_joint_pos + action * scale
   The PD controller in each actuator computes torque:
     torque = stiffness * (target - current_pos) + damping * (0 - current_vel)
   This torque is applied to MuJoCo.

2. STEP PHYSICS  (decimation times)
   MuJoCo integrates the equations of motion.
   The policy step is 4× slower than the physics step (decimation=4):
     physics dt = 0.005 s   →  200 Hz
     policy dt  = 0.020 s   →  50 Hz
   Each policy step runs 4 sub-steps of MuJoCo.

3. FIRE INTERVAL EVENTS
   Events registered with mode="interval" may fire on this step.
   Example: push_robot fires every 8–10 s of episode time.

4. COMPUTE OBSERVATIONS
   For each term in each observation group (actor, critic):
     call the term's function with the current env state
     optionally add uniform noise (if enable_corruption=True)
   Concatenate all terms into a single flat tensor per group.

5. COMPUTE REWARDS
   For each term in the reward dict:
     call the term's function
     multiply by the term's weight
   Sum all weighted terms to get the scalar reward for this step.

6. CHECK TERMINATIONS
   For each term in the termination dict:
     evaluate the term's function (returns bool tensor)
   If any term fires:
     the episode for that environment resets on the next step.

7. RETURN (obs, reward, done, info)
   The runner receives these and stores them in the rollout buffer.
```

---

## 1.3 The Outer Loop (one PPO iteration)

```
COLLECT ROLLOUT
  For num_steps_per_env (24) steps:
    policy(actor_obs) → action
    env.step(action)  → next_obs, reward, done, info
    store (obs, action, reward, done, value, log_prob) in buffer

COMPUTE RETURNS AND ADVANTAGES
  Bootstrap terminal value with critic(critic_obs)
  GAE-lambda (γ=0.99, λ=0.95) computes per-step advantages A_t

UPDATE NETWORK (num_learning_epochs=5 passes over the buffer)
  Split buffer into num_mini_batches=4 mini-batches
  For each mini-batch:
    actor loss  = -E[min(r_t * A_t, clip(r_t, 1-ε, 1+ε) * A_t)]
    critic loss = value_loss_coef * E[(V(s) - R_t)²]
    entropy_bonus = entropy_coef * H[π]
    total_loss = actor_loss + critic_loss - entropy_bonus
    gradient step via Adam (lr=1e-3, adaptive schedule)

RESET DONE ENVIRONMENTS
  Environments that terminated fire mode="reset" events:
    reset_base: randomize root position and yaw
    reset_robot_joints: add ±0.5 rad random offsets to joints
```

---

## 1.4 The Observation Split: Actor vs Critic

The policy uses **asymmetric actor-critic**. The actor (deployed on the real robot) sees a
noisy, minimal observation. The critic (only exists during training) sees a richer, clean
observation including privileged information like contact forces.

```
Actor obs (what the real robot will see):
  base_ang_vel      — IMU angular velocity (noisy)
  projected_gravity — gravity in body frame (noisy)
  joint_pos         — joint angles relative to default (noisy)
  joint_vel         — joint velocities (noisy)
  actions           — previous action (no noise)

Critic obs (everything the actor sees, plus):
  base_lin_vel          — IMU linear velocity (noisy but less conservative)
  foot_contact          — binary contact per foot (clean)
  foot_contact_forces   — net contact force magnitude (clean, log-scaled)
```

The critic learns a better value function because it has access to ground-truth physics
information. The actor learns to imitate what the critic would do, guided by the advantages
computed from the critic's value estimates. At deployment, only the actor network is used.

---

## 1.5 The Reset-and-Randomize Cycle

Every time an environment terminates (robot fell) or times out (15 s elapsed):

```
reset_base fires:
  - Randomly scatter base position: x, y ∈ [-0.5, 0.5] m
  - Randomly scatter yaw:           yaw ∈ [-π, π] rad
  - Zero velocities

reset_robot_joints fires:
  - Start from HOME_KEYFRAME joint positions
  - Add uniform noise U(-0.5, +0.5) rad to every joint
  - Set all joint velocities to 0.0
```

The random joint offsets are the core training signal source. The policy sees a robot that is
never in exactly the same position at episode start, so it must learn a general recovery
strategy, not a specific motion sequence.

---

## 1.6 How Tensors Flow Through the Code

Most functions in this codebase work on batched tensors. The batch dimension `B` is
`num_envs` — all environments are simulated in parallel on the GPU.

```
B   = num_envs (e.g. 4096 during training)
N   = number of joints (G1 full: 29 DOF)
T   = decimation (4 physics substeps per policy step)
H   = history_length (1 in this policy)

Typical tensor shapes:
  joint_pos:      [B, N]     — one pos per joint per env
  contact_found:  [B, 2]     — two feet, each 0 or 1
  rewards:        [B]        — one scalar reward per env
  observations:   [B, D_obs] — flat obs vector per env
```

Every reward function receives `env: ManagerBasedRlEnv` and returns `torch.Tensor` of shape
`[B]`. The reward manager multiplies by the weight scalar and sums across terms, producing a
final `[B]` reward tensor.

---

## 1.7 Time Scales and Why They Matter

| Quantity | Value | Notes |
|---------|-------|-------|
| MuJoCo physics timestep | 5 ms (200 Hz) | Set in `MujocoCfg.timestep` |
| Policy step dt | 20 ms (50 Hz) | `decimation=4` × 5 ms |
| Episode length | 15 s | `episode_length_s=15.0` |
| Steps per episode | 750 | 15 s / 20 ms |
| Push interval | 8–10 s | `interval_range_s=(8.0, 10.0)` |

The 50 Hz policy rate is important because:
- It matches typical real-robot control rates for the G1.
- At 50 Hz, joint velocities change slowly enough for the policy to track them.
- The PD controller runs at 200 Hz internally, so it can damp high-frequency oscillations
  that the policy does not see.

The 4-substep decimation is the mechanism that lets you run a 200 Hz physics simulation
while only querying the neural network at 50 Hz. The network runs 4× less often → 4× cheaper.
