# Chapter 11: Terminations — When Episodes End

Termination conditions define when an episode ends. Getting them right is critical:
too aggressive terminations prevent the policy from learning anything; too permissive
terminations let the robot keep receiving rewards while already in a failed state.

---

## 11.1 `TerminationTermCfg`

```python
@dataclass
class TerminationTermCfg:
    func:     Callable   # Returns bool tensor [B] — True means terminate
    time_out: bool       # True = this is a timeout (not a failure)
    params:   dict       # Extra keyword arguments for func
```

The framework ORs all termination functions together. If **any** termination fires for
environment `b`, that environment resets.

---

## 11.2 `time_out: TerminationTermCfg(func=mdp.time_out, time_out=True)`

```python
# From mjlab.envs.mdp:
def time_out(env) -> torch.Tensor:
    return env.episode_length_buf >= env.max_episode_length  # [B] bool
```

**`time_out=True`:** This flag tells the RL framework that this termination is due to
**time running out**, not due to the robot failing. This distinction matters for PPO's
advantage calculation.

**Why `time_out=True` matters:**
- In standard PPO, when an episode ends, the value estimate for the terminal state is
  zero (the episode is "done").
- But when an episode ends due to timeout (not failure), the agent would have continued
  receiving rewards if the episode were longer. The terminal state is not actually bad.
- PPO corrects for this by using the critic's value estimate to "bootstrap" the return
  at the end of a timeout episode, rather than setting it to zero.
- Without `time_out=True`, the policy would be biased against long episodes — even
  perfect behavior would be penalized because the episode ending would look like a failure.

**`episode_length_s=15.0` → `max_episode_length=750 steps` at 50 Hz:**
After 750 policy steps, this termination fires and the episode resets.

---

## 11.3 `fell_over: TerminationTermCfg(func=mdp.bad_orientation, ...)`

```python
terminations["fell_over"] = TerminationTermCfg(
    func=mdp.bad_orientation,
    params={"limit_angle": math.radians(70.0)},   # 70° ≈ 1.22 rad
)
```

```python
# From mjlab.envs.mdp (approximate):
def bad_orientation(env, limit_angle: float) -> torch.Tensor:
    # Compute angle between root z-axis and world z-axis
    # Returns True when the robot is tilted more than limit_angle
    projected_gravity = env.scene["robot"].data.projected_gravity_b  # [B, 3]
    # projected_gravity is gravity in body frame
    # When upright: [0, 0, -1]
    # The z-component = cos(tilt_angle), so tilt_angle = arccos(-projected_gravity_z)
    # Terminate if tilt_angle > limit_angle
    ...
```

**`limit_angle=70°`:** The episode ends when the robot tilts more than 70° from vertical.

**Why 70° (not 45° or 90°)?**
- At 45°, many natural poses (e.g., leaning forward to stand up from the floor) would trigger
  termination too early, cutting off promising trajectories.
- At 90°, the robot is horizontal — it is already on the ground and continuing to simulate
  is wasteful.
- At 70°, the robot is clearly falling but has not yet hit the ground, which gives the
  `is_terminated` penalty time to register before the physics becomes unreliable.

**No `time_out=True` flag:** This termination is a genuine failure. The terminal state value
is set to zero (the episode is over, no future reward expected). The policy receives the
`is_terminated` penalty reward alongside this termination.

---

## 11.4 The Termination-Reward Relationship

Notice that `is_terminated` is a **reward term**, not a termination term. The sequence is:

```
Step N:
  1. Compute observations
  2. Compute rewards (including is_terminated if terminated)
  3. Check terminations → set done[b] = True
  4. Return (obs, reward, done, info) to the runner

Step N+1:
  For environments where done[b] == True:
    Fire reset events
    Compute new initial observations
```

The `is_terminated` reward fires **on the same step that the termination occurs**, before
the reset. This ensures the penalty is included in the trajectory data that PPO uses for
the gradient update.

If you remove `is_terminated` from the rewards, the policy still terminates when bad
orientation is detected, but there is no direct penalty in the reward signal — only the
implicit penalty of losing all future positive rewards after termination. For this task,
the explicit `-200.0` penalty works better because it creates a strong, immediate signal.

---

## 11.5 Choosing a `limit_angle`

**For standing/transition tasks:** 70° is appropriate. The robot should not tilt more than
70° while trying to stand up.

**For walking tasks on rough terrain:** Consider 70°-80°. Walking up steep slopes can cause
the torso to lean significantly. Too aggressive termination on slopes would penalize normal
walking behavior.

**For motion tracking tasks:** May want a tighter threshold (e.g., 45°-60°) because the
reference motion defines valid body orientations, and large deviations indicate failure to
track.

---

## 11.6 Adding a Custom Termination

```python
def my_termination(
    env: ManagerBasedRlEnv,
    threshold: float,
) -> torch.Tensor:
    """Terminate if the robot's CoM height drops below threshold meters."""
    root_pos = env.scene["robot"].data.root_link_pos_w   # [B, 3]
    return root_pos[:, 2] < threshold  # [B] bool

terminations["low_com"] = TerminationTermCfg(
    func=my_termination,
    params={"threshold": 0.5},
)
```

**Rules:**
1. Return a `torch.Tensor` of dtype `bool` and shape `[B]`.
2. `True` means "this environment should terminate."
3. The function must not modify state — it is a pure sensor reading.

---

## 11.7 Disabling Terminations for Play

In play mode, you may want to observe the policy's behavior even after failure. The
`scripts/play.py` supports `--no-terminations` which does:

```python
if cfg.no_terminations:
    env_cfg.terminations = {}
```

Setting `terminations={}` means no termination ever fires. The episode only ends via the
`time_out` mechanism (which the runner handles differently). Since play mode sets
`episode_length_s = int(1e9)`, effectively the environment runs forever.

This is useful for:
- Watching the policy recover from falls (even if it falls, it keeps running)
- Identifying edge cases where the policy fails
- Comparing behavior with and without disturbances
