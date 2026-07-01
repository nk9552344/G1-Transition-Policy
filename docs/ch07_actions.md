# Chapter 7: Actions — How the Policy Controls the Robot

The action space is what the policy outputs. Understanding actions means understanding the
entire pipeline from neural network output to physical torque.

---

## 7.1 The Action Pipeline

```
Neural network output:  a ∈ ℝ^N  (raw, approximately [-1, 1])
         ↓
Scale multiplication:   a_scaled = a * scale  (per-joint scale)
         ↓
Default offset:         target = q_default + a_scaled
         ↓
PD controller (200 Hz): torque = Kp * (target - q) + Kd * (0 - q_dot)
         ↓
Effort limit clamp:     torque = clamp(torque, -effort_limit, effort_limit)
         ↓
MuJoCo physics step:    dynamics integration
```

Each step of this pipeline has design implications. A mistake at any step produces a
policy that is difficult or impossible to train.

---

## 7.2 `JointPositionActionCfg`

```python
actions = {
    "joint_pos": JointPositionActionCfg(
        entity_name="robot",           # Which scene entity to control
        actuator_names=(".*",),        # Regex: select all actuators
        scale=0.25,                    # Placeholder (overridden per robot)
        use_default_offset=True,       # target = q_default + action * scale
    )
}
```

### `entity_name="robot"`

The entity to control must match a key in `cfg.scene.entities`. The G1 override sets
`cfg.scene.entities = {"robot": get_g1_robot_cfg()}`, so `"robot"` is the correct name.

### `actuator_names=(".*",)`

This regex is matched against the names of the actuators defined in the robot's
`EntityArticulationInfoCfg`. The `.*` matches all actuators. You could instead write
`(".*_leg_.*",)` to control only leg joints, leaving arm joints at their default positions.

The matched actuators determine the **dimension of the action space**: if there are 29
matched joints, the network must output 29 values.

### `scale`

After the G1 override:
```python
joint_pos_action.scale = G1_ACTION_SCALE
```

`G1_ACTION_SCALE` is a `dict[str, float]` where keys are actuator name patterns and values
are per-joint scales (derived from motor specs as discussed in Chapter 4).

When `scale` is a dict, the framework expands it: each joint's scale is looked up by matching
the joint name against the dict keys. If multiple dict keys match a joint name, the last
matching one wins.

**Alternative — scalar scale:**
```python
scale=0.25  # All joints use the same scale
```
This is the placeholder value in the base config. It is a reasonable default if you do not
have per-joint motor data, but gives suboptimal results because joints with small effort
limits (wrists, 5 N·m) would be given the same scale as strong joints (knees, 139 N·m).

### `use_default_offset=True`

With this flag:
```
target_joint_pos = q_default + action * scale
```

The `q_default` here is the HOME_KEYFRAME joint positions — the positions the robot should
return to. The policy's output `action` is thus the **deviation from home**, not an absolute
position.

If `use_default_offset=False`:
```
target_joint_pos = action * scale
```
The output would need to encode absolute positions, which is much harder to learn (different
joints have different offsets in their natural configuration).

---

## 7.3 The PD Controller

The MuJoCo `BuiltinPositionActuator` runs a PD controller at the physics timestep (200 Hz):

```
torque = Kp * (target - current_pos) + Kd * (0 - current_vel)
torque = clamp(torque, -effort_limit, effort_limit)
```

**Why `Kd * (0 - current_vel)` not `Kd * (target_vel - current_vel)`?**

The PD target velocity is zero because we want the joint to slow down as it approaches the
target. In continuous-time optimal control terms, we are asking the joint to "arrive and
stop." If we set `target_vel = 0` (not `target_vel = velocity_toward_target`), the damping
term always opposes motion, which ensures the system is overdamped and stops at the target
without oscillation.

---

## 7.4 How Action Scale Affects Learning

If the action scale is too **large**:
- Small network outputs cause large joint movements.
- The policy must learn to output very small values, which is hard for a policy initialized
  with random weights (which output ~N(0,1) values).
- Early training is chaotic and the robot falls immediately.
- The `is_terminated` penalty fires constantly, providing a poor learning signal.

If the action scale is too **small**:
- The policy cannot achieve large enough joint deviations to complete the task.
- Example: if the joint is offset by 0.5 rad and the max action * scale is 0.05 rad, the
  policy will spend hundreds of steps converging and the episode ends before convergence.
- `pose_convergence` is consistently near zero, providing no gradient.

The formula `scale = 0.25 * effort_limit / stiffness` gives a scale where:
- A unit action (±1.0) moves the joint approximately ±0.25 of the maximum PD-range.
- The maximum PD-range is the deflection that saturates the effort limit: `effort/stiffness`.
- With the initial random policy (outputs ≈ N(0, 1)), the joints will deflect by about
  `0.25 × effort/stiffness` rad on average — large enough to explore but not so large as to
  immediately destabilize the robot.

---

## 7.5 The Action Dimension and Ordering

The action tensor is ordered to match the actuator ordering in the `EntityArticulationInfoCfg`.
In the G1:
1. 5020 motors: elbows, shoulder joints, wrist roll
2. 7520-14 motors: hip pitch, hip yaw, waist yaw
3. 7520-22 motors: hip roll, knee
4. 4010 motors: wrist pitch, wrist yaw
5. WAIST actuator: waist pitch, waist roll
6. ANKLE actuator: ankle pitch, ankle roll

The MJCF joint names determine the order within each group. The framework resolves this
ordering automatically — you never need to hand-craft action ordering.

---

## 7.6 Clipping Actions Before Sending to the Environment

In `play.py` and `train.py`, the environment is wrapped with:
```python
env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
```

`clip_actions` is a boolean (default `True`). When `True`, the network output is clipped
to `[-1, 1]` before being passed to the environment. Since the scale is applied inside the
environment, this clip is applied to the raw network output.

Clipping prevents the policy from sending extreme positions during early training when the
network is poorly initialized. It costs nothing during inference (a well-trained policy
stays within [-1, 1] naturally due to the entropy coefficient encouraging moderate outputs).

---

## 7.7 Adding Arm Control vs Leg Control Only

If you want to train a policy that only controls leg joints (keeping arms fixed at home):

```python
actions = {
    "joint_pos": JointPositionActionCfg(
        entity_name="robot",
        actuator_names=(
            ".*_hip_roll_joint",
            ".*_hip_pitch_joint",
            ".*_hip_yaw_joint",
            ".*_knee_joint",
            ".*_ankle_pitch_joint",
            ".*_ankle_roll_joint",
        ),
        scale=G1_LEG_ACTION_SCALE,  # dict with only leg joints
        use_default_offset=True,
    )
}
```

The arm joints not listed here will be held at their `q_default` positions by the PD
controller (since no policy-generated target is sent to them, they stay at their last target,
which is `q_default` from the keyframe initialization).

This reduces the action dimension from 29 to 12 for the G1, which can speed up learning
for tasks that do not require arm motion.
