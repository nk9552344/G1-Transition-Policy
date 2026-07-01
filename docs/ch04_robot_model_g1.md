# Chapter 4: The G1 Robot Model — Hardware, Motors, and Action Scale

To write a good policy for the G1, you need to understand its physical structure: how many
joints it has, which motors drive them, what their mechanical properties are, and how those
properties translate into the `scale` parameter you write in the action config.
Getting the scale wrong is one of the most common reasons a policy fails to learn.

---

## 4.1 The G1 Anatomy

The G1 is a full-body humanoid with:
- **29 active joints** in the full (47-DOF MJCF) model, controlled by actuators
- **Two legs**: each has hip roll, hip pitch, hip yaw, knee, ankle pitch, ankle roll
- **Two arms**: each has shoulder pitch, shoulder roll, shoulder yaw, elbow, wrist roll,
  wrist pitch, wrist yaw
- **Torso**: waist yaw, waist pitch, waist roll

The MJCF file is at `src/assets/robots/unitree_g1/xmls/g1.xml`. This is the physical
description of the robot — link masses, inertias, joint limits, contact geometry, and
actuator specs. You generally do not need to edit this file.

### Home Keyframe

```python
HOME_KEYFRAME = EntityCfg.InitialStateCfg(
    pos=(0, 0, 0.8),              # CoM height 0.8 m above ground
    joint_pos={
        ".*_hip_pitch_joint":   -0.1,  # Hips slightly back
        ".*_knee_joint":         0.3,  # Knees slightly bent
        ".*_ankle_pitch_joint": -0.2,  # Ankles compensate for knee bend
        ".*_shoulder_pitch_joint": 0.35, # Arms slightly forward
        ".*_elbow_joint":        0.87, # Elbows at ~90°
        "left_shoulder_roll_joint":   0.18,  # Arms slightly out
        "right_shoulder_roll_joint": -0.18,  # (mirror-symmetric)
    },
    joint_vel={".*": 0.0},  # Zero velocity
)
```

All joints not listed default to `0.0`. This pose represents a stable, slightly flexed
standing configuration. It is the **target** that the transition policy must reach.

Any joint that deviates from these values by the random offset `U(-0.5, 0.5)` rad at
episode reset is the "problem" the policy must solve.

---

## 4.2 The Four Motor Families

The G1 uses four distinct brushless DC motor designs. Each motor family has:
- A rotor inertia (the rotational inertia of the spinning rotor itself)
- A gear ratio (how many rotor revolutions per joint revolution)
- A velocity limit (maximum joint speed under the motor's current limit)
- An effort (torque) limit at the joint output

| Family | Joints | Vel Limit | Torque Limit |
|--------|--------|-----------|--------------|
| 5020   | Elbows, shoulder pitch/roll/yaw, wrist roll | 37 rad/s | 25 N·m |
| 7520-14 | Hip pitch/yaw, waist yaw | 32 rad/s | 88 N·m |
| 7520-22 | Hip roll, knee | 20 rad/s | 139 N·m |
| 4010   | Wrist pitch/yaw | 22 rad/s | 5 N·m |
| WAIST  | Waist pitch/roll (2× 5020 in parallel linkage) | 37 rad/s | 50 N·m |
| ANKLE  | Ankle pitch/roll (2× 5020 in parallel linkage) | 37 rad/s | 50 N·m |

The hip roll and knee use the strongest motor (7520-22, 139 N·m) because those joints
bear the full body weight during stance phase. The wrists use the weakest (4010, 5 N·m)
because the hands are light.

---

## 4.3 Reflected Inertia from a Two-Stage Planetary Gearbox

The `reflected_inertia_from_two_stage_planetary` function computes the effective rotational
inertia seen at the joint output, accounting for all gear stages.

For a single-stage gear of ratio `r`, the rotor inertia `I_rotor` reflected to the output:
```
I_reflected = I_rotor * r²
```

The G1 motors use a **two-stage planetary gearbox** with three rotating elements:
1. The primary rotor (stage 0 gear ratio = 1, i.e., the motor shaft itself)
2. The planet carrier of stage 1 (gear ratio = 1 + (46/18) for 5020)
3. The planet carrier of stage 2 (gear ratio = 1 + (56/16) for 5020)

The total reflected inertia is the sum of all three, each scaled by the square of the
cumulative gear ratio from that stage to the output:

```
I_reflected = I_rotor_0 * r_0²   # Stage 0 contribution
            + I_rotor_1 * r_1²   # Stage 1 contribution (r_1 = gear_ratio[1])
            + I_rotor_2 * r_2²   # Stage 2 contribution (r_2 = gear_ratio[1] * gear_ratio[2])
```

For the 5020 motor:
```python
ROTOR_INERTIAS_5020 = (0.139e-4, 0.017e-4, 0.169e-4)  # kg·m²
GEARS_5020 = (1, 1 + 46/18, 1 + 56/16)                 # ≈ (1, 3.556, 4.5)
```

This calculation gives `ARMATURE_5020` in units of kg·m². The name "armature" in the
MuJoCo/mjlab context means "effective reflected inertia at the joint."

**Why this matters:** A joint with high reflected inertia is harder to accelerate. The motor
must work against its own inertia (reflected through the gears) as well as the load. If you
set the armature wrong, the simulated robot's dynamics will not match the real hardware.

---

## 4.4 Stiffness and Damping Derivation

The PD controller for each joint has two parameters: stiffness `Kp` and damping `Kd`.
Rather than tuning these manually, this codebase derives them from a second-order system
analogy.

**Target second-order system:**
```
I·q̈ + b·q̇ + k·q = 0
```
where `I = armature`, `k = stiffness`, `b = damping`.

This system has:
- Natural frequency: `ω_n = sqrt(k / I)` → `k = I · ω_n²`
- Damping ratio: `ζ = b / (2 · sqrt(k · I))` → `b = 2 · ζ · sqrt(k · I) = 2 · ζ · I · ω_n`

The code sets:
```python
NATURAL_FREQ = 10 * 2.0 * math.pi  # 10 Hz → ω_n = 62.83 rad/s
DAMPING_RATIO = 2.0                 # Critically overdamped × 2
```

For 5020 motors:
```python
STIFFNESS_5020 = ARMATURE_5020 * NATURAL_FREQ**2
# = ARMATURE_5020 * (62.83)² ≈ ARMATURE_5020 * 3948
DAMPING_5020 = 2.0 * DAMPING_RATIO * ARMATURE_5020 * NATURAL_FREQ
# = 2 * 2 * ARMATURE_5020 * 62.83 = 251 * ARMATURE_5020
```

**Why 10 Hz natural frequency?**
At 10 Hz (≈ 63 rad/s), the joint responds quickly enough to follow policy commands (which
arrive at 50 Hz) while being slow enough that the 5 ms physics timestep resolves the
dynamics without numerical instability. A higher frequency would require more sub-steps.

**Why 2× critical damping?**
Critical damping (`ζ=1`) gives the fastest rise time without overshoot. `ζ=2` (overdamped)
makes the joint slower but more stable. For standing/transition tasks, overshoot is dangerous
(can cause oscillation or joint limit violation), so extra damping is preferred.

---

## 4.5 The Action Scale Formula

This is the most critical derived quantity in the robot config.

```python
G1_ACTION_SCALE: dict[str, float] = {}
for a in G1_ARTICULATION.actuators:
    e = a.effort_limit     # Maximum joint torque [N·m]
    s = a.stiffness        # PD spring constant [N·m/rad]
    for n in a.target_names_expr:
        G1_ACTION_SCALE[n] = 0.25 * e / s
```

### Physical Interpretation

The ratio `e / s` has units of `[N·m] / [N·m/rad] = rad`. It represents the joint deflection
you would get if you applied the maximum torque against a spring with stiffness `s`. More
concretely, it is the **maximum sensible deviation** from the default position given the
motor's capabilities.

Multiplying by `0.25` limits the policy to using at most 25% of that maximum deviation,
which:
1. Prevents the policy from commanding positions far outside the robot's physical reach.
2. Keeps joint trajectories smooth (large commands require many steps to execute through the
   PD controller, naturally smoothing them).
3. Prevents aggressive joint limit violations during early training.

### Worked Example: Knee Joint (7520-22 Motor)

```
ARMATURE_7520_22 ≈ ... (computed from planetary gear formula)
STIFFNESS_7520_22 = ARMATURE_7520_22 * (62.83)²
effort_limit = 139 N·m
action_scale = 0.25 * 139 / STIFFNESS_7520_22
```

The result is approximately 0.3-0.5 rad per unit of network output. A network output of +1.0
moves the knee about 0.3-0.5 rad beyond its default position.

### Why Per-Joint Scales?

Different joints have very different effort limits and stiffnesses. The 7520-22 (knee/hip
roll, 139 N·m) and the 4010 (wrist, 5 N·m) would need wildly different scales if treated
equally. Per-joint scaling ensures that a network output of ±1.0 represents roughly the same
"difficulty" across all joints.

---

## 4.6 `BuiltinPositionActuatorCfg`

```python
G1_ACTUATOR_7520_22 = BuiltinPositionActuatorCfg(
    target_names_expr=(".*_hip_roll_joint", ".*_knee_joint"),
    stiffness=STIFFNESS_7520_22,
    damping=DAMPING_7520_22,
    effort_limit=ACTUATOR_7520_22.effort_limit,
    armature=ACTUATOR_7520_22.reflected_inertia,
)
```

- `target_names_expr`: Regex patterns matched against joint names in the MJCF. The `.*`
  wildcard makes them match both left and right joints.
- `stiffness` / `damping`: PD gains. These are applied in MuJoCo's built-in position
  actuator model.
- `effort_limit`: Clamps the torque output. Even if the PD formula computes a higher torque,
  it is clamped to this value.
- `armature`: The reflected rotor inertia added to the joint. This makes the dynamics match
  the real hardware's response to torques.

---

## 4.7 Soft Joint Limit Factor

```python
G1_ARTICULATION = EntityArticulationInfoCfg(
    actuators=(...),
    soft_joint_pos_limit_factor=0.9,
)
```

The `soft_joint_pos_limit_factor=0.9` means that reward penalties for joint limit violations
activate at 90% of the hardware limit, not at 100%. This gives the policy a safety margin.
Without this, the policy would only be penalized after it has already exceeded 90% of the
limit — by then, the joint velocity might be too high to stop before the hard limit.

---

## 4.8 Collision Configuration

```python
FULL_COLLISION = CollisionCfg(
    geom_names_expr=(".*_collision",),
    condim={
        r"^(left|right)_foot[1-7]_collision$": 3,  # Feet: friction model
        ".*_collision": 1,                           # Other body: no friction
    },
    priority={r"^(left|right)_foot[1-7]_collision$": 1},
    friction={r"^(left|right)_foot[1-7]_collision$": (0.6,)},
)
```

**`condim`** controls the MuJoCo contact dimension:
- `condim=1`: Normal force only (frictionless). Used for self-collisions where you do not
  want friction to cause the limbs to catch on each other unrealistically.
- `condim=3`: Normal + 2D friction (standard friction model). Used for feet touching ground.

**`priority`**: When two contact candidates have different priorities, the higher-priority
geom's properties win. Setting priority=1 for feet ensures foot-ground contacts always use
the foot's friction model.

**`friction=(0.6,)`**: The friction coefficient for foot-ground contacts. This represents
a typical indoor floor with light wear. Domain randomization (`foot_friction` event) will
vary this between 0.3 and 1.6 during training, so the policy learns to be robust to
different surfaces.

---

## 4.9 The `get_g1_robot_cfg()` Factory

```python
def get_g1_robot_cfg() -> EntityCfg:
    return EntityCfg(
        init_state=HOME_KEYFRAME,
        collisions=(FULL_COLLISION,),
        spec_fn=get_spec,        # Function that loads and returns the MjSpec
        articulation=G1_ARTICULATION,
    )
```

This function returns a **fresh instance** each time. This is important: if you cached the
return value and used it in two different task configs, modifying one would corrupt the other.
Always call `get_g1_robot_cfg()`, never reference a cached result.

`spec_fn=get_spec` is a lazy loader. The MJCF is only parsed when the environment is
materialized, not when the config object is created. This keeps startup fast.

---

## 4.10 The Waist and Ankle Actuators — 4-Bar Linkage

```python
G1_ACTUATOR_WAIST = BuiltinPositionActuatorCfg(
    target_names_expr=("waist_pitch_joint", "waist_roll_joint"),
    stiffness=STIFFNESS_5020 * 2,   # Two motors in parallel
    damping=DAMPING_5020 * 2,
    effort_limit=ACTUATOR_5020.effort_limit * 2,
    armature=ACTUATOR_5020.reflected_inertia * 2,
)
```

The waist pitch/roll and ankle joints are driven by **two 5020 motors acting through a
4-bar linkage** (a parallel actuator mechanism). Since the exact geometry is unknown, the
code approximates the effective stiffness, damping, effort, and armature as `2×` the single
motor's values. This is correct for an ideal 1:1 parallel linkage.

The consequence: if you tune the action scale for these joints by changing the effort limit
or stiffness, you must change both factors together, or the 2× relationship will be wrong.
