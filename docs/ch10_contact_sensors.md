# Chapter 10: Contact Sensors — Configuration and Usage

Contact sensors are how the policy learns about which parts of the robot are touching what.
They are essential for foot-contact rewards, self-collision penalties, and for giving the
critic privileged ground-truth contact information.

---

## 10.1 `ContactSensorCfg` — The Declaration

```python
from mjlab.sensor import ContactSensorCfg, ContactMatch

feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",            # Key for env.scene[name]
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

---

## 10.2 `ContactMatch` — Selecting Which Bodies to Watch

`ContactMatch` specifies which **MuJoCo bodies** to consider as one side of a contact pair.

```python
ContactMatch(
    mode: str,     # "body", "subtree", or "geom"
    pattern: str,  # Regex or name to match bodies/geoms
    entity: str,   # Which scene entity to search in ("robot", "terrain")
)
```

### Mode Options

**`mode="body"`:**
Matches a single named body exactly (or by regex).
```python
ContactMatch(mode="body", pattern="terrain")  # Matches the terrain body
```

**`mode="subtree"`:**
Matches a body **and all of its child bodies** in the kinematic tree. This is the most
common mode for foot contact detection.
```python
ContactMatch(
    mode="subtree",
    pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$",
    entity="robot",
)
```
This matches `left_ankle_roll_link`, `right_ankle_roll_link`, and all bodies in their
subtrees (foot links, toe links, etc.). Any contact between any of these bodies and the
`secondary` match counts as foot contact.

**`mode="geom"`:**
Matches specific collision geometries by name. More precise than `"body"` but requires
knowing exact geom names.

### The Regex Pattern

The `pattern` field is a Python regex matched against body (or geom) names in the MJCF.
- `r"^(left_ankle_roll_link|right_ankle_roll_link)$"` — exactly these two bodies
- `"pelvis"` — any body containing the string "pelvis"
- `".*_foot[1-7]_collision"` — all 14 foot collision geoms

---

## 10.3 `fields` — What Data to Collect

```python
fields=("found", "force")
```

| Field | Shape | Description |
|-------|-------|-------------|
| `"found"` | `[B, N_slots]` | Integer count of contacts found (0 or positive integer) |
| `"force"` | `[B, N_slots, 3]` | Net contact force vector per slot (N) |
| `"force_history"` | `[B, N_slots, H, 3]` | Contact force over history_length substeps |
| `"torque"` | `[B, N_slots, 3]` | Net contact torque per slot |

**Not every field is always available.** If you request `"force"` in `fields`, then
`sensor.data.force` will be populated. If you do not request it, the field is `None`.
Accessing `sensor.data.force` when force was not requested causes an `AttributeError` at
runtime (not at config time).

---

## 10.4 `reduce` — How Multiple Contacts Become One Value

When a subtree matches multiple bodies that each have multiple contact points, the sensor
needs to aggregate all those contact forces into something useful.

```python
reduce="netforce"  # Sum all force vectors → one net force per slot
reduce="none"      # Keep all individual contact forces (up to num_slots)
```

**`reduce="netforce"`** (used for foot-ground contact):
All contact forces between the primary and secondary bodies are summed into a single net
force vector. This gives you the total ground reaction force under each foot. Appropriate
when you want to know "how hard is the foot pushing on the ground" rather than the
distribution across specific contact points.

**`reduce="none"`** (used for self-collision):
Individual contact events are kept. Combined with `history_length`, this allows counting
how many contacts occurred during a window. Appropriate when you want "did any part of
the subtree touch itself" rather than the force magnitude.

---

## 10.5 `num_slots` — Capacity

```python
num_slots=1  # One aggregate value per foot
```

`num_slots` defines how many independent contact entries the sensor tracks. For
`reduce="netforce"` and two feet, `num_slots=1` means each primary match (left foot,
right foot) gets one slot. The total sensor data shape is `[B, 2, ...]` (two feet, one
slot each becomes effectively `[B, 2]` for `found` after the primary pattern matches two
subtrees).

**More precisely:** The sensor creates `num_primary_matches × num_slots` tracking slots.
For the foot-ground sensor where the primary pattern matches 2 subtrees (left and right
ankle), `num_slots=1` means 2 total slots = 2 readings per environment.

For `reduce="none"` with `history_length=4`, the shape would be `[B, 1, 4, 3]` — one slot,
4 history steps, 3D force.

---

## 10.6 `track_air_time=True`

When `True`, the sensor tracks how long each foot has been in the air (not in contact) and
how long it has been in contact. These are accessed via:
- `sensor.data.current_air_time` — `[B, N_feet]` — seconds since foot left the ground
- `sensor.data.current_contact_time` — `[B, N_feet]` — seconds since foot hit the ground

This is only needed for gait-related rewards (`feet_air_time`). For the transition task,
it is set to `True` even though we do not use it, because the underlying sensor is shared
with the `both_feet_contact` reward (which only uses `found`). Setting it to `False` when
not needed saves a small amount of memory.

---

## 10.7 The Ground-Contact Sensor — Detailed Breakdown

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

**Primary:** Two subtrees — left and right ankle roll link and all their children (foot
links, collision geoms, etc.)
**Secondary:** The terrain body — the ground plane.

**What it detects:** Any contact between the foot subtree and the terrain.

**`sensor.data.found`** shape: `[B, 2]`
- `found[b, 0] > 0` → left foot is in contact with terrain
- `found[b, 1] > 0` → right foot is in contact with terrain

**`sensor.data.force`** shape: `[B, 2, 3]`
- `force[b, 0, :]` → net ground reaction force under left foot
- `force[b, 1, :]` → net ground reaction force under right foot
- Magnitude is typically 0-400 N (the robot weighs ~60 kg, so 600 N total, distributed
  between two feet; in single stance, one foot gets ~600 N)

---

## 10.8 The Self-Collision Sensor — Detailed Breakdown

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

**Primary = Secondary = pelvis subtree:** This creates a sensor that detects when the robot's
body touches itself. Any contact between two bodies that are both children of the pelvis counts.

**`reduce="none"` + `history_length=4`:**
The force_history tensor is `[B, 1, 4, 3]`:
- 1 slot (as configured)
- 4 history steps (last 4 physics substeps = last 20 ms)
- 3D force vector per step

**Usage in `self_collision_cost`:**
```python
force_mag = torch.norm(data.force_history, dim=-1)   # [B, 1, 4]
hit = (force_mag > force_threshold).any(dim=1)        # [B, 4] — any slot in that substep?
return hit.sum(dim=-1).float()                        # [B] — count of substeps with collision
```

Returns a value in `[0, 4]` — how many of the last 4 substeps had a self-collision above
the force threshold.

---

## 10.9 Wiring Sensors to Observations and Rewards

After defining sensors, they must be connected to the terms that use them:

```python
# Wiring in the G1 override:
cfg.observations["critic"].terms["foot_contact"].params["sensor_name"] = "feet_ground_contact"
cfg.observations["critic"].terms["foot_contact_forces"].params["sensor_name"] = "feet_ground_contact"
cfg.rewards["both_feet_contact"].params["sensor_name"] = "feet_ground_contact"
cfg.rewards["self_collisions"].params["sensor_name"] = "self_collision"
```

The base config in `transition_env_cfg.py` uses placeholder sensor names in the reward
configs, and the robot-specific override fills them in. This is why the base config does not
need to know about sensor names — it only declares *that* contact information will be used.

---

## 10.10 Debugging Contact Sensors

**Symptom: `both_feet_contact` is always 0:**
- Check that `sensor_name` in the reward params matches the `name` in `ContactSensorCfg`.
- Check that `cfg.scene.sensors` includes the sensor config object.
- Check that `primary.pattern` matches actual body names in the MJCF. You can print
  `env.scene.articulations["robot"].body_names` to see all body names.

**Symptom: `sensor.data.found` is None at runtime:**
- `"found"` must be in the `fields` tuple in `ContactSensorCfg`.

**Symptom: `sensor.data.force_history` is None:**
- `"force"` must be in `fields` AND `history_length > 1` must be set.

**Symptom: Contact sensor returns unexpected shapes:**
- Verify `num_slots` and `reduce` settings. With `reduce="netforce"` and `num_slots=1`,
  the `found` shape should be `[B, num_primary_matches * num_slots]`.
- Print `sensor.data.found.shape` inside a reward function to verify.
