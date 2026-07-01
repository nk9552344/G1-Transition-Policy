# Chapter 5: ManagerBasedRlEnvCfg — Every Parameter Explained

The environment configuration is the central artifact of this codebase. Understanding every
field in `ManagerBasedRlEnvCfg` lets you know exactly what to change when you want to modify
episode length, the number of training environments, physics fidelity, or the rendering camera.

---

## 5.1 The Top-Level Config

```python
ManagerBasedRlEnvCfg(
    scene=SceneCfg(...),
    observations={...},
    actions={...},
    commands={},           # Empty for transition policy (no velocity commands)
    events={...},
    rewards={...},
    terminations={...},
    curriculum={},         # Empty for transition policy (no terrain curriculum)
    metrics={...},
    viewer=ViewerConfig(...),
    sim=SimulationCfg(...),
    decimation=4,
    episode_length_s=15.0,
)
```

We will go through each field.

---

## 5.2 `scene: SceneCfg`

```python
scene=SceneCfg(
    terrain=TerrainEntityCfg(terrain_type="plane"),
    num_envs=1,       # Overridden to 4096+ at train time via CLI
    extent=2.0,       # 2m × 2m tile per environment
)
```

### `terrain_type="plane"`

A flat infinite ground plane at z=0. The G1 is spawned with its CoM at z=0.8 m, which
gives approximately 0.1 m ground clearance in the default standing pose.

Alternative: `terrain_type="generator"` creates procedurally generated heightfield terrain.
You would use this for walking policies that need to generalize to rough terrain. For
standing/transition policies, the plane is correct.

### `num_envs=1`

The default is 1 (for visualization/debugging). During training you always override this:
```bash
python scripts/train.py Unitree-G1-Transition --env.scene.num-envs 4096
```

More environments = more data per iteration = more stable gradient estimates. 4096 is a
reasonable default for GPUs with 40+ GB VRAM. On smaller GPUs, use 1024 or 2048.

### `extent=2.0`

Each environment is placed on a grid with 2m spacing. With 4096 environments, the grid is
64×64, covering 128m×128m. Environments that are too close together would have their robots
colliding with each other — the extent prevents this.

### `entities` and `sensors` (set by the G1 override)

```python
cfg.scene.entities = {"robot": get_g1_robot_cfg()}
cfg.scene.sensors = (feet_ground_cfg, self_collision_cfg)
```

`entities` is a dict mapping string names to `EntityCfg` objects. The string name (`"robot"`)
is the key used throughout the rest of the config to refer to this entity — in
`SceneEntityCfg("robot")`, in `JointPositionActionCfg(entity_name="robot")`, etc.

---

## 5.3 `sim: SimulationCfg`

```python
sim=SimulationCfg(
    nconmax=None,   # Let the framework auto-size (set to 64 in G1 override via contact_sensor_maxmatch)
    njmax=300,      # Maximum contact constraints in the solver
    mujoco=MujocoCfg(
        timestep=0.005,      # 5 ms = 200 Hz physics
        iterations=10,       # Constraint solver iterations per step
        ls_iterations=20,    # Line search iterations in solver
        ccd_iterations=50,   # Continuous collision detection iterations
    ),
)
```

### How to choose `njmax`

`njmax` limits how many contact constraints the MuJoCo solver processes simultaneously.
Too low → contacts are dropped silently → the robot can pass through the ground.
Too high → wastes memory and compute.

For the G1 standing/transition task:
- The G1 has 7 foot geoms per foot × 2 feet = 14 maximum foot-ground contacts.
- Each contact generates up to 4 constraints in `condim=3` mode (1 normal + 3 friction).
- Self-collision adds some more.
- `njmax=300` gives ample headroom.

For walking on rough terrain, you may need `njmax=1500` because each foot can contact
multiple terrain features simultaneously.

### How to choose `timestep`

The physics timestep must be small enough that the PD control forces don't cause numerical
instability. The rule of thumb:

```
timestep < 1 / (10 × max_natural_frequency_of_any_joint)
```

For G1 joints at 10 Hz natural frequency: `timestep < 1/(10×62.83) ≈ 1.6 ms`. Our 5 ms
timestep is actually above this, but in practice the solver's stability region is wider than
this simplified analysis suggests. You can verify stability by checking that joint positions
don't diverge early in training.

### `decimation=4`

The policy acts every `decimation × timestep = 4 × 0.005 = 0.02 s = 50 Hz`. The PD
controller runs at 200 Hz between policy steps. This means:
- The policy output (target joint position) is held constant for 4 physics steps.
- The PD controller drives toward this target at 200 Hz.
- High-frequency oscillations that would destabilize the 50 Hz policy are absorbed by the
  200 Hz PD controller.

**If you increase `decimation`:** The policy acts less frequently, making fast motions
harder to learn. At `decimation=8` (25 Hz), walking gaits with 10 Hz leg frequency become
very difficult to control.

**If you decrease `decimation` to 1:** The policy must directly control joint forces at
200 Hz, which is hard to learn and does not match real deployment (which typically runs at
50-100 Hz).

---

## 5.4 `episode_length_s=15.0`

Each episode runs for at most 15 seconds. At 50 Hz (policy rate), this is 750 steps.

**How to choose episode length:**
- Long enough that the policy can learn the complete behavior from start to finish.
  For the transition task: the robot needs ~3-5 seconds to recover from a 0.5 rad offset,
  then must hold for the remainder. 15 seconds is comfortable.
- Not so long that failed policies (robot falls) waste time. With `is_terminated=-200` and
  a fall criterion at 70°, fallen robots terminate quickly.
- Shorter episodes → more resets per hour of training → more exposure to the random initial
  conditions. But if the task requires many seconds to solve, short episodes force the policy
  to be successful from the first step, which is a harder curriculum.

The velocity task uses 20 s because walking requires sustained behavior and longer horizons
for the GAE advantage estimates to be accurate.

---

## 5.5 The `commands={}` Choice

The transition policy has `commands={}` — an empty dict. This means:
- There is no command signal in the observations.
- The policy's goal is implicit: always drive to the default joint positions.
- You do not need a `command_name` parameter in any reward function.

If you were writing a velocity-tracking policy, `commands` would contain:
```python
commands={"twist": UniformVelocityCommandCfg(...)}
```
And the actor observations would include:
```python
"command": ObservationTermCfg(func=mdp.generated_commands, params={"command_name": "twist"})
```

For a transition policy that targets a specific non-default pose, you could add a command
that specifies the target pose, or hardcode a different `default_joint_pos` in the keyframe.

---

## 5.6 `curriculum={}`

No curriculum is used in the transition task. Curriculum learning gradually increases task
difficulty (e.g., terrain roughness, command speed) as the policy improves.

For the transition task:
- The initial offset range `JOINT_OFFSET_RANGE=0.5 rad` is fixed throughout training.
- A curriculum could increase this range from 0.1 → 0.5 rad as training progresses.
- This is not implemented but would be a natural extension for harder policies.

---

## 5.7 How `num_envs=1` Becomes 4096

The registered config has `num_envs=1`. This default exists so that visualization
(`scripts/play.py`) starts with a single environment by default.

At training time, you override via CLI:
```bash
python scripts/train.py Unitree-G1-Transition --env.scene.num-envs 4096
```

The `tyro` CLI framework parses `--env.scene.num-envs 4096` and sets
`cfg.scene.num_envs = 4096` before the environment is created. The environment never sees
the `num_envs=1` default.

---

## 5.8 The `contact_sensor_maxmatch=64` Setting (G1 Override)

```python
cfg.sim.contact_sensor_maxmatch = 64
```

This controls how many contact pairs the sensor subsystem tracks per frame. For the
transition task with flat terrain and two feet, 64 is more than enough. For rough terrain
tasks, this needs to be higher (500 in the velocity task's G1 config).

Setting this too low means some contacts will be dropped from sensor readings, making
`foot_contact` potentially unreliable. Setting it too high wastes memory.

---

## 5.9 The Play Mode Override

```python
if play:
    cfg.episode_length_s = int(1e9)            # Effectively infinite
    cfg.observations["actor"].enable_corruption = False  # No noise in playback
    cfg.events.pop("push_robot", None)          # No disturbances
    cfg.events["randomize_terrain"] = EventTermCfg(
        func=envs_mdp.randomize_terrain,
        mode="reset",
        params={},
    )
```

Play mode is for evaluation and visualization:
- **Infinite episode length**: You can watch the robot without it timing out.
- **No observation noise**: Lets you see the policy's true behavior, not noise-injected behavior.
- **No push events**: Removes disturbances so you can evaluate the clean policy.
- **Terrain randomization**: In play mode with multiple environments, each episode samples a
  new terrain tile to show diversity. (For the transition task on flat terrain, this has no
  visible effect since the terrain is always flat.)
