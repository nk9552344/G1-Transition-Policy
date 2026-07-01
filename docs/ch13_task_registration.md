# Chapter 13: Task Registration — How Everything Gets Wired Together

Task registration is the mechanism that connects all the pieces: the env config, the RL
config, the robot config, and the runner. Understanding it is essential for debugging
"why is my task not showing up" or "why is it loading the wrong config."

---

## 13.1 `register_mjlab_task`

```python
# src/tasks/transition/config/g1/__init__.py

from mjlab.tasks.registry import register_mjlab_task
from src.tasks.transition.rl import TransitionOnPolicyRunner
from .env_cfgs import unitree_g1_transition_env_cfg
from .rl_cfg import unitree_g1_transition_ppo_runner_cfg

register_mjlab_task(
    task_id="Unitree-G1-Transition",
    env_cfg=unitree_g1_transition_env_cfg(),         # Training config (num_envs=1 default)
    play_env_cfg=unitree_g1_transition_env_cfg(play=True),  # Eval config
    rl_cfg=unitree_g1_transition_ppo_runner_cfg(),   # PPO config
    runner_cls=TransitionOnPolicyRunner,             # Custom runner
)
```

**What `register_mjlab_task` does:**
Stores a record in a global dict keyed by `task_id`. The record contains:
- `env_cfg`: used by `scripts/train.py` for training
- `play_env_cfg`: used by `scripts/play.py` for evaluation
- `rl_cfg`: used by both train and play
- `runner_cls`: used by train to instantiate the runner

**When registration happens:**
The `__init__.py` file runs at import time. Registration happens when `src.tasks` is imported
(which triggers `import_packages`, which recursively imports all task packages).

**In `scripts/train.py`:**
```python
import mjlab.tasks  # Registers mjlab's built-in tasks
import src.tasks    # Registers our custom tasks (triggers __init__.py in all config dirs)

# Now the registry has all tasks. Query it:
env_cfg = load_env_cfg("Unitree-G1-Transition")  # Returns the registered env_cfg
```

---

## 13.2 `load_env_cfg`, `load_rl_cfg`, `load_runner_cls`

```python
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

env_cfg = load_env_cfg("Unitree-G1-Transition")           # Returns ManagerBasedRlEnvCfg
play_env_cfg = load_env_cfg("Unitree-G1-Transition", play=True)  # Returns play config
rl_cfg = load_rl_cfg("Unitree-G1-Transition")             # Returns RslRlOnPolicyRunnerCfg
runner_cls = load_runner_cls("Unitree-G1-Transition")     # Returns TransitionOnPolicyRunner
```

These are simple lookups into the global registry dict. If the task is not registered,
they raise a `KeyError` or return `None` (depending on the function).

---

## 13.3 The Task ID Naming Convention

Task IDs follow the pattern:
```
Unitree-{RobotName}-{TaskType}
```

Examples:
- `Unitree-G1-Transition` — G1 full body, transition task
- `Unitree-G1-Rough` — G1 full body, velocity tracking on rough terrain
- `Unitree-G1-Flat` — G1 full body, velocity tracking on flat terrain
- `Unitree-G1-Tracking` — G1 full body, motion tracking
- `Unitree-G1-23Dof-Rough` — G1 23-DOF variant, rough terrain

When you create a new task, follow this convention. The ID is a string — there is no
automated validation. Typos in the task ID will silently create a new unreachable registry
entry.

---

## 13.4 The `play=False` Config Split

The registration takes two configs: one for training, one for play. The differences:

| Setting | Training | Play |
|---------|----------|------|
| `episode_length_s` | 15.0 | 1e9 (infinite) |
| `enable_corruption` | True | False |
| `push_robot` event | Present | Removed |
| `num_envs` | 1 (CLI override) | 1 (usually) |
| Terrain randomization | Not present | Added |

The play config is returned when you call `load_env_cfg(task_id, play=True)`, which
`scripts/play.py` does automatically.

---

## 13.5 How `import_packages` Discovers Tasks

```python
# src/tasks/__init__.py
from mjlab.utils.lab_api.tasks.importer import import_packages
_BLACKLIST_PKGS = ["utils", ".mdp"]
import_packages(__name__, _BLACKLIST_PKGS)
```

`import_packages` performs a recursive import of all sub-packages of `src.tasks`,
**excluding** packages whose names contain any blacklist string.

The blacklist `["utils", ".mdp"]` excludes:
- `src.tasks.velocity.mdp` — this is a library module, not a task config
- `src.tasks.transition.mdp` — same
- Any package named `utils` — utility packages do not contain task registrations

**Packages that ARE imported (and thus register their tasks):**
- `src.tasks.velocity.config.g1` → registers `Unitree-G1-Flat`, `Unitree-G1-Rough`
- `src.tasks.transition.config.g1` → registers `Unitree-G1-Transition`
- All other `config/{robot}/__init__.py` files

**Warning:** If your new task's `__init__.py` is in a directory named `utils` or `mdp`,
it will be blacklisted and never discovered. Keep config directories under `config/{robot}/`.

---

## 13.6 Registering Multiple Configs for One Robot

It is common to register multiple variants (flat vs rough, different DOF counts):

```python
# src/tasks/velocity/config/g1/__init__.py
register_mjlab_task(
    task_id="Unitree-G1-Rough",
    env_cfg=unitree_g1_rough_env_cfg(),
    play_env_cfg=unitree_g1_rough_env_cfg(play=True),
    rl_cfg=unitree_g1_ppo_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)
register_mjlab_task(
    task_id="Unitree-G1-Flat",
    env_cfg=unitree_g1_flat_env_cfg(),
    play_env_cfg=unitree_g1_flat_env_cfg(play=True),
    rl_cfg=unitree_g1_ppo_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)
```

Both calls happen in the same `__init__.py`. The RL config is shared — same PPO settings
for both terrain types.

---

## 13.7 Task Listing

```bash
uv run python scripts/list_envs.py
```

This script imports all tasks (triggering discovery) then prints all registered task IDs.
Use it to verify your new task is being discovered:
- If it appears: the `__init__.py` ran successfully and `register_mjlab_task` was called.
- If it does not appear: check the import chain (directory name, blacklist, typos).
