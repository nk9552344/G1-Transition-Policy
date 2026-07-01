# Chapter 2: Project Structure — Every File and Why It Exists

Every file in this project has a deliberate purpose. This chapter walks through the full
directory tree and explains what each file does, what it imports from, and what depends on it.

---

## 2.1 Root Directory

```
transition-policy/
├── docs/           ← This documentation
├── scripts/        ← Entry-point scripts (train, play, utilities)
├── simulate/       ← Deploy-time simulation config (not covered here)
├── src/            ← All source code for policies
├── deploy/         ← C++ deployment code for the real robot
├── setup.py        ← Python package definition
└── README.md       ← Quick-start commands
```

The critical separation is `scripts/` vs `src/`. Scripts are **entry points** — you invoke
them from the command line. `src/` is the **library** — it contains all the configuration,
reward functions, and robot definitions. Scripts import from `src/`, not the reverse.

---

## 2.2 `setup.py`

```python
INSTALL_REQUIRES = [
    "tyro",           # CLI argument parsing from dataclasses
    "mjlab==1.2.0",   # The RL simulation framework
    "mujoco-warp==3.5.0",  # GPU-accelerated MuJoCo (JAX/XLA backend)
    "mujoco==3.5.0",       # CPU MuJoCo (viewer, MJCF parsing)
]
```

This installs the `src` package as `unitree_rl_mjlab`. The version pins are critical —
`mjlab` is an internal framework and its API may change between versions.

---

## 2.3 `src/` — The Library

```
src/
├── __init__.py          ← Exports SRC_PATH (the absolute path to this directory)
├── assets/
│   ├── __init__.py
│   ├── motions/         ← Motion reference files (not used by transition policy)
│   └── robots/
│       ├── __init__.py  ← Re-exports all robot get_*_robot_cfg() and ACTION_SCALE
│       └── unitree_g1/
│           ├── __init__.py
│           ├── g1_constants.py      ← G1 29-DOF full body (the one we use)
│           ├── g1_23dof_constants.py ← G1 23-DOF variant (no wrists)
│           └── xmls/
│               └── g1.xml           ← MuJoCo MJCF model file
└── tasks/
    ├── __init__.py      ← Auto-imports all task packages to populate the registry
    └── transition/
        ├── __init__.py
        ├── transition_env_cfg.py ← Base environment configuration factory
        ├── config/
        │   ├── __init__.py
        │   └── g1/
        │       ├── __init__.py  ← Calls register_mjlab_task (runs at import time)
        │       ├── env_cfgs.py  ← G1-specific env overrides
        │       └── rl_cfg.py    ← PPO hyperparameters
        ├── mdp/
        │   ├── __init__.py      ← Re-exports all MDP components
        │   └── rewards.py       ← Transition-specific reward functions
        └── rl/
            ├── __init__.py
            └── runner.py        ← Custom runner that exports ONNX on each save
```

---

## 2.4 The Role of `__init__.py` Files

In Python, `__init__.py` makes a directory importable as a package. In this codebase, they
serve four distinct roles:

### Role 1: Pure namespace marker (empty or trivial)
```python
# src/assets/robots/unitree_g1/__init__.py
"""Unitree G1 humanoid."""
```
These exist only so Python recognizes the directory as a package.

### Role 2: Public API re-export
```python
# src/assets/robots/__init__.py
from .unitree_g1.g1_constants import G1_ACTION_SCALE as G1_ACTION_SCALE
from .unitree_g1.g1_constants import get_g1_robot_cfg as get_g1_robot_cfg
```
This is the "barrel" pattern. Any code that needs the G1 config can import from
`src.assets.robots` rather than knowing the exact internal path. If you later move
`g1_constants.py`, you only change one place.

### Role 3: Auto-import trigger (task discovery)
```python
# src/tasks/__init__.py
from mjlab.utils.lab_api.tasks.importer import import_packages
import_packages(__name__, ["utils", ".mdp"])
```
`import_packages` recursively imports all sub-packages of `src.tasks`, excluding
packages whose names match the blacklist. This causes every `config/g1/__init__.py`
to run, which calls `register_mjlab_task(...)`. This is how all tasks get registered
into the global registry at startup without manually listing them.

### Role 4: Task registration
```python
# src/tasks/transition/config/g1/__init__.py
from mjlab.tasks.registry import register_mjlab_task
from src.tasks.transition.rl import TransitionOnPolicyRunner
from .env_cfgs import unitree_g1_transition_env_cfg
from .rl_cfg import unitree_g1_transition_ppo_runner_cfg

register_mjlab_task(
    task_id="Unitree-G1-Transition",
    env_cfg=unitree_g1_transition_env_cfg(),
    play_env_cfg=unitree_g1_transition_env_cfg(play=True),
    rl_cfg=unitree_g1_transition_ppo_runner_cfg(),
    runner_cls=TransitionOnPolicyRunner,
)
```
This runs once at import time and adds the task to the global task registry.

---

## 2.5 The Configuration Layering Pattern

The env config is split into two layers, and this is a deliberate architectural choice:

```
Layer 1: Base config (transition_env_cfg.py)
  make_transition_env_cfg() → ManagerBasedRlEnvCfg

  Contains everything that is robot-agnostic:
  - Observation terms (by function reference, sensor names as placeholders)
  - Reward terms (by function reference, body names as placeholders)
  - Actions (scale left as 0.25 placeholder)
  - Events (geom names as empty tuples — placeholders)
  - Terminations
  - Sim settings
  - Episode length
```

```
Layer 2: Robot-specific override (config/g1/env_cfgs.py)
  unitree_g1_transition_env_cfg() → ManagerBasedRlEnvCfg

  Starts by calling make_transition_env_cfg()
  Then fills in all the placeholders:
  - Adds the G1 robot entity
  - Adds contact sensors (with real sensor names)
  - Sets action scale to G1_ACTION_SCALE
  - Sets body names for orientation/ang-vel rewards
  - Sets geom names for foot friction DR
  - Adds self-collision reward (G1-specific)
```

This layering means you can create a new robot variant (e.g., G1 with arms removed) by
writing a new `env_cfgs.py` that calls the same base factory. The base layer never changes.

---

## 2.6 `src/tasks/transition/mdp/__init__.py` — The MDP Namespace

```python
from mjlab.envs.mdp import *           # All built-in MDP functions
from src.tasks.velocity.mdp.observations import foot_contact, foot_contact_forces
from src.tasks.velocity.mdp.rewards import (
    angular_momentum_penalty,
    body_angular_velocity_penalty,
    body_orientation_l2,
    self_collision_cost,
)
from .rewards import *                 # Transition-specific: pose_convergence, etc.
```

This creates a single flat namespace `mdp` that transition task code can use. In
`transition_env_cfg.py`, you write `mdp.pose_convergence` and it resolves to the function
defined in `transition/mdp/rewards.py`. You write `mdp.foot_contact` and it resolves to the
function in `velocity/mdp/observations.py`. The consumer never needs to know the source.

When you add a new reward function for the transition task, you add it to `rewards.py` and
it automatically appears in the `mdp` namespace (because of `from .rewards import *`).

---

## 2.7 `scripts/` — Entry Points

### `scripts/train.py`
- Parses CLI arguments with `tyro` (first arg = task ID, remaining = config overrides)
- Creates `ManagerBasedRlEnv` with the env config
- Wraps it with `RslRlVecEnvWrapper` (provides the RSL-RL interface)
- Instantiates the runner, calls `runner.learn()`
- Handles GPU selection, multi-GPU via `torchrunx`, video recording

### `scripts/play.py`
- Same startup as train, but loads a checkpoint instead of training
- Supports three agent modes: `zero` (all zeros), `random` (random actions), `trained`
- Launches either `NativeMujocoViewer` or `ViserPlayViewer`
- The play env config disables noise, disables push events, enables infinite episodes

### `scripts/list_envs.py`
- Imports all tasks (triggering registration) then prints all registered task IDs

### `scripts/csv_to_npz.py` and `scripts/visualize_terrain.py`
- Utility scripts not related to the transition policy

---

## 2.8 Key Dependency Graph

```
scripts/train.py
  └── mjlab.tasks.registry (load_env_cfg, load_rl_cfg, load_runner_cls)
        └── src.tasks (auto-imported via import_packages)
              └── src.tasks.transition.config.g1
                    ├── env_cfgs.py
                    │     ├── src.tasks.transition.transition_env_cfg (base layer)
                    │     ├── src.assets.robots (G1_ACTION_SCALE, get_g1_robot_cfg)
                    │     └── src.tasks.transition.mdp (reward/obs functions)
                    ├── rl_cfg.py
                    │     └── mjlab.rl (RslRlOnPolicyRunnerCfg, etc.)
                    └── __init__.py
                          └── register_mjlab_task(...)
```

This means: if `register_mjlab_task` is not being called, the issue is in the import chain
somewhere between `scripts/train.py → src.tasks → config/g1/__init__.py`.
