## Setup

Create a virtual environment with Python 3.10 using `uv`:

```bash
uv venv .venv --python 3.10
source .venv/bin/activate
```

## Installation

Install the package in editable mode:

```bash
uv pip install -e .
```

## List Environments

```bash
uv run python scripts/list_envs.py
```

Available environments:

**Transition**
- `Unitree-G1-Transition`

**Velocity**
- `Unitree-G1-Rough` / `Unitree-G1-Flat`
- `Unitree-G1-23Dof-Rough` / `Unitree-G1-23Dof-Flat`
- `Unitree-H1_2-Rough` / `Unitree-H1_2-Flat`
- `Unitree-H2-Rough` / `Unitree-H2-Flat`
- `Unitree-R1-Rough` / `Unitree-R1-Flat`
- `Unitree-Go2-Rough` / `Unitree-Go2-Flat`
- `Unitree-A2-Rough` / `Unitree-A2-Flat`
- `Unitree-As2-Rough` / `Unitree-As2-Flat`

**Tracking**
- `Unitree-G1-Tracking` / `Unitree-G1-Tracking-No-State-Estimation`
- `Unitree-G1-23Dof-Tracking` / `Unitree-G1-23Dof-Tracking-No-State-Estimation`

## Train

**CPU (no GPU):**

```bash
uv run python scripts/train.py Unitree-G1-Transition --agent.gpu-ids null
```

**Single GPU (default, GPU 0):**

```bash
uv run python scripts/train.py Unitree-G1-Transition --env.scene.num-envs 4096
```

**Specific GPU:**

```bash
uv run python scripts/train.py Unitree-G1-Transition --agent.gpu-ids 1
```

**Multi-GPU:**

```bash
uv run python scripts/train.py Unitree-G1-Transition --agent.gpu-ids all
```

## Play

Play back a saved `.pt` checkpoint:

```bash
uv run python scripts/play.py Unitree-G1-Transition --checkpoint-file logs/<run_dir>/model_<iter>.pt
```

Use `--viewer native` (requires `$DISPLAY`) or `--viewer viser` (browser-based, no display needed):

```bash
uv run python scripts/play.py Unitree-G1-Transition --checkpoint-file logs/<run_dir>/model_<iter>.pt --viewer viser
```

Disable termination conditions to observe the full motion:

```bash
uv run python scripts/play.py Unitree-G1-Transition --checkpoint-file logs/<run_dir>/model_<iter>.pt --no-terminations
```
