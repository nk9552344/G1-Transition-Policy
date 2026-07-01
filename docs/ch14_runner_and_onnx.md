# Chapter 14: The Runner and ONNX Export

The runner orchestrates training and handles checkpointing. The custom runner in this
codebase extends the base runner to automatically export a deployment-ready ONNX model
every time a checkpoint is saved.

---

## 14.1 `MjlabOnPolicyRunner` (the base)

The base runner (from `mjlab.rl.runner`) handles:
- Collecting rollouts from the environment
- Computing GAE advantages
- Running PPO updates
- Logging to WandB or local TensorBoard
- Saving checkpoints to the log directory

You rarely need to modify the base runner. The custom `TransitionOnPolicyRunner` only
overrides the `save` method.

---

## 14.2 `TransitionOnPolicyRunner`

```python
# src/tasks/transition/rl/runner.py

class TransitionOnPolicyRunner(MjlabOnPolicyRunner):
    env: RslRlVecEnvWrapper

    def save(self, path: str, infos=None):
        super().save(path, infos)                    # Save the .pt checkpoint normally

        policy_path = path.split("model")[0]          # Extract the run directory
        filename = "policy.onnx"

        self.export_policy_to_onnx(policy_path, filename)  # Export ONNX

        # Determine run name for metadata
        run_name: str = (
            wandb.run.name
            if self.logger.logger_type == "wandb" and wandb.run
            else "local"
        )

        onnx_path = os.path.join(policy_path, filename)
        metadata = get_base_metadata(self.env.unwrapped, run_name)
        attach_metadata_to_onnx(onnx_path, metadata)   # Embed env metadata

        if self.logger.logger_type in ["wandb"]:
            wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))
```

**`export_policy_to_onnx(policy_path, filename)`:**
Converts the actor network (with observation normalization) to ONNX format. ONNX is a
standard format for neural network inference across different frameworks (PyTorch, TensorFlow,
C++ runtimes).

The exported ONNX model takes the actor observation vector as input and produces joint
position action targets as output. This is what runs on the real robot's onboard computer.

**`get_base_metadata(env, run_name)` and `attach_metadata_to_onnx`:**
Embeds metadata into the ONNX file's custom properties:
- Observation dimension and ordering
- Action dimension and ordering
- Joint names and action scales
- Environment configuration summary
- WandB run name (for traceability)

This metadata is read by the deployment code (`deploy/` directory, C++) to verify that the
ONNX model is compatible with the robot's current configuration.

---

## 14.3 Checkpoint File Structure

After training, the log directory contains:

```
logs/rsl_rl/g1_transition/2024-01-15_14-30-00/
├── params/
│   ├── env.yaml      # Full environment config dump
│   └── agent.yaml    # Full PPO config dump
├── model_100.pt      # Checkpoint at iteration 100
├── model_200.pt
├── ...
├── model_10000.pt
├── policy.onnx       # Deployment-ready ONNX (from last save)
└── videos/           # Training videos (if --video flag used)
    └── train/
```

**`.pt` files:** PyTorch checkpoint files containing:
- Actor network weights and observation normalization stats
- Critic network weights (not needed at deployment)
- Optimizer state (for resuming training)
- Current iteration number

**`policy.onnx`:** The deployment model. Always reflects the most recent saved checkpoint.
Only the actor is exported (critic is training-only).

---

## 14.4 Loading a Checkpoint for Play

```python
# In scripts/play.py:
runner = runner_cls(env, asdict(agent_cfg), device=device)
runner.load(
    str(resume_path),
    load_cfg={"actor": True},  # Only load actor, not critic
    strict=True,               # Fail if any weight is missing or mismatched
    map_location=device,       # Load to the specified device
)
policy = runner.get_inference_policy(device=device)
```

**`load_cfg={"actor": True}`:** In play mode, we only need the actor. The critic can be
ignored even if it exists in the checkpoint.

**`strict=True`:** If the model architecture does not match the checkpoint (wrong hidden
dims, wrong obs size), this raises an error. Change to `strict=False` if you want to load
partial weights (e.g., transfer learning from a different task).

**`get_inference_policy(device)`:** Returns a callable that:
1. Applies the stored observation normalization to the input
2. Runs the actor forward pass
3. Returns the mean of the Gaussian distribution (no sampling — deterministic inference)

---

## 14.5 Resuming Training

```python
# scripts/train.py:
if cfg.agent.resume:
    resume_path = get_checkpoint_path(
        log_root_path, cfg.agent.load_run, cfg.agent.load_checkpoint
    )
runner.load(str(resume_path))  # Loads all weights + optimizer state + iteration count
runner.learn(...)               # Continues from the loaded iteration
```

To resume a run from the command line:
```bash
uv run python scripts/train.py Unitree-G1-Transition \
    --agent.resume True \
    --agent.load-run 2024-01-15_14-30-00 \
    --agent.load-checkpoint model_5000.pt \
    --env.scene.num-envs 4096
```

The observation normalization stats are included in the checkpoint, so training continues
with the same statistics (not reset).

---

## 14.6 WandB Integration

The runner supports WandB for experiment tracking:

```bash
# Before training:
wandb login

# During training (add --agent.logger wandb):
uv run python scripts/train.py Unitree-G1-Transition --agent.logger wandb
```

With WandB enabled:
- All reward components are logged per iteration
- Policy/critic losses are logged
- `policy.onnx` is uploaded as a WandB artifact after each save
- The `run_name` from WandB is embedded in the ONNX metadata

Without WandB:
- Metrics are printed to stdout
- `run_name="local"` in ONNX metadata
- Checkpoints saved locally only

---

## 14.7 `RslRlVecEnvWrapper`

The environment is wrapped before being passed to the runner:

```python
env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
```

This wrapper adapts the `ManagerBasedRlEnv` interface to the RSL-RL expected interface:
- Splits the observation dict into `actor_obs` and `critic_obs`
- Clips actions to `[-1, 1]` if `clip_actions=True`
- Provides `env.observation_space`, `env.action_space` as standard gym spaces
- Handles the `extras["log"]` dict for per-step metric logging

The `.unwrapped` attribute accesses the underlying `ManagerBasedRlEnv`.
