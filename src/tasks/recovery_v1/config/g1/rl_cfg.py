"""RL configuration for the Unitree G1 recovery-v1 task.

Same network architecture as the transition series.  Adjustments:
  num_steps_per_env: scaled proportionally with the longer 35 s episode
    (v3 uses 40 steps for 25 s → 56 steps for 35 s).
  max_iterations: increased to 20 001 — floor recovery is a harder task
    than squat-to-stand and needs more training time.
  init_std: kept at 1.0 — the task is harder but the same exploration
    breadth is appropriate; the extra iterations handle the difficulty.
  entropy_coef: increased to 0.03 (was 0.01) — floor recovery from flat
    requires more exploration diversity.  In the failed run, std collapsed
    from 0.9 to 0.35 by step 400, leaving flat episodes with insufficient
    action variety to discover the get-up motion.  Higher entropy resists
    this premature convergence.
"""

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)


def unitree_g1_recovery_v1_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create RL runner configuration for the G1 recovery-v1 task."""
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.03,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="g1_recovery_v1",
    save_interval=100,
    num_steps_per_env=56,   # Scaled with 35 s episode (35/25 * 40 ≈ 56)
    max_iterations=20001,
  )
