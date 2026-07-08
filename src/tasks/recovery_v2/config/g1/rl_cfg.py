"""RL configuration for the Unitree G1 recovery-v2 task.

Iteration analysis
──────────────────
The task progression and empirical scaling:

  Task           ep_len   steps/env   templates   fallen %   iters
  ─────────────────────────────────────────────────────────────────
  transition-v3    25 s      40          4 bent        0 %   15 001
  recovery-v1      35 s      56     2F + 4B = 6       33 %   20 001
  recovery-v2      45 s      72     4F + 4B = 8       50 %   28 001

Derivation for recovery-v2
  Base: 20 001 iterations from recovery-v1.

  +4 000 for side-lying complexity
    Side poses introduce a new motor primitive (lateral rolling → prone/supine →
    push-up) that the policy must discover from scratch.  This is qualitatively
    different from adding more squat depths and requires dedicated exploration.

  +3 000 for template diversity and horizon
    8 templates vs 6 (+33 %); 45 s vs 35 s (+29 %) → roughly 40 % more
    behavioural contexts per iteration to master.

  +1 000 engineering margin
    Nets to 28 001.  Rounds to a clean 280 checkpoints at save_interval=100.

  Total data per env: 28 001 × 72 ≈ 2.02 M steps/env
    vs recovery-v1:   20 001 × 56 ≈ 1.12 M steps/env (+80 %)
    This is appropriate given the additional complexity.

num_steps_per_env
  Scaled proportionally with episode length: 40 × (45 / 25) ≈ 72 steps.
  (Recovery-v1 used 40 × (35 / 25) = 56 steps.)

entropy_coef: 0.03 → 0.04
  Side-lying requires discovering a lateral rolling primitive not present in
  v1.  Higher entropy resists premature convergence during the exploration
  phase.  0.04 is a moderate increase — high enough to prevent std collapse
  for the new motion, low enough not to destabilise the already-learned
  standing skill.

init_std: unchanged at 1.0
  The same broad exploration breadth appropriate; the extra iterations and
  entropy handle the increased difficulty.
"""

from mjlab.rl import (
    RslRlModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)


def unitree_g1_recovery_v2_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    """Create RL runner configuration for the G1 recovery-v2 task."""
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
            entropy_coef=0.04,       # 0.03 in v1; increased for lateral-roll exploration
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1.0e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
        experiment_name="g1_recovery_v2",
        save_interval=100,
        num_steps_per_env=72,        # 40 × (45 / 25): scaled with 45 s episode
        max_iterations=28001,        # see derivation in module docstring
    )
