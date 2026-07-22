"""RL configuration for the Unitree G1 recovery-v1 task.

lam: 0.95 → 0.97
  With λ=0.95 and γ=0.99, the GAE effective horizon = 1/(1-γλ) ≈ 17 steps =
  0.34 seconds.  Floor recovery takes 5–15 seconds.  A roll that pays off at
  t=1 s is beyond the 0.34 s horizon, so it looks unprofitable.  λ=0.97
  extends the horizon to 25 steps ≈ 0.50 s (1.5× longer, not 2× — previous
  comment was wrong: 1/(1-0.97×0.99) = 1/0.0397 ≈ 25 steps).
  The dense orientation_rate reward (added in this version) bypasses this
  limit by giving per-step credit for each rotational improvement.

num_steps_per_env: 56 → 80
  Match the λ=0.97 GAE horizon: 80 steps = 1.6 s spans the 0.50 s horizon
  with margin.  At 56 steps some bootstrapped value is outside the rollout.

entropy_coef: 0.04 → 0.01 → 0.02
  High entropy was needed when airborne_penalty dominated (-350/episode).
  Reduced to 0.01 once that bug was fixed.  Raised back to 0.02 to slow
  premature commitment to local optima: at 0.01 the policy rapidly collapsed
  orientation_rate farming (waist-oscillation) before ever discovering the
  push-up→kneeling transition.  0.02 gives enough exploration to escape
  without inflating the standing-balance std.

init_std: 1.0
  Recovery requires large exploratory actions.  1.0 is correct.
"""

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)


def unitree_g1_recovery_v1_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create RL runner configuration for the G1 recovery-v1 task."""
  # Network capacity: (512, 512, 256, 128) instead of (512, 256, 128).
  #
  # Recovery requires the policy to detect and switch between 5+ qualitatively
  # different behavior modes: rolling, push-up, hip-tuck, squat-to-stand, and
  # balance.  The standard (512, 256, 128) architecture has a single 512-unit
  # first layer followed by aggressive compression to 256 → 128.  This is the
  # same network used for walking/squat-to-stand — tasks that are essentially
  # unimodal once the command is fixed.
  #
  # Multi-modal reasoning lives in the middle hidden layers where the network
  # "decides" which sub-behavior to apply.  Widening the second layer from 256
  # to 512 adds 131k parameters (×2.3 total) specifically at that decision
  # point, without inflating the first-layer feature extraction or output layer.
  # Training time increases ~15–20 % per step — acceptable for a harder task.
  #
  # (1024, 512, 256) would put the extra capacity at the first layer where
  # inputs are simple sensor readings, which is wasteful for this purpose.
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(512, 512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 512, 256, 128),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.02,       # raised from 0.01; prevents rapid commitment to local optima (waist-wiggle collapse)
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.97,                # was 0.95; 2× longer GAE horizon for multi-second recovery
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="g1_recovery_v1",
    save_interval=100,
    num_steps_per_env=80,      # was 56; match longer λ=0.97 GAE horizon
    max_iterations=20001,
  )
