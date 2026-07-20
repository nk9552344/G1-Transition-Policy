"""RL configuration for the Unitree G1 recovery-v1 task.

Key changes from previous failed training:

  lam: 0.95 → 0.97
    This is the single most impactful fix.  With λ=0.95 and γ=0.99, the GAE
    effective horizon = 1/(1-γλ) ≈ 17 steps = 0.34 seconds.  Floor recovery
    takes 5–15 seconds: rolling from supine (1 s), push-up (1 s), sit-to-stand
    (2 s), stabilisation (1 s).  A policy at step 0 that initiates a roll can
    only see rewards within 0.34 s.  The roll pays off at t=1 s — beyond the
    GAE horizon — so it looks unprofitable.  With λ=0.97 the effective horizon
    doubles to 0.67 s, making 1-second recovery actions visible and profitable.

    Numerical verification (roll taking 50 steps / 1 s):
      λ=0.95: discounted orientation gain = 0.18 × Σ(0.9405)^t ≈ +2.89
              discounted ang_vel penalty  = 0.072 × Σ(0.9405)^t ≈ −1.16
              net advantage = +1.73  (positive but noisy; std collapse still wins)
      λ=0.97: discounted orientation gain = 0.18 × Σ(0.9603)^t ≈ +3.87
              discounted ang_vel penalty  = 0.072 × Σ(0.9603)^t ≈ −1.55
              net advantage = +2.32  (stronger signal → resists std collapse)
      (ang_vel penalty is also height-gated to zero during floor phase, making
       the advantage even larger in practice.)

  num_steps_per_env: 56 → 80
    Longer rollouts match the longer GAE horizon: with λ=0.97 you need to
    capture at least 1/(1-0.97) ≈ 33 steps of future rewards per update.
    80 steps = 1.6 s of experience per env per update, fully spanning the
    0.67 s effective horizon.  At 56 steps some of the bootstrapped value
    is outside the rollout, adding noise to the GAE calculation.

  entropy_coef: 0.03 → 0.05
    Std collapsed from 0.9 → 0.35 by step 400 in the previous run, leaving
    fallen episodes with insufficient action variety to discover get-up motions.
    Higher entropy coefficient maintains a larger action distribution and
    slows the premature convergence to the bouncing local optimum.

  init_std: kept at 1.0
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
      entropy_coef=0.01,       # reduced from 0.04; 44 % upright starts no longer need high entropy to prevent std collapse
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
