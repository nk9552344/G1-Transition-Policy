"""Unitree G1 recovery-v2 task registration."""

from mjlab.tasks.registry import register_mjlab_task

from src.tasks.transition.rl import TransitionOnPolicyRunner

from .env_cfgs import unitree_g1_recovery_v2_env_cfg
from .rl_cfg import unitree_g1_recovery_v2_ppo_runner_cfg

register_mjlab_task(
  task_id="Unitree-G1-RecoveryV2",
  env_cfg=unitree_g1_recovery_v2_env_cfg(),
  play_env_cfg=unitree_g1_recovery_v2_env_cfg(play=True),
  rl_cfg=unitree_g1_recovery_v2_ppo_runner_cfg(),
  runner_cls=TransitionOnPolicyRunner,
)
