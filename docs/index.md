# Policy Engineering Documentation
## A Technical Reference for Writing Reinforcement Learning Policies on the G1 Humanoid

This documentation is written like a technical book, not a README. Each chapter goes deep into
the theory, the math, the data structures, and the engineering decisions behind every line of
the transition policy. After reading this, you should be able to design, debug, and train any
kind of standing, walking, or motion-tracking policy for the G1 from scratch.

---

## Reading Order

If you are new to this codebase, read the chapters in order. If you are looking for a specific
topic, use the chapter list below.

| Chapter | File | What you learn |
|---------|------|----------------|
| 1 | [ch01_system_architecture.md](ch01_system_architecture.md) | The complete data flow from reset to action to reward to parameter update |
| 2 | [ch02_project_structure.md](ch02_project_structure.md) | Every directory and file, its exact role, and why it is where it is |
| 3 | [ch03_framework_and_imports.md](ch03_framework_and_imports.md) | The mjlab framework: every import explained at the class level |
| 4 | [ch04_robot_model_g1.md](ch04_robot_model_g1.md) | G1 anatomy, joints, motor hardware, armature calculation, action scale derivation |
| 5 | [ch05_env_cfg_anatomy.md](ch05_env_cfg_anatomy.md) | ManagerBasedRlEnvCfg: scene, sim, decimation, episode length — every parameter |
| 6 | [ch06_observations.md](ch06_observations.md) | ObservationTermCfg, actor vs critic split, noise, history, concatenation |
| 7 | [ch07_actions.md](ch07_actions.md) | JointPositionActionCfg, scale, use_default_offset, PD controller theory |
| 8 | [ch08_events_domain_randomization.md](ch08_events_domain_randomization.md) | EventTermCfg modes, every reset function, domain randomization parameters |
| 9 | [ch09_rewards_deep_dive.md](ch09_rewards_deep_dive.md) | Full theory of reward shaping, weight selection, std tuning, every reward function |
| 10 | [ch10_contact_sensors.md](ch10_contact_sensors.md) | ContactSensorCfg, ContactMatch patterns, fields, reduce, history |
| 11 | [ch11_terminations.md](ch11_terminations.md) | TerminationTermCfg, bad_orientation, time_out, how terminations affect training |
| 12 | [ch12_rl_hyperparameters.md](ch12_rl_hyperparameters.md) | PPO theory, every hyperparameter explained and how to tune it |
| 13 | [ch13_task_registration.md](ch13_task_registration.md) | register_mjlab_task, the config split, play vs train modes |
| 14 | [ch14_runner_and_onnx.md](ch14_runner_and_onnx.md) | OnPolicyRunner, checkpoint saving, ONNX export, deployment metadata |
| 15 | [ch15_writing_new_policy.md](ch15_writing_new_policy.md) | Complete step-by-step recipe for authoring a new policy from zero |

---

## The Philosophy of This Codebase

Policies in this codebase are built as **manager-based RL environments**. The environment is
not one monolithic class — it is a configuration object that declares what sensors exist, what
observations to read, what rewards to compute, and what events fire at each episode reset. The
`mjlab` framework reads this configuration and assembles the environment at runtime.

This means writing a new policy is mostly about writing a configuration, not procedural code.
The exceptions are:
- **Custom reward functions** — you write these as plain Python functions.
- **Custom observations** — same pattern.
- **The robot-specific override layer** — a function that takes the base config and applies
  robot-specific sensor names, body names, and scales.

The transition policy (`Unitree-G1-Transition`) trains a neural network that drives the G1
from any random upright configuration back to its home stance. It requires no velocity commands,
no terrain, no gait rewards. It is the simplest meaningful policy in this codebase, which makes
it the ideal starting point for understanding the system.
