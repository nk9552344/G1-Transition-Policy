"""Recovery-v2: full floor-recovery from any fallen orientation for the Unitree G1.

Extends recovery-v1 by training the policy to stand up from all four ground-level
orientations — supine, prone, side_left, side_right — rather than only supine and
prone.  The final target state is identical to all prior policies: the default
neutral standing pose (HOME_KEYFRAME).

Key differences over recovery-v1:
  • All 4 fallen templates included (recovery-v1 deferred side_left/side_right).
  • 50 / 50 fallen-to-bent split (8 templates: 4 fallen + 4 bent).
  • New orientation_velocity_reward provides analytic gradient for lateral rolling,
    the key motion not covered by upward_base_velocity alone.
  • Longer episode (45 s) and more training (28 001 iterations) to master the
    wider behavioural repertoire.
"""
