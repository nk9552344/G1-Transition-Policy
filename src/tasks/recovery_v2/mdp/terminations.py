"""Termination functions for recovery-v2.

Reuses bad_orientation_while_elevated from recovery-v1 unchanged.
The height gate of 0.50 m is still correct for all four fallen orientations —
side-lying also settles at h ≈ 0.10–0.15 m, well below the gate.
"""

from src.tasks.recovery_v1.mdp.terminations import bad_orientation_while_elevated  # noqa: F401
