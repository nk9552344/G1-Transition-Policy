"""Recovery-v1: full floor-recovery task for the Unitree G1.

Trains a policy to stand up from any ground-level starting orientation
(supine, prone, side-left, side-right) as well as all the bent-upright
configurations introduced in transition-v3.  The final target state is
identical to all prior transition policies: the default neutral standing
pose (HOME_KEYFRAME).
"""
