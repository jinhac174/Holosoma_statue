# run02_angvel_fix

- **Date:** 2026-06-26
- **Checkpoint:** SAME as run01 — `model_0050000` (step 50000). **No retrain.**
- **Type:** sim2sim bridge bug fix (debugging, not training).

## Changes from previous run
Removed the erroneous world→body rotation of the base angular velocity in the
ZMQ bridge (`bridge/zmq/zmq_bridge.py`) and the eval runner. MuJoCo's freejoint
`qvel[3:6]` is already body-frame (what the policy expects); the extra rotation
was ≈identity when upright but, while turning, rotated the roll/pitch angular-
velocity components by the accumulating yaw angle and corrupted the balance
feedback → 100% fall on yaw/mixed.

## How it was found
Friction ablation diagnostic: bumping foot torsional friction, then cranking all
friction 5×, left yaw fall at 100% → friction ruled out. Toggling the angular-
velocity rotation took yaw fall 100% → 0% (translation unaffected), across the
whole command grid.

## Result (from scorecard.txt)
- **Fall rate 33.3% → 0.0%.** Every command survives, including yaw and mixed.
- **Sim2sim gap PASS on all axes** (vx −10%, vy −9%, yaw −19%; spec <30%).
  MuJoCo fall 0.0% vs IsaacGym 0.9%. Sim2sim transfer solved with no retraining.
- Still PASS: tracking RMS (0.115/0.122/0.120), CoT (0.607), foot clearance,
  joint pos-limits.
- Still FAIL (genuine, now on a turning robot): symmetry 0.114 (≤0.10),
  torque safety 1.00 (≥1.25, ankle_roll binding), scuff 0.069 (≤0.02).

## → Next lever (run03 — first real retrain)
Reward-weight tuning for the three remaining fails, one lever at a time:
torque penalty (ankle_roll margin) and/or symmetry reward weight; revisit the
foot-clearance/scuff term. Retrain → eval both sims → compare.
