# run01_baseline

- **Date:** 2026-06-26
- **Checkpoint:** `logs/hv-statue-manager/20260624_111307-statue_28dof_fast_sac_manager-locomotion/model_0050000.pt` (step 50000); ONNX exported alongside.
- **W&B run:** KAIST_AI/hv-statue-manager/x2nnogg6

## Changes from previous run
None — this is the baseline. Reward weights and domain randomization inherited
directly from the G1 FastSAC preset, no Statue-specific tuning yet.

## Hypothesis
Establish where the inherited-G1 policy stands against the Statue spec, and
locate the first failure mode from data.

## Result (from scorecard.txt)
- **Translation meets spec** (fwd/lat/back: 0% falls, tracking RMS in range).
- **Turning is the headline failure:** yaw ±0.5 and all mixed commands fall 100%
  in MuJoCo — but ~0% in IsaacGym. The policy *learned to turn*; turning *does
  not transfer*. ⇒ sim2sim problem (yaw/contact), not training.
- Hardware-safety fails: torque safety factor 1.00 (peak hits actuator limit),
  516k joint-position-limit violations.
- Symmetry 0.118 (just over the 0.10 target).
- Forward tracking degrades with speed (rms_vx 0.33 @ 1.0 m/s commanded).

## → Next lever
Investigate the yaw transfer mechanism before changing anything (compare a yaw
rollout IG vs MuJoCo: base tilt, foot slip, torque saturation). Then, per spec
§4.4 ("sim2sim gap large → revisit DR / obs normalization"), most likely widen
foot-contact / friction (and possibly motor-strength) DR and retrain.
