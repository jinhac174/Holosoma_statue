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

- Foot **scuffing fails** (scuff fraction 0.066 vs target ≤0.02): a foot drags
  near the ground ~7% of the time. (The swing-apex "min foot clearance" metric
  looked fine at 0.07 m — apex height hides drag; the horizontal-speed-near-ground
  detector catches it.)

## Failure-mode search — yaw transfer (done)
Compared a `yawL05` rollout IG (survives) vs MuJoCo (falls @5.3s):

- **IsaacGym:** tracks yaw 0.5 steadily, tilt stable ~3.5°, height 0.77 throughout.
- **MuJoCo:** turns fine for ~3 s, then stance-foot slip accumulates → tilt grows
  (4°→11°→57°) → catastrophic fall at ~5.3 s, ending face-down (tilt >100°).
- **Smoking gun — stance-foot slip during contact:** MuJoCo 0.18 m/s vs IG 0.09 m/s.
  Aggregated over 20 rollouts/command: pure-yaw slip ratio MuJoCo/IG = **1.5–2.2×**,
  while translation (fwd/lat) is ~1.0× (similar or lower). Turning specifically
  induces excess foot slip in MuJoCo.

**Mechanism:** turning loads the foot in *torsional/rotational* contact. Training
friction DR randomizes only **sliding** friction `[0.5, 1.25]`; torsional contact
is never varied, so the policy learned a turn that relies on PhysX's foot grip and
slips out in MuJoCo. This is the "contact model differences → sliding" failure mode
(§3.4) — a sim2sim problem, confirmed not a training-capability problem.

## → Next lever (run02)
Per spec §4.4 ("sim2sim gap large → revisit DR / motor strength / friction"):
widen contact DR in training — lower the friction floor and/or randomize foot
torsional friction (and consider motor-strength DR) — so the policy learns turning
robust to low foot grip. Re-run the full eval and compare yaw fall rate + stance
slip. (Secondary: torque-margin and symmetry fails will likely need their own
reward-weight tuning afterward.)
