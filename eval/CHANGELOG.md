# Statue Locomotion — Eval / Iteration Log

Cross-run history of the train → sim2sim → analyze → iterate cycle. One entry
per run. Each run's full numbers live in `runs/<run>/scorecard.txt`; this file
is the short "what changed and why" trail.

Spec thresholds (targets to beat):

| metric | target |
|---|---|
| tracking RMS lin (vx, vy) | < 0.15 m/s |
| tracking RMS yaw | < 0.2 rad/s |
| torque safety factor | > 1.25 (peak ≤ 0.8·stall) |
| symmetry index | < 0.1 |
| CoT @ 0.8 m/s | < 2.5 |
| joint pos-limit violations | 0 |
| foot scuffing | none (min swing clearance > 0) |
| sim2sim tracking degradation | < 30% |

---

## run01_baseline — 2026-06-26
**Checkpoint:** `model_0050000.pt` / `.onnx` (step 50000), G1 reward+DR inherited as-is.

**Scorecard headline (MuJoCo, 1200 rollouts):**
- Translation (fwd/lat/back): in spec, 0% falls.
- **Yaw ±0.5 and all mixed commands: 100% fall.**
- Forward under-tracks at speed (rms_vx 0.33 @ 1.0 m/s).
- FAIL symmetry (0.118), torque safety (1.00), pos-limit violations (516k).
- PASS tracking RMS (avg over survivors), CoT (0.63), foot clearance.

**Sim2sim:** translation gap −7..−21% (PASS). Fall rate IG 0.9% vs MuJoCo 33.3%.
**Turning is learned in IsaacGym (≈0% fall on yaw) but fails to transfer to MuJoCo
(100% fall)** → sim2sim / yaw-contact problem, not a training problem.

**Failure-mode search (yaw):** stance-foot slip during contact is **1.5–2.2× higher
in MuJoCo for pure-yaw commands** (vs ~1.0× for translation). A yaw rollout turns
fine ~3 s, then slip accumulates → tilt 4°→57° → falls @5.3 s. Mechanism: turning
loads *torsional* foot contact, but training only randomizes *sliding* friction, so
the learned turn relies on PhysX grip and slips out in MuJoCo. Confirmed sim2sim
(contact), not training. Also: foot **scuffing fails** (0.066 vs ≤0.02).

**→ Next (run02):** widen contact DR in training — lower friction floor and/or
randomize foot torsional friction (± motor-strength DR) — retrain, re-eval, compare
yaw fall rate + stance slip. See `runs/run01_baseline/run_info.md`.

## run02_angvel_fix — 2026-06-26
**Same policy as run01 (model_0050000). NOT a retrain — a sim2sim bridge bug fix.**

**Changed:** removed an erroneous world→body rotation of the base angular velocity in
the ZMQ bridge (`zmq_bridge.py`) + eval runner. MuJoCo's freejoint `qvel[3:6]` is already
body-frame; the extra rotation was ≈identity upright but, while turning, rotated the
roll/pitch angular-velocity by the accumulating yaw → corrupted balance feedback → fell.

**Found by:** friction ablation diagnostic (ruled out friction: even 5× sliding + high
torsional → still 100% fall), then toggling the rotation (100% → 0%).

**Result:** **fall rate 33.3% → 0.0%**; yaw/mixed 100% → 0% fall. Sim2sim gap now
PASS on all axes (vx −10%, vy −9%, yaw −19%; spec <30%). MuJoCo fall 0.0% vs IG 0.9%.
Sim2sim transfer **solved, no retraining** (spec §3 satisfied).

**Genuine remaining fails (now on a robot that turns):** symmetry 0.114 (≤0.10),
torque safety 1.00 (≥1.25; ankle_roll binding), scuff 0.069 (≤0.02). Tracking/CoT/
clearance/pos-limits all PASS.

**→ Next (run03, first real RETRAIN):** reward-weight tuning for the three fails.
Likely: torque penalty (ankle_roll margin) + symmetry weight; revisit foot-clearance/
scuff term. One lever at a time.

## run03_actionrate — 2026-06-26  (REVERTED, negative result)
**Changed:** FastSAC `penalty_action_rate` −2.0 → −4.0, targeting torque margin (ankle_roll
saturating). Trained full 50k.

**Result:** **net regression, reverted.**
- torque safety: 1.000 → 1.000 (no change — action_rate cannot touch torque demand).
- training curves worse: tracking_lin_vel reward −25%, tracking_ang_vel −20%, episode
  length −14% (less stable). Eval tracking RMS vx 0.115 → 0.139.
- symmetry "improved" 0.114 → 0.085 (PASS) — but this is **incidental**: over-penalizing
  actions produced a sluggish/conservative gait that is naturally more symmetric, not a
  genuinely better policy. Curves confirm the policy got worse.

**Lessons:** (1) read training curves alongside the eval scorecard — symmetry PASS alone
was a false positive. (2) symmetry is **not converged at 15k** (0.223@15k → 0.085@50k), so
gait-quality metrics must be judged at full 50k, not the 15k quick-eval.

**Conclusion:** existing minimalist knobs are exhausted for torque (needs a torque term) and
symmetry (augmentation already on; only conservatism moved it). Reverted action_rate to −2.0.

## run02 — spec-grade robustness battery (real tests) — 2026-06-27
Upgraded the robustness tests to be spec-faithful and re-ran on run02 (baseline policy):
- **Push: torso link** (`waist_pitch_link`), 100 N/0.2 s, 600 rollouts → **0% fall**.
  (Earlier 0% was a pelvis push; torso has more leverage and still survives.)
- **Rough terrain: Holosoma's real `terrain_locomotion_mix`** (trimesh rough 0.6, the
  trained terrain), IsaacGym, 600 rollouts → **3.5% fall**, tracking RMS vx 0.135 (≈ flat).
  (Replaces the earlier custom-heightfield approximation.)
- **Self-collision: 0** events.
Verdict: run02 genuinely **passes the robustness spec** with real tests. Remaining
rigor for the final report: bump to 100 rollouts/command.

## run04_feetphase — 2026-06-27  (partial win, kept direction)
**Changed (from run02 baseline, action_rate −2.0):** `feet_phase` swing_height 0.09 → 0.12,
targeting scuff. Trained full 50k. Eval: MuJoCo flat, 50/command.

**Result (vs run02):** genuine, non-conservative improvement —
- foot clearance 0.070 → **0.090** (feet lift higher, mechanism worked)
- scuff 0.069 → **0.056** (↓ ~19%, but STILL FAILS <0.02)
- tracking vx 0.115 → **0.101** (improved); vy/yaw still pass
- symmetry 0.114 → 0.127 (~, fails); torque 1.00 (unchanged, fails); fall 0%
Unlike run03 this didn't trade away tracking/stability → keep the change.

**→ Next:** scuff still ~3× over spec. Options: push feet_phase further (0.15 / higher
weight) to chase scuff <0.02, or accept + document. torque + symmetry still need dedicated
reward terms (existing knobs exhausted).

<!-- template for next entry:
## runNN_<short-name> — <date>
**Checkpoint:** ...
**Changed from previous:** <reward term / DR / config> because <metric from prev scorecard>.
**Result:** <which thresholds moved>.
**→ Next:** ...
-->
