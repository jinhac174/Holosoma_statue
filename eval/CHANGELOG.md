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

<!-- template for next entry:
## runNN_<short-name> — <date>
**Checkpoint:** ...
**Changed from previous:** <reward term / DR / config> because <metric from prev scorecard>.
**Result:** <which thresholds moved>.
**→ Next:** ...
-->
