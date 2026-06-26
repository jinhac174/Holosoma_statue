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

<!-- template for next entry:
## runNN_<short-name> — <date>
**Checkpoint:** ...
**Changed from previous:** <reward term / DR / config> because <metric from prev scorecard>.
**Result:** <which thresholds moved>.
**→ Next:** ...
-->
