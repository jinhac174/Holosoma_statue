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

## run10_torque — 2026-06-29
**Changed:** added `penalty_torque` = Σ(τ/stall)² (weight −0.5), targeting torque margin.
**Result (MuJoCo 30/cmd):** torque safety **1.00 unchanged** — −0.5 too weak to remove the
ankle-peak spikes. Tracking fine (vx .113/vy .128/yaw .176), symmetry 0.102, scuff 0.0, fall 0%.
**→ run11:** increase penalty_torque −0.5 → −1.0.

## run11_torque1 — 2026-06-29 (training)
**Changed:** penalty_torque weight −0.5 → −1.0 (torque unmoved at −0.5; tracking had room).

## run11_torque1 — 2026-06-29 (KEEP: torque dist improving)
penalty_torque -0.5 -> -1.0. min safety still 1.00 (worst-case DR outliers) BUT distribution
clearly up: %rollouts torque-safe 22%->38%, mean 1.29->1.32. Tracking held (vx .111/vy .136/
yaw .165), scuff 0, symmetry 0.153 (noise). The penalty works; worst-case ankle spike on
hardest DR persists. Also fixed GPU-2 graphics leak (--logger.video.enabled False -> graphics_device_id=-1).
**→ run12:** penalty_torque -1.0 -> -2.0 (push distribution further).

## run12_torque2 — 2026-06-30 (REVERT: penalty plateaued)
penalty_torque -1.0 -> -2.0. Torque %pass 38%->40% (mean 1.32->1.34) — marginal; min still 1.00.
Tracking vx degraded 0.111->0.135. Diminishing returns + tracking cost -> revert to run11 (-1.0).
CONCLUSION (torque): penalty_torque improves the torque distribution (11%->38% rollouts safe,
mean 1.13->1.32) but cannot reach the strict spec min>1.25 "at all times" — the 60 Nm ankle_roll
saturates on worst-case DR rollouts regardless. Best torque setting = run11 (-1.0).
NEXT (user decision): (a) lower ankle_roll kp to cap torque capacity (risks balance), or
(b) accept distribution result + document worst-case as hardware-ankle-limited.
**New best policy = run11** (torque dist 38% vs run05 11%; tracking holds; sym noise).

## run13_torquehinge — 2026-06-30 (REDESIGNED penalty_torque — training)
**Why:** runs 10-12 proved the OLD penalty (sum (tau/limit)^2 over ALL torque) is the wrong
shape — it penalizes useful in-envelope torque, biases toward a lazy gait, and (because applied
torque is CLIPPED at the limit) has ~zero gradient exactly at saturation, so it plateaued.
**Change (2 edits, user-directed):**
  1. joint_control.py now caches the PRE-CLIP PD torque demand as `torques_raw`.
  2. penalty_torque is now a one-sided HINGE on the demand:
       sum_j ( relu(|tau_j| - 0.8*limit_j) / limit_j )^2
     = 0 below 0.8*limit, quadratic above, harder the more it exceeds, with LIVE gradient
     past saturation (1.5*limit penalized harder than 1.0*limit). Directly targets the
     safety-factor spec (peak <= 0.8*stall) instead of generic torque minimization.
  3. weight -1.0 -> -25.0 (hinge magnitude is ~50-100x smaller; -25 makes a saturating ankle
     cost ~ -1 reward vs alive +10 / tracking +2). First point of a new weight sweep.
**Hypothesis:** worst-case min torque safety factor rises above 1.00 (ideally toward 1.25)
WITHOUT the run12-style tracking degradation, because in-band torque is no longer penalized.
**Result (full battery, MuJoCo 1200 + IG 1200 + push + rough):** MIXED — the hinge SHAPE is
validated, but weight -25 has side effects.
  WINS: tracking improved on every axis (vx .110->.107, vy .135->.121, vyaw .165->.116 — and
        NO run12-style degradation, the hinge preserves tracking); ROUGH now PASSES (5.2%->4.8%);
        mean torque safety up (1.323->1.369), median up (1.091->1.173) = fewer extreme
        over-commands. sim2sim gap PASS (vx -16/vy -9/vyaw -27%); push 0%; scuff 0; CoT 1.35.
  LOSSES: SYMMETRY clearly WORSE, incl. forward-walk (0.046->0.170, n=300) — not the low-energy
        artifact this time; weight -25 induced a genuinely asymmetric gait (policy likely favors
        one leg to hold ankle torque down). Torque %rollouts strictly-safe(>=1.25) DOWN 39%->31%,
        and worst-case min STILL 1.000 (ankle_roll clips on hardest DR — hardware-bound, same as
        run11/12 regardless of penalty).
**Read:** the redesigned hinge is the right TOOL (tracking preserved/improved, mean margin up,
unlike the old clipped-squared term), but -25 is TOO AGGRESSIVE — it bought central-tendency
torque margin at the cost of symmetry without raising the strict-safe fraction. Best BALANCED
policy stays run11 until a lower hinge weight keeps run11's symmetry+strict-torque AND run13's
tracking/rough.
**-> run14:** lower hinge weight (-25 -> ~-10), ADD logger:wandb + video (with GPU-6 startup
guard). Goal: keep mean-margin/tracking/rough gains, recover symmetry. Torque strict-min is now
confirmed hardware-limited across 11/12/13 -> separate decision: ankle_roll kp down, or accept +
document. (run14b option, user-raised: add dof_torque=tau/limit from torques_raw to critic_obs
only — privileged critic feature, zero deployment impact, to sharpen credit assignment.)

## run14_critic_lower — 2026-06-30 (hinge -10 + dof_torque in critic_obs + W&B)
**Changed (2 levers at once):** penalty_torque hinge weight -25 -> -10; added dof_torque
(tau/limit, pre-clip) to critic_obs ONLY (+ mirror_obs_dof_torque for use_symmetry). W&B
scalars on (video OFF — it leaked graphics onto another GPU). W&B run ddvtz2p3.
**Result (full battery):** SYMMETRY RECOVERED, TORQUE COLLAPSED — a clean Pareto move.
  vs run13: symmetry agg 0.235->0.132, fwd-walk 0.170->0.076 (PASSES <0.10!); tracking best
  yet (vx .102/vy .114/vyaw .134); CoT 0.698 (very efficient); rough 3.8% (PASS); push 0%.
  BUT torque distribution COLLAPSED: %rollouts safe(>=1.25) 31%->8.5%, mean 1.369->1.152,
  median 1.173->1.000 (half the rollouts now saturate a joint). Per-command sf ~1.00 on
  almost every command. min still 1.000.
**Read:** penalty_torque weight is a direct TORQUE<->SYMMETRY/tracking knob. -25 = good torque
mean, broken symmetry. -10 = recovered symmetry + best tracking/CoT, but torque careless.
The sweet spot is between (~-15..-18). The critic-torque effect is CONFOUNDED with the weight
drop (both moved); torque got worse not better, so the critic did NOT rescue torque here — its
benefit (if any) can't be isolated from this run. Symmetry strict spec still failed (0.132) but
fwd-walk passes; torque strict-min STILL 1.000 across run11/12/13/14 = definitively
HARDWARE-BOUND (no penalty weight reaches it).
**-> run15:** weight ~-15 (Pareto knee: claw back torque distribution while keeping run14's
symmetry/tracking/rough). Keep dof_torque in critic (free, may help at the knee). Torque
strict-min: accept + document as hardware-limited (ankle_roll undersized) unless we try
ankle_roll kp. Best-balanced so far = run14 (fails only torque-dist + marginal symmetry;
tracking/CoT/rough/push/scuff/sim2sim all pass) — pending the run15 knee.

## run15_ankle_actionscale — 2026-07-01 (REVERTED — negative result, but decisive)
**Changed (physical torque cap, on top of run14):** ankle_roll kp 70->50 + action_scale
0.25->0.20. Hypothesis: cap ankle torque by construction -> raise safety factor.
**Result:** BACKFIRED on tracking, did NOT fix torque.
  torque: min STILL 1.000, %safe 8.5%->11.7% (≈flat), median still 1.000; ankle_roll still the
    saturating joint in 1086/1200 rollouts.
  tracking: ALL axes now FAIL (vx .102->.171, vy .114->.164, vyaw .134->.251); vx@1.0 RMS 0.436
    (can't reach top speed with reduced authority). pos-limit violations 0->68. rough 3.8%->8.3%
    (FAIL). sim2sim vyaw gap +50% (FAIL). fall ~0.1%.
**Mechanism (the real lesson):** ankle_roll saturation is BALANCE-driven, not authority-driven.
Lower kp produces less torque per unit error, but the robot still NEEDS max ankle torque to stay
upright under worst-case DR, so it drives the ankle to the clip anyway — just with larger position
error -> worse tracking + joints pushed past limits, for ZERO torque benefit. You cannot cap
worst-case ankle torque via kp without falling.
**CONCLUSION (torque, now THREE independent failures):** reward shaping (run10-13), penalty
weight (run14), and physical kp/action-scale cap (run15) ALL leave min=1.000 with ankle_roll
saturating. The strict spec "min torque safety > 1.25 at all times" is DEFINITIVELY hardware-bound:
the 60 Nm ankle_roll is undersized for worst-case lateral/balance loads on this robot. Accept +
document; optimize the DISTRIBUTION instead (best = run11, 39% safe). Reverted kp + action_scale.
**Best policy = run14** (only torque-dist + marginal symmetry 0.132 fail; everything else passes).
**-> run16:** return to the torque<->symmetry knee (penalty weight ~-15, run14 base) as the last
reward-side lever; otherwise lock run14 as final and write up torque as hardware-limited.

## run16_weight15 — 2026-07-01 (KEEP — NEW BEST, the knee)
**Changed:** penalty_torque hinge weight -10 -> -15 (only change from run14; hinge + dof_torque
in critic retained). W&B run xea391fx.
**Result (full battery):** best-balanced policy of the campaign -- it hit the knee.
  torque: %rollouts safe>=1.25 = 43.8% (BEST, beats run11's 39.3%), mean 1.271, median 1.190;
    forward-walk sf strong again (vx0.5 1.40 / vx0.8 1.44 / vx1.0 1.50). min still 1.000 (ankle
    hardware-bound, as always).
  symmetry: agg 0.118 (BEST aggregate, closest to 0.10), fwd-walk 0.051 (PASSES, ~run11's 0.046).
  tracking: vx 0.110 / vy 0.135 / vyaw 0.142 (all PASS). CoT 1.226. clearance 0.114 (best). scuff
    0. pos-limit 0. fall 0%. sim2sim gap -11/-1/-21% (PASS). push 0%.
  ONLY regressions: rough 3.8%->5.3% (marginal FAIL, heavier torque penalty slightly stiffens the
    gait); symmetry agg still >0.10 (but best yet + fwd passes).
**Read:** -15 recovers run11-level torque distribution (actually better, 43.8%) AND run14-level
symmetry simultaneously -- the torque<->symmetry knee we were looking for. Passes every spec item
except the three known ones: torque strict-min (hardware-bound), aggregate symmetry (marginal,
low-energy-command artifact; fwd passes), rough (5.3%, marginal).
**NEW BEST = run16.** Remaining: rough is a hair over 5% (run14 had it at 3.8% -- run-to-run
variance / torque-penalty stiffness); decide whether to nudge it or accept. Torque min + aggregate
symmetry are the documented hardware/metric limits.

## run17_roll (intermediate, visual-only) — 2026-07-02
**Changed:** penalty_ang_vel_xy -1.0 -> -3.0 (damp the step-synced lateral roll rocking).
**Read:** user assessed in MuJoCo — lateral roll "a lot better", but foot clearance too high and
stance too narrow (feet A-frame inward). Not formally eval'd; superseded by run17_stance_clearance.

## run17_stance_clearance — 2026-07-02 (KEEP — big improvement, symmetry the cost)
**Changed (stacks the roll fix):** penalty_ang_vel_xy -1->-3; feet_phase swing_height 0.15->0.10
  (foot clearance); penalty_close_feet_xy threshold 0.15->0.20 (stance width).
**Result (vs run16):** roll RMS fwd 2.54->1.37 deg (rocking ~halved); foot clearance mean 0.136->0.084m,
  min 0.104->0.070m; stance width 0.335->0.353m; CoT 1.226->0.573; rough fall 5.3%->2.8% (now PASS);
  tracking vx 0.110->0.088, vyaw 0.142->0.111; ankle_roll torque IMPROVED (left peak 1.67->1.42x,
  saturation 0.25%->0.01%). REGRESSION: symmetry 0.118->0.167 (worse, still FAIL); vy 0.135->0.146;
  scuff 0.000->0.003 (still PASS). torque min still 1.000 (hardware-bound). push 0%, self-collision 0.
**Read:** all three visual targets (roll, clearance, stance) improved + CoT/rough/tracking/torque bonus.
  Symmetry is the price (both aggregate and fwd-walk ~doubled). Widening stance did NOT load ankle_roll
  as feared — it helped. KEEP as new base.

## run18_feetslip — 2026-07-02 (launched, GPU6 video-off)
**Changed (from run17_stance_clearance):** + penalty_feet_slip (NEW term) weight -2.0 — penalize
  sum of contact-foot xy-speed^2 (foot must not slide while planted).
**Hypothesis:** fixes the MuJoCo standing outward-slip limit cycle (feet slide out, near-fall, recover,
  repeat). Root cause: penalty_torque makes holding hip_roll adductors expensive, so a splayed low-torque
  stance is "free" and there's no slip term to oppose the drift. This term is orthogonal to the torque
  budget. **Watch:** standing foot drift (should stop), symmetry (already regressed — hopefully not worse),
  tracking, and that stance-foot micro-motion during normal gait isn't over-penalized (would stiffen gait).
