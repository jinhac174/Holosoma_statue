# Autonomous Tuning Loop — ledger

Self-driven train → full-eval → diagnose → next-lever loop. Runs until the user returns.

## Rules
- **GPU 6 ONLY** — never use any other GPU for training or eval (shared box; other GPUs are other users).
- **No new reward terms** (only weights/params of already-enabled terms, DR, algo flags).
- **One change per run**, logged below (and in CHANGELOG.md).
- **Keep/revert**: keep a change if a *failing* target metric improves AND no *passing*
  spec metric regresses to failing; otherwise revert to the current best.
- Each run: full eval (MuJoCo+IsaacGym flat 100/cmd, push, rough), report, purge raw NPZ.
- Judge gait-quality metrics at full 50k (15k is unreliable for symmetry/scuff).

## Targets (run04 baseline fails)
- scuff 0.056 (→ <0.02)  — lever: feet_phase swing_height / weight / tracking_sigma
- symmetry 0.127 (→ <0.10) — lever: pose / penalty_feet_ori / close_feet (indirect; FastSAC
  symmetry augmentation already on, no scalar). May be hard without a term.
- torque 1.00 (→ >1.25) — **BLOCKED**: no torque term exists, action_rate proven useless.
  Cannot fix in this loop without a new term. Will remain failed; flagged for the user.

## Current best
- **run04**: feet_phase swing_height=0.12, weight=5.0; action_rate=-2.0 (everything else G1-inherited)

## Lever queue (existing knobs)
1. feet_phase swing_height 0.12 → 0.15        (scuff)
2. feet_phase weight 5.0 → 8.0                (scuff/clearance)
3. feet_phase tracking_sigma 0.008 → 0.005    (scuff)
4. pose weight -0.5 → -1.5                    (symmetry attempt)
5. penalty_feet_ori -5.0 → -10.0             (symmetry/scuff)
(adaptive after that, based on results)

## Iteration log
| run | change (from best) | scuff | symmetry | torque | trk vx | trk vyaw | keep? |
|-----|--------------------|-------|----------|--------|--------|----------|-------|
| run02 | bridge fix (baseline) | 0.069 | 0.114 | 1.00 | 0.115 | 0.120 | baseline |
| run03 | action_rate -2→-4 | 0.072 | 0.085* | 1.00 | 0.139 | 0.118 | REVERT (curves worse; sym=conservatism) |
| run04 | feet_phase swing 0.12 | 0.056 | 0.127 | 1.00 | 0.101 | 0.137 | KEEP (scuff↓, clearance↑) |
| run05 | feet_phase swing 0.15 | 0.049 | 0.109 | 1.00 | 0.101 | 0.179 | KEEP (scuff↓, sym↓; vyaw margin thin) |

Trend: feet_phase swing height ↓scuff, ↑clearance, ↓symmetry — but ↑vyaw-RMS (0.12→0.18,
near 0.20 cap). Swing height near its useful limit. Next: feet_phase **weight** (5→8) or
**tracking_sigma** (0.008→0.005) to tighten swing without raising height further. torque
remains blocked (no term). **Current best = run05.**
