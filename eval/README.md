# Statue Evaluation Workflow

Repeatable train → sim2sim → analyze → iterate loop. Every iteration produces a
self-contained, threshold-checked **run folder** so changes are comparable.

## Layout

```
eval/
├── CHANGELOG.md                 # one line per run: what changed & why (the iteration trail)
├── runs/
│   └── runNN_<name>/
│       ├── run_info.md          # checkpoint, date, what changed, hypothesis, result
│       ├── scorecard.txt        # ★ spec PASS/FAIL + per-command fall + sim2sim gap
│       ├── metrics_mujoco.csv    # per-rollout metrics (tracked in git)
│       ├── metrics_isaacgym.csv
│       ├── plots/               # 6 metric figures, IG vs MuJoCo overlaid (tracked)
│       ├── mujoco/              # raw NPZ rollouts (gitignored, purgeable)
│       └── isaacgym/            # raw NPZ rollouts (gitignored, purgeable)
└── _archive/                    # old/ad-hoc runs (gitignored)
```

Git tracks the small artifacts (scorecard, csvs, plots, run_info, CHANGELOG);
the bulky raw NPZ (`mujoco/`, `isaacgym/`) are gitignored — purge them after the
report if disk is tight (the CSVs keep all per-rollout numbers; only the
torque-vs-time plot needs raw NPZ to regenerate).

## One iteration

Pick the next run number/name, then:

```bash
RUN=eval/runs/run02_<name>
CKPT=logs/.../model_XXXXXX.pt          # the .pt; ONNX is exported alongside
ONNX=logs/.../model_XXXXXX.onnx

# 1. MuJoCo sim2sim suite (flat, DR on, 100/command)  — hsmujoco env
source scripts/source_mujoco_setup.sh
python -m holosoma.eval.run_suite --model-path $ONNX --out-dir $RUN/mujoco \
    --n-per-command 100 --duration 12

# 2. IsaacGym suite (flat, matched commands, 100/command)  — hsgym env, GPU
source scripts/source_isaacgym_setup.sh
CUDA_VISIBLE_DEVICES=6 python -m holosoma.eval.ig_suite --checkpoint $CKPT \
    --out-dir $RUN/isaacgym --n-per-command 100 --duration 12

# 3. Report: scorecard + csvs + all plots  — hsmujoco env
source scripts/source_mujoco_setup.sh
python -m holosoma.eval.report --run-dir $RUN

# 4. Record the iteration
#    - edit $RUN/run_info.md   (what you changed, hypothesis, result)
#    - add one entry to eval/CHANGELOG.md
#    - (optional) rm -rf $RUN/mujoco $RUN/isaacgym   # purge raw NPZ
```

The scorecard prints to stdout and is saved to `$RUN/scorecard.txt`. Compare
runs by diffing scorecards or reading `CHANGELOG.md`.

## Spec thresholds (the bar each run is graded against)

| metric | target | scorecard field |
|---|---|---|
| tracking RMS vx, vy | < 0.15 m/s | tracking RMS vx/vy |
| tracking RMS yaw | < 0.2 rad/s | tracking RMS vyaw |
| torque safety factor | > 1.25 | min torque safety factor |
| symmetry index | < 0.1 | symmetry index |
| CoT @ 0.8 m/s | < 2.5 | cost of transport |
| joint pos-limit violations | 0 | joint pos-limit violations |
| foot scuffing | min clearance > 0 | min foot clearance |
| sim2sim degradation | < 30% | SIM2SIM GAP section |

## Notes on methodology

- **Tracking + sim2sim gap are measured on flat terrain**, in-spec command grid,
  DR on (friction/mass/initial-state). This is the deployment-facing comparison.
- **Rough terrain and push survival are separate robustness checks** (spec §1),
  not part of the tracking/gap numbers — run them as dedicated suites when needed.
- The aggregate sim2sim "gap %" only averages over *surviving* rollouts; always
  read the **fall-rate** line and the **TRANSFER FAILURES** table in the
  scorecard — that's where transfer breaks show up (e.g. run01: yaw/mixed).
- MuJoCo runner fidelity is verified by `python -m holosoma.eval.validate`.
