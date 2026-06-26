# Gait Evaluation Framework (`holosoma.eval`)

Quantitative locomotion evaluation for the Statue policy, covering assignment §4.
Both simulators write the **same NPZ schema** (`schema.py`), graded by the **same
metrics** (`metrics.py`), so IsaacGym (in-distribution) and MuJoCo (sim2sim, OOD)
are directly comparable.

```
schema.py        canonical NPZ layout + loader + quat/frame utils   (sim-agnostic)
metrics.py       tracking RMS, CoT, symmetry, foot clearance,        (sim-agnostic)
                 torque margin, fall detection
mujoco_runner.py single-process runner: real inference policy +
                 in-process MuJoCo + per-rollout DR  -> NPZ
run_suite.py     orchestrate 100+ rollouts over a command grid -> dir of NPZ
analyze.py       NPZ dir -> metrics.csv + spec pass/fail + sim2sim gap
plots.py         the four required figures
```

The IsaacGym side reuses the existing `agents/callbacks/recording.py`
(`EvalRecordingCallback`), which already writes this schema.

Run everything in the **`hsmujoco`** env (it has `mujoco` + `onnxruntime` +
`holosoma_inference`).

---

## Metrics (spec §4.1)

| Metric | Definition | Spec target |
|---|---|---|
| Tracking RMS (vx, vy, vyaw) | `sqrt(mean((cmd-actual)^2))`, steady-state only | < 0.15 m/s lin, < 0.2 rad/s yaw |
| Cost of Transport | `mean(Σ\|τ·dq\|) / (m g v)` | < 2.5 at 0.8 m/s |
| Symmetry index | `\|E_L-E_R\| / (0.5(E_L+E_R))` (per-leg mech. energy) | < 0.1 |
| Foot clearance | peak foot height per swing (mean / min) | no scuffing |
| Torque safety factor | `min_j(stall_j / peak_j)` | > 1.25 |
| Fall rate | base too low or tilted past threshold | — |

Steady-state excludes the first 0.5 s after each command change. Fallen rollouts
are scored only up to the fall.

---

## MuJoCo sim2sim evaluation

```bash
source scripts/source_mujoco_setup.sh
MODEL=~/repos/holosoma/model_0050000.onnx

# Run the suite (100 rollouts/command x 12 commands, 12s each, DR on ~ 45 min)
python -m holosoma.eval.run_suite --model-path $MODEL --out-dir ~/eval/mujoco \
    --n-per-command 100 --duration 12

# Quick smoke (1/command, 6s):
python -m holosoma.eval.run_suite --model-path $MODEL --out-dir /tmp/mj_eval \
    --n-per-command 1 --duration 6

# Aggregate -> metrics.csv + spec report
python -m holosoma.eval.analyze --rollout-dir ~/eval/mujoco

# Figures
python -m holosoma.eval.plots --rollout-dir ~/eval/mujoco
```

Per-rollout DR (from the training ranges): friction `[0.5,1.25]`, added base mass
`[-1,3] kg`, link-mass scale `[0.9,1.2]`, initial joint noise `±0.05 rad`. Use
`--no-dr` for the nominal model.

The runner reuses the real `LocomotionPolicy`, so the observation/action path is
identical to the live ZMQ sim2sim — just single-process, seeded, and headless.

---

## IsaacGym (training-distribution) evaluation

Uses the existing recording callback (writes the same schema):

```bash
source scripts/source_isaacgym_setup.sh   # or the IsaacGym env
python src/holosoma/holosoma/eval_agent.py \
    --checkpoint=<CKPT or wandb://...> \
    --recording.config.enabled True \
    --recording.config.output-path ig_rollout.npz \
    --training.max-eval-steps 600
```

This writes one trajectory (command resamples internally). Run a few times with
different seeds for a distribution. (Future: extend the callback to log all
parallel envs for hundreds of rollouts in a single run.)

---

## Sim2sim gap (spec: degradation < 30%)

```bash
python -m holosoma.eval.analyze --rollout-dir ~/eval/mujoco --compare-dir ~/eval/isaacgym
python -m holosoma.eval.plots   --rollout-dir ~/eval/mujoco --compare-dir ~/eval/isaacgym
```

Prints per-axis tracking-RMS degradation MuJoCo-vs-IsaacGym and overlays both on
the figures.

---

## Reading results → tuning (spec §4.4)

- torque margin poor (safety factor < 1.25) → tighten the torque penalty, retrain
- symmetry poor (> 0.1) → raise symmetry reward weight / `algo.config.use-symmetry`, retrain
- sim2sim gap large (> 30%) → revisit DR (mass/friction/motor strength) or obs normalization, retrain
