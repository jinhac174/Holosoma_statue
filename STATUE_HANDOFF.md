# Statue 28-DOF Sim2Sim & Eval — Handoff

> Living context doc for the Statue locomotion sim2sim + evaluation work.
> If you are a fresh Claude session: read this top-to-bottom, then continue from **§7 Next steps**.
> Last updated after sim2sim pipeline was verified working and the eval module was scoped (not yet built).

---

## 1. What this is

Statue is a **28-DOF humanoid** (DeLabor, for shipyard welding/grinding/fit-up). The goal of this
work stream: take a locomotion policy **trained in IsaacGym**, run it in **MuJoCo** via Holosoma's
sim2sim inference pipeline (no retraining), **quantify the sim2sim gap**, then **tune** (reward
weights + domain randomization) to meet a tight locomotion spec.

Assignment stages: **3. Sim2Sim** (done up to driving the policy) → **4. Analysis** (build a
quantitative gait-eval framework — this is the main remaining build).

### Joint layout (28 DOF)
`left leg (6) → right leg (6) → waist (2) → left arm (7) → right arm (7)`
This order is consistent across the MJCF actuators, the inference `dof_names`, and the training
order (verified — see §5).

---

## 2. Paths, envs, model

| Thing | Location |
|---|---|
| Repo | `~/repos/holosoma` (fork: `git@github.com:jinhac174/Holosoma_statue.git`) |
| Robot config | `src/holosoma/holosoma/config_values/robot.py:1108` (`statue_28dof`), MJCF at `:1544` (`statue/statue.xml`) |
| MJCF + meshes | `src/holosoma/holosoma/data/robots/statue/` (tracked in git) |
| Inference config | `src/holosoma_inference/holosoma_inference/config/config_values/inference.py:34` (`statue_28dof_loco`, CLI key `statue-28dof-loco`) |
| Conda envs | `hsmujoco` (sim), `hsinference` (policy) — **NOT in git, rebuild per repo install docs** |
| Setup scripts | `scripts/source_mujoco_setup.sh`, `scripts/source_inference_setup.sh` |
| Trained ONNX model | `model_0050000.onnx` (~940KB, **gitignored** — copy manually). Original lived under `logs/hv-statue-manager/20260624_111307-statue_28dof_fast_sac_manager-locomotion/` |

**Rendering**: this server has NVIDIA GPUs. Use `MUJOCO_GL=egl`. If rendering is slow (~1fps), the
software Mesa renderer was picked — force the GPU with `MUJOCO_EGL_DEVICE_ID=0`.

---

## 3. How to run sim2sim (two terminals)

**Terminal 1 — Sim (MuJoCo + viewer):**
```bash
cd ~/repos/holosoma
source scripts/source_mujoco_setup.sh
fuser -k 5555/tcp 5556/tcp 2>/dev/null   # free ZMQ ports if a prior run lingered
MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 python src/holosoma/holosoma/run_sim.py robot:statue-28dof
```

**Terminal 2 — Policy (inference):**
```bash
cd ~/repos/holosoma
source scripts/source_inference_setup.sh
python src/holosoma_inference/holosoma_inference/run_policy.py inference:statue-28dof-loco \
    --task.model-path ~/repos/holosoma/model_0050000.onnx \
    --task.no-use-joystick --task.interface lo --task.auto-start
```

**Controls (in Terminal 2 unless noted):** `=` walk/stand toggle · `w/a/s/d` velocity ·
`]` policy on/off · in the **viewer window**: `x` walk toggle, `9` toggle gantry, `Backspace`/`Del` reset.

**Headless eval + video (one command):**
```bash
source scripts/source_mujoco_setup.sh
python src/holosoma/holosoma/eval_sim.py \
    --model-path ~/repos/holosoma/model_0050000.onnx \
    --duration 60 --save-dir ~/statue_eval
```
Spawns sim (headless) + scripted policy + records 30fps video, then SIGTERM-flushes cleanly.
Random velocity resamples every 10s (matches the IsaacGym eval cadence).

---

## 4. Changes made this work stream (all committed)

| Area | File | What / why |
|---|---|---|
| **Angular velocity frame** | `bridge/zmq/zmq_bridge.py` | MuJoCo freejoint `qvel[3:6]` is **world frame**; policy expects **body frame** (IMU convention). `_world_to_body_angvel()` rotates it before publishing. **This was the cause of gradual collapse.** |
| **Viewer black screen** | `simulator/mujoco/mujoco.py` `setup_viewer()` | `XInitThreads()` + EGL context API. GLX has no Core Profile here; EGL works. |
| **Spawn height** | `mujoco.py` `load_assets()` | Virtual gantry anchor at z=3m with `length=0` yanked robot up. Set `length = gantry_height - init_z = 2.22m` so spring force = 0 at spawn (z=0.78m). |
| **Reset** | `mujoco.py` `_do_reset()` + `_pending_reset` flag | `Backspace`/`Del` resets pose + gantry length cleanly in the physics thread (was falling from sky). |
| **Ground friction** | `simulator/mujoco/scene_manager.py:202,234,308` | Was hardcoded `0.7` for all terrain types (28% more slippery than training). Set to `1.0` to match training default (`static_friction=1.0`, DR range `[0.5,1.25]`). **Foot geoms left at MuJoCo default — do not edit the MJCF without a source.** |
| **Video pipeline** | `video_recorder.py`, `video_utils.py`, `config_types/video.py` | Capture at `output_fps` (30, not 500). Raw-pipe frames straight to ffmpeg (skips slow cv2/mp4v step). |
| **Clean shutdown** | `run_sim.py` | SIGTERM handler raises `KeyboardInterrupt` so video flushes on `kill`. |
| **auto_start** | `inference .../config_types/task.py`, `policies/base.py` | `--task.auto-start` starts policy without the `]` keypress. |
| **ScriptedInput** | `inference .../inputs/impl/scripted.py` + `inputs/__init__.py` | New input source `scripted`: random velocity sampling, emits `WALK` on start. Wire via `--task.velocity-input scripted --task.state-input scripted`. Config fields `scripted_v{x,y,yaw}_range`, `scripted_command_interval` in `task.py`. |
| **eval_sim.py** | `src/holosoma/holosoma/eval_sim.py` | One-command headless sim+policy+video orchestrator. |

---

## 5. Correctness audit — ALL PASS (one fix)

Verified end-to-end before trusting any eval numbers:

| Check | Result |
|---|---|
| Joint order (MJCF / inference / training) | ✅ identical |
| Default joint angles | ✅ match (`hip_pitch=-0.312, knee=0.669, ankle=-0.363, shoulder_roll=±0.2, elbow=0.6`) |
| KP/KD (vs ONNX metadata) | ✅ exact |
| Action scale | ✅ `0.25 × 28`; `q_target = action*0.25 + default_angles` |
| Observation vector | ✅ 97-dim: `ang_vel(3)+proj_grav(3)+cmd(3)+dof_pos(28)+dof_vel(28)+actions(28)+sin_phase(2)+cos_phase(2)`; scales `ang_vel×0.25, dof_vel×0.05` |
| Gait phase | ✅ `gait_period=1.0`, initial L/R phases `[0, π]` |
| Angular velocity frame | ✅ fixed (see §4) |
| Control frequency | ✅ train 200Hz physics / mujoco 2000Hz physics, **both 50Hz policy**. PD gains have no dt term → identical impulse per step; higher physics rate only improves numerical accuracy. **Not a problem.** |
| Contact / friction | ⚠️→✅ ground fixed 0.7→1.0 |

**Bottom line:** pipeline is fundamentally correct. No scrambled obs, no bad scales, no frame bug.

---

## 6. Current behavior & open issues

- **Walking: good.** Robot follows velocity commands well on flat ground.
- **Single-axis commands: fine.** Forward / lateral / yaw individually track.
- **Mixed (vx+vy+yaw together): falls after a number of steps.** → primary tuning target.
- **Static standing: weak.** Feet slide and it can fall while stationary. Policy was trained for
  walking; standing robustness is under-trained. Friction fix helps but doesn't fully solve it.

These are **policy/tuning issues, not sim bugs** (audit clean). Fix path = reward-weight + DR tuning
and **retrain**, validated with the eval module below.

---

## 7. Next steps — build the Eval Module (assignment §4)

No existing gait-eval framework in Holosoma and the team confirmed **build our own** (do NOT pull in
SymmetryTracker or other modules). This is the main remaining code.

### Spec targets (what "good" means)
- Velocity ranges: vx `0–1.0 m/s`, vy `-0.3–0.3 m/s`, yaw `-0.5–0.5 rad/s`
- **Steady-state tracking RMS:** `< 0.15 m/s` linear, `< 0.2 rad/s` yaw
- **Sim2sim gap:** tracking error degradation IsaacGym→MuJoCo `< 30%`
- **Torque safety factor `> 1.25`** (peak torque ≤ 0.8 × stall)
- Joint positions within URDF limits (no clamping); no self-collision in 30s
- **Symmetry index `< 0.1`**; **CoT `< 2.5` at 0.8 m/s**; no foot-scuffing in swing
- Robustness: survive 100N/0.2s torso push; rough terrain; recover from joint-randomized IC

### Metrics to compute (per rollout)
- Velocity tracking RMS (x, y, yaw): `sqrt(mean((cmd - actual)²))`
- **Cost of Transport:** `Σ|τ·dq| / (m·g·v_forward)` (human ≈0.2, humanoids 1–5)
- **Symmetry index:** `|L−R| / (0.5(L+R))` on stance duration or stance torque integral
- Foot clearance during swing (min foot height when contact < threshold)
- Max joint torque vs stall torque (safety factor)
- Fall rate

### Proposed design (decisions still open — confirm with user)
- **Shared log schema** so IsaacGym and MuJoCo rollouts are directly comparable:
  `t, cmd_vx/vy/vyaw, base_vx/vy/vyaw, base_pos[3], base_quat[4], q[28], dq[28], tau[28], foot_contact[2], foot_pos[2,3]`
- **Two loggers, one analysis layer:**
  - MuJoCo: log from the ZMQ bridge / sim loop (sim is already running — fast iteration)
  - IsaacGym: log via an eval callback in `eval_agent.py`
  - Analysis code reads logs and is simulator-agnostic.
- **Open questions for the user:**
  1. Build order — MuJoCo logger + metrics + plots first (start tuning immediately), then add the
     IsaacGym side for cross-sim comparison? **(recommended)**
  2. Log format — `.npz` (simple) vs parquet (better for 100s of rollouts + pandas)? **(rec: .npz per rollout)**
  3. Location — new `eval/` package vs script dir? **(rec: new `eval/` package)**
- **Scale (spec §4.2):** ≥100 rollouts per (command, simulator), varying command / terrain seed /
  initial state / mass perturbation; run in both IsaacGym (in-distribution) and MuJoCo (sim2sim, OOD).
- **Tuning decision tree (spec §4.4):** poor torque margin → tighten torque penalty, retrain;
  poor symmetry → raise symmetry reward weight or revisit `algo.config.use-symmetry`, retrain;
  large sim2sim gap → revisit DR (mass/friction/motor strength) or obs normalization, retrain.

**Recommended immediate action:** build MuJoCo logger + metrics + plotting with `.npz` per rollout in
a new `eval/` package, validate on a live rollout, then replicate the logger on the IsaacGym side.

---

## 8. Migration notes (server move)

- Code is safe on the fork (`jinhac174/Holosoma_statue`, branch `main`).
- **Not in git:** conda envs (rebuild), the ONNX model (copied manually to `~/repos/holosoma/model_0050000.onnx`).
- New server: `danielc174@165.132.142.207`. On the new box, model path is `~/repos/holosoma/model_0050000.onnx`
  (the old wandb symlink tree does not exist — point `--task.model-path` straight at the file).
- GitHub auth on a new box: generate a fresh SSH key (`ssh-keygen -t ed25519`), add the `.pub` to
  GitHub → Settings → SSH keys; or clone over HTTPS with a PAT.
