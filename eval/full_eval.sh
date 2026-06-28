#!/bin/bash
# Standardized FULL evaluation for one run: gait (MuJoCo+IsaacGym flat, 100/cmd) +
# robustness (push 100N torso, rough = Holosoma terrain term) + report + purge raw NPZ.
# Each env-step runs in its OWN `bash -c` so conda env switching is clean
# (sourcing hsgym after hsmujoco in the same shell does NOT switch — that was a bug).
# Usage: full_eval.sh <run_dir> <onnx> <pt> [gpu=4] [n_per_cmd=100]
set -e
# GPU is ALWAYS 6 (our only allocated device). Do not change — other GPUs belong to other users.
RUN_DIR="$1"; ONNX="$2"; PT="$3"; GPU=6; N="${5:-100}"
cd /home/jinhac174/repos/holosoma
MJ='source scripts/source_mujoco_setup.sh >/dev/null 2>&1 &&'
IG='source scripts/source_isaacgym_setup.sh >/dev/null 2>&1 &&'

bash -c "$MJ python -m holosoma.eval.run_suite --model-path '$ONNX' --out-dir '$RUN_DIR/mujoco' --n-per-command $N --duration 12 --workers 16"
bash -c "$MJ python -m holosoma.eval.run_suite --model-path '$ONNX' --out-dir '$RUN_DIR/push'   --n-per-command 50 --duration 12 --workers 16 --push 100"
bash -c "$IG CUDA_VISIBLE_DEVICES=$GPU python -m holosoma.eval.ig_suite --checkpoint '$PT' --out-dir '$RUN_DIR/isaacgym' --n-per-command $N --duration 12 --terrain flat"
bash -c "$IG CUDA_VISIBLE_DEVICES=$GPU python -m holosoma.eval.ig_suite --checkpoint '$PT' --out-dir '$RUN_DIR/rough'    --n-per-command 50 --duration 12 --terrain rough"
bash -c "$MJ python -m holosoma.eval.report --run-dir '$RUN_DIR'"
rm -rf "$RUN_DIR/mujoco" "$RUN_DIR/isaacgym" "$RUN_DIR/push" "$RUN_DIR/rough"
echo "FULL_EVAL_DONE $RUN_DIR"
