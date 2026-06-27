#!/bin/bash
# Standardized FULL evaluation for one run: gait (MuJoCo+IsaacGym flat, 100/cmd) +
# robustness (push 100N torso, rough = Holosoma terrain term) + report + purge raw NPZ.
# Usage: full_eval.sh <run_dir> <onnx> <pt> [gpu=4] [n_per_cmd=100]
set -e
RUN_DIR="$1"; ONNX="$2"; PT="$3"; GPU="${4:-4}"; N="${5:-100}"
cd /home/jinhac174/repos/holosoma

source scripts/source_mujoco_setup.sh >/dev/null 2>&1
python -m holosoma.eval.run_suite --model-path "$ONNX" --out-dir "$RUN_DIR/mujoco" \
    --n-per-command "$N" --duration 12 --workers 16
python -m holosoma.eval.run_suite --model-path "$ONNX" --out-dir "$RUN_DIR/push" \
    --n-per-command 50 --duration 12 --workers 16 --push 100

source scripts/source_isaacgym_setup.sh >/dev/null 2>&1
CUDA_VISIBLE_DEVICES="$GPU" python -m holosoma.eval.ig_suite --checkpoint "$PT" \
    --out-dir "$RUN_DIR/isaacgym" --n-per-command "$N" --duration 12 --terrain flat
CUDA_VISIBLE_DEVICES="$GPU" python -m holosoma.eval.ig_suite --checkpoint "$PT" \
    --out-dir "$RUN_DIR/rough" --n-per-command 50 --duration 12 --terrain rough

source scripts/source_mujoco_setup.sh >/dev/null 2>&1
python -m holosoma.eval.report --run-dir "$RUN_DIR"
# keep scorecard/csvs/plots; drop bulky raw rollouts to protect disk
rm -rf "$RUN_DIR/mujoco" "$RUN_DIR/isaacgym" "$RUN_DIR/push" "$RUN_DIR/rough"
echo "FULL_EVAL_DONE $RUN_DIR"
