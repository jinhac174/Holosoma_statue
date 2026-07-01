#!/bin/bash
# Wait for run13 training to produce the 50k checkpoint, then auto-run full_eval (GPU 6).
cd /home/jinhac174/repos/holosoma
LOG=logs/hv-statue-manager/20260630_034350-statue_28dof_fast_sac_manager-locomotion
RUN=eval/runs/run13_torquehinge
for i in $(seq 1 360); do                      # up to 6h
  if [ -f "$LOG/model_0050000.onnx" ] && [ -f "$LOG/model_0050000.pt" ]; then
    sleep 45                                   # let checkpoint finish writing
    echo "RUN13_TRAIN_COMPLETE -> starting full eval"
    bash eval/full_eval.sh "$RUN" "$LOG/model_0050000.onnx" "$LOG/model_0050000.pt"
    exit 0
  fi
  if ! pgrep -f 'train_agent' | grep -q .; then # training proc gone w/o checkpoint
    echo "RUN13_TRAIN_DIED_NO_CHECKPOINT"; exit 1
  fi
  sleep 60
done
echo "RUN13_WAIT_TIMEOUT"; exit 2
