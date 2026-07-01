#!/bin/bash
cd /home/jinhac174/repos/holosoma
LOG=logs/hv-statue-manager/20260630_092545-statue_28dof_fast_sac_manager-locomotion
RUN=eval/runs/run14_critic_lower
for i in $(seq 1 360); do
  if [ -f "$LOG/model_0050000.onnx" ] && [ -f "$LOG/model_0050000.pt" ]; then
    sleep 45
    echo "RUN14_TRAIN_COMPLETE -> full eval"
    bash eval/full_eval.sh "$RUN" "$LOG/model_0050000.onnx" "$LOG/model_0050000.pt"
    exit 0
  fi
  if ! pgrep -f 'train_agent' | grep -q .; then echo "RUN14_TRAIN_DIED_NO_CHECKPOINT"; exit 1; fi
  sleep 60
done
echo "RUN14_WAIT_TIMEOUT"; exit 2
