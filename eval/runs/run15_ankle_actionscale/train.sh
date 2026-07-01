#!/bin/bash
# run15: ankle_roll kp 70->50 + action_scale 0.25->0.20 (physical torque cap), on top of
# run14 (hinge -10 + dof_torque in critic). W&B scalars, video OFF (GPU-6 only). 
set -e
cd /home/jinhac174/repos/holosoma
source scripts/source_isaacgym_setup.sh >/dev/null 2>&1
env -u DISPLAY CUDA_VISIBLE_DEVICES=6 python -m holosoma.train_agent \
  exp:statue-28dof-fast-sac logger:wandb \
  --training.headless True --logger.video.enabled False
echo "RUN15_TRAIN_DONE"
