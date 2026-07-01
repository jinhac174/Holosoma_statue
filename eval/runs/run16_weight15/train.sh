#!/bin/bash
# run16: penalty_torque hinge weight -10 -> -15 (torque<->symmetry knee), run14 base
# (hinge + dof_torque in critic). W&B scalars, video OFF, GPU 6 only.
set -e
cd /home/jinhac174/repos/holosoma
source scripts/source_isaacgym_setup.sh >/dev/null 2>&1
env -u DISPLAY CUDA_VISIBLE_DEVICES=6 python -m holosoma.train_agent \
  exp:statue-28dof-fast-sac logger:wandb \
  --training.headless True --logger.video.enabled False
echo "RUN16_TRAIN_DONE"
