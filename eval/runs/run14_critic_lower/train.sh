#!/bin/bash
# run14: hinge penalty_torque -10 (down from -25) + dof_torque in critic_obs + W&B SCALARS.
# Video DISABLED: --logger.video.enabled True leaked ~3.2 GB onto GPU 2 (Vulkan graphics
# ignores CUDA_VISIBLE_DEVICES), violating GPU-6-only. Scalars-only keeps graphics_device_id=-1.
set -e
cd /home/jinhac174/repos/holosoma
source scripts/source_isaacgym_setup.sh >/dev/null 2>&1
env -u DISPLAY CUDA_VISIBLE_DEVICES=6 python -m holosoma.train_agent \
  exp:statue-28dof-fast-sac logger:wandb \
  --training.headless True --logger.video.enabled False
echo "RUN14_TRAIN_DONE"
