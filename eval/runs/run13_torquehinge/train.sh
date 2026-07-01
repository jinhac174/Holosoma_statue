#!/bin/bash
# run13: redesigned hinge penalty_torque (relu(|tau|-0.8*limit)^2 on pre-clip demand),
# weight -25. Same recipe as run11/12 (headless, video off -> graphics_device_id=-1,
# DISPLAY unset, GPU 6 ONLY, 4096 envs, 50k iters, seed 1).
set -e
cd /home/jinhac174/repos/holosoma
source scripts/source_isaacgym_setup.sh >/dev/null 2>&1
env -u DISPLAY CUDA_VISIBLE_DEVICES=6 python -m holosoma.train_agent exp:statue-28dof-fast-sac \
  --training.headless True --logger.video.enabled False
echo "RUN13_TRAIN_DONE"
