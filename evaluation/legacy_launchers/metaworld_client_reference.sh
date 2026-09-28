#!/bin/bash


# source examples/metaworld/.venv135/bin/activate

export MUJOCO_GL=egl

unset LEROBOT_HOME

# libero评估的命令
export HF_ENDPOINT=https://hf-mirror.com \
       HF_HOME="/mnt/workspace/tangzuojin.tzj/.server135/.cache/huggingface" \
       OPENPI_DATA_HOME=/mnt/workspace/tangzuojin.tzj/openpi_base \
       HF_LEROBOT_HOME=/mnt/workspace/tangzuojin.tzj/ \
       PYTHONPATH=$PYTHONPATH:/mnt/workspace/tangzuojin.tzj/physics_latent_world/third_party/libero \

# ## metaworld_mt50评估的命令 new evaluating 04/03

CUDA_VISIBLE_DEVICES=0 python examples/metaworld/eval_metaworld_policy_client_0409.py  --port 3333  --chunk_size 5   --model_name phy --cam_pos_x 0.72




# CUDA_VISIBLE_DEVICES=0   python examples/libero/main.py --args.port 9621 --args.task-suite-name "libero_10" --args.model_name glam_test  --args.replan_steps 5 