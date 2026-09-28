unset LEROBOT_HOME

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

export HF_ENDPOINT=https://hf-mirror.com \
       HF_HOME="/mnt/workspace/tangzuojin.tzj/.cache/huggingface" \
       OPENPI_DATA_HOME=/mnt/workspace/tangzuojin.tzj/openpi_base \
       HF_LEROBOT_HOME=/mnt/workspace/tangzuojin.tzj/


# ## metaworld_mt50评估的命令




# XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 CUDA_VISIBLE_DEVICES=3 uv run scripts/serve_policy.py --port=3333 policy:checkpoint --policy.config=phy_metaworld_full_finetune --policy.dir=/mnt/workspace/tangzuojin.tzj/physics_va_flow_checkpoints/phy_metaworld_full_finetune/phylam_gpu1_ah6_l1_loss_steps30001_0412_phy2_mix_11_larger_dataset_epoch19_step_58216/30000


# XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 CUDA_VISIBLE_DEVICES=3 uv run scripts/serve_policy.py --port=4333 policy:checkpoint --policy.config=phy_metaworld_full_finetune --policy.dir=/mnt/workspace/tangzuojin.tzj/physics_va_flow_checkpoints/phy_metaworld_full_finetune/phylam_gpu1_ah6_l1_loss_steps30001_0412_phy2_mix_11_larger_dataset_epoch19_step_58216/30000


# XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 CUDA_VISIBLE_DEVICES=0 uv run scripts/serve_policy.py --port=5333 policy:checkpoint --policy.config=phy_metaworld_full_finetune --policy.dir=/mnt/workspace/tangzuojin.tzj/physics_va_flow_checkpoints/phy_metaworld_full_finetune/phylam_gpu1_ah6_l1_loss_steps30001_0412_phy2_mix_11_larger_dataset_epoch19_step_58216/30000


# XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 CUDA_VISIBLE_DEVICES=1 uv run scripts/serve_policy.py --port=6333 policy:checkpoint --policy.config=phy_metaworld_full_finetune --policy.dir=/mnt/workspace/tangzuojin.tzj/physics_va_flow_checkpoints/phy_metaworld_full_finetune/phylam_mix_ablation_gpu8_ah6_l1_loss_steps30001_0505_phy2_mix_11_larger_dataset_32_train_fp32_mix_11_larger_dataset_saved_epoch_4_step_12256/30000


## 0506 final test的命令
# XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 CUDA_VISIBLE_DEVICES=0 uv run scripts/serve_policy.py --port=3333 policy:checkpoint --policy.config=phy_metaworld_full_finetune --policy.dir=/mnt/workspace/tangzuojin.tzj/physics_va_flow_checkpoints/phy_metaworld_full_finetune/phylam_mix_ablation_without_add_loss_and_rev_loss_gpu8_ah6_l1_loss_steps30001_0506_phy2_mix_11_larger_dataset_32_train_fp32_mix_11_larger_dataset_saved_epoch_4_step_12256/30000


# XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 CUDA_VISIBLE_DEVICES=1 uv run scripts/serve_policy.py --port=4333 policy:checkpoint --policy.config=phy_metaworld_full_finetune --policy.dir=/mnt/workspace/tangzuojin.tzj/physics_va_flow_checkpoints/phy_metaworld_full_finetune/phylam_mix_ablation_without_add_loss_gpu8_ah6_l1_loss_steps30001_0506_phy2_mix_11_larger_dataset_32_train_fp32_mix_11_larger_dataset_saved_epoch_4_step_12256/30000


XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 CUDA_VISIBLE_DEVICES=1 uv run scripts/serve_policy.py --port=5333 policy:checkpoint --policy.config=phy_metaworld_full_finetune --policy.dir=/mnt/workspace/tangzuojin.tzj/physics_va_flow_checkpoints/phy_metaworld_full_finetune/phylam_mix_ablation_without_rev_loss_gpu8_ah6_l1_loss_steps30001_0506_phy2_mix_11_larger_dataset_32_train_fp32_mix_11_larger_dataset_saved_epoch_4_step_12256/30000


# XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 CUDA_VISIBLE_DEVICES=0 uv run scripts/serve_policy.py --port=6333 policy:checkpoint --policy.config=phy_metaworld_full_finetune --policy.dir=/mnt/workspace/tangzuojin.tzj/physics_va_flow_checkpoints/phy_metaworld_full_finetune/phylam_ah5_infer5_mix_data_0505_train_random_latent_context/30000




## libero评估的命令

# XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 CUDA_VISIBLE_DEVICES=1 uv run scripts/serve_policy.py --port=9621 policy:checkpoint --policy.config=phy_libero_full_finetune --policy.dir=/mnt/workspace/tangzuojin.tzj/physics_va_flow_checkpoints/phy_libero_full_finetune/phylam_gpu1_ah21_l1_loss_steps30001_0412_phy2_mix_11_larger_dataset_epoch19_step_58216/30000


# XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 CUDA_VISIBLE_DEVICES=0 uv run scripts/serve_policy.py --port=8621 policy:checkpoint --policy.config=phy_libero_full_finetune --policy.dir=/mnt/workspace/tangzuojin.tzj/physics_va_flow_checkpoints/phy_libero_full_finetune/phylam_gpu1_ah21_l1_loss_steps30001_0412_phy2_mix_11_larger_dataset_epoch19_step_58216/30000


# XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 CUDA_VISIBLE_DEVICES=1 uv run scripts/serve_policy.py --port=9621 policy:checkpoint --policy.config=phy_libero_full_finetune --policy.dir=/mnt/workspace/tangzuojin.tzj/physics_va_flow_checkpoints/phy_libero_full_finetune/phylam_gpu1_ah21_l1_loss_steps30001_0412_phy2_mix_11_larger_dataset_epoch19_step_58216/30000

