#!/bin/bash

# ---------- 激活 uv 环境 ----------
source examples/libero/.venv135/bin/activate

export CUDA_VISIBLE_DEVICES=0

PORT=6621

export MUJOCO_GL=egl
unset LEROBOT_HOME

export HF_ENDPOINT=https://hf-mirror.com \
       HF_HOME="/mnt/workspace/tangzuojin.tzj/.server48/.cache/huggingface" \
       OPENPI_DATA_HOME=/mnt/workspace/tangzuojin.tzj/openpi_base \
       HF_LEROBOT_HOME=/mnt/workspace/tangzuojin.tzj/ \
       PYTHONPATH=$PYTHONPATH:/mnt/workspace/tangzuojin.tzj/physics_latent_world/third_party/libero

MODEL_NAME="phylam_0504_infer_14_gpu8_ah21_l1_loss_steps30001_0418_phy2_mix_11_larger_dataset_32_train_fp32_mix_11_larger_dataset_continue_from_epoch_24_step_73536_add_saved_epoch_16_step_49024"

SUITE="libero_object"

LOG_DIR="/mnt/workspace/tangzuojin.tzj/physics_latent_world/checkpoints/pi0_libero_low_mem_finetune"
LOG_FILE="${LOG_DIR}/libero_replan_libero_object_phylam_train_20_infer_14_eval_30k_0504.log"

mkdir -p ${LOG_DIR}

# ---------- 日志头 ----------
echo "==================================================" | tee -a ${LOG_FILE}
echo "$(date '+%F %T') | Start LIBERO libero_object replan sweep (replan 6~18, all parallel)" | tee -a ${LOG_FILE}
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}" | tee -a ${LOG_FILE}
echo "PORT: ${PORT}" | tee -a ${LOG_FILE}
echo "Python: $(which python)" | tee -a ${LOG_FILE}
echo "Log dir: ${LOG_DIR}" | tee -a ${LOG_FILE}
echo "==================================================" | tee -a ${LOG_FILE}

JOB_COUNTER=0

run_job() {
  local replan_steps=$1
  local task_log="${LOG_DIR}/replan${replan_steps}_${SUITE}_phylam_train_20_infer_14_eval_30k_0504.log"
  JOB_COUNTER=$(( JOB_COUNTER + 1 ))

  echo "$(date '+%F %T') | Launching suite=${SUITE}, replan_steps=${replan_steps}, PORT=${PORT}" \
    | tee -a ${LOG_FILE}

  (
    source examples/libero/.venv135/bin/activate
    python examples/libero/main.py \
      --args.port ${PORT} \
      --args.task-suite-name "${SUITE}" \
      --args.model_name "${MODEL_NAME}" \
      --args.replan_steps ${replan_steps} \
      > >(tee -a "${task_log}" | tee -a ${LOG_FILE}) \
      2>&1
    echo "$(date '+%F %T') | Done suite=${SUITE}, replan_steps=${replan_steps}" \
      | tee -a ${LOG_FILE}
  ) &
}

# ---------- 全部同时启动，replan_steps 6~18 ----------
for replan_steps in $(seq 6 14); do
  run_job ${replan_steps}
done

# ---------- 等待所有任务完成 ----------
echo "$(date '+%F %T') | All ${JOB_COUNTER} jobs launched, waiting ..." | tee -a ${LOG_FILE}
wait

echo "==================================================" | tee -a ${LOG_FILE}
echo "$(date '+%F %T') | All jobs completed." | tee -a ${LOG_FILE}
echo "==================================================" | tee -a ${LOG_FILE}
