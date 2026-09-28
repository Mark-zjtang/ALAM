#!/bin/bash

# ---------- 激活 uv 环境 ----------
source examples/libero/.venv135/bin/activate

export CUDA_VISIBLE_DEVICES=0

PORT=8621

infer_ah = 14

SUITE="libero_goal"

Date=0506_last

export MUJOCO_GL=egl
unset LEROBOT_HOME

export HF_ENDPOINT=https://hf-mirror.com \
       HF_HOME="/mnt/workspace/tangzuojin.tzj/.server48/.cache/huggingface" \
       OPENPI_DATA_HOME=/mnt/workspace/tangzuojin.tzj/openpi_base \
       HF_LEROBOT_HOME=/mnt/workspace/tangzuojin.tzj/ \
       PYTHONPATH=$PYTHONPATH:/mnt/workspace/tangzuojin.tzj/physics_latent_world/third_party/libero

MODEL_NAME="phylam_gpu1_0506_ah21_l1_loss_steps30001_0412_phy2_mix_11_larger_dataset_epoch19_step_58216"



LOG_DIR="/mnt/workspace/tangzuojin.tzj/physics_latent_world/checkpoints/pi0_libero_low_mem_finetune"
LOG_FILE="${LOG_DIR}/libero_replan_${SUITE}_phylam_train_20_infer_${infer_ah}_eval_30k_${Date}.log"

mkdir -p ${LOG_DIR}

# ---------- 日志头 ----------
echo "==================================================" | tee -a ${LOG_FILE}
echo "$(date '+%F %T') | Start LIBERO libero_10 replan sweep (replan 6~18, all parallel)" | tee -a ${LOG_FILE}
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}" | tee -a ${LOG_FILE}
echo "PORT: ${PORT}" | tee -a ${LOG_FILE}
echo "Python: $(which python)" | tee -a ${LOG_FILE}
echo "Log dir: ${LOG_DIR}" | tee -a ${LOG_FILE}
echo "==================================================" | tee -a ${LOG_FILE}

JOB_COUNTER=0

run_job() {
  local replan_steps=$1
  local task_log="${LOG_DIR}/replan${replan_steps}_${SUITE}_phylam_train_20_infer_${infer_ah}_eval_30k_${Date}.log"
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
for replan_steps in $(seq 4 14); do
  run_job ${replan_steps}
done

# ---------- 等待所有任务完成 ----------
echo "$(date '+%F %T') | All ${JOB_COUNTER} jobs launched, waiting ..." | tee -a ${LOG_FILE}
wait

echo "==================================================" | tee -a ${LOG_FILE}
echo "$(date '+%F %T') | All jobs completed." | tee -a ${LOG_FILE}
echo "==================================================" | tee -a ${LOG_FILE}
