#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$REPO_ROOT/workflows/common/read_env.sh"
load_project_env "$REPO_ROOT/.env"
ENV_ROOT="${ALAM_ENV_ROOT:-$REPO_ROOT/.venvs}"
[[ "$ENV_ROOT" == /* ]] || ENV_ROOT="$REPO_ROOT/$ENV_ROOT"

BENCHMARK="${1:-}"
GPU_DEVICES="${2:-}"
if [[ "$BENCHMARK" != "metaworld" && "$BENCHMARK" != "libero" ]] || [[ -z "$GPU_DEVICES" ]]; then
  echo "Usage: $0 {metaworld|libero} GPU_IDS [EXP_NAME]  # example GPU_IDS=0,1" >&2
  exit 2
fi
if [[ ! "$GPU_DEVICES" =~ ^[0-9]+(,[0-9]+)+$ ]]; then
  echo "The full-parameter pi0 optimization smoke requires at least two comma-separated GPU ids (for example 0,1)." >&2
  exit 2
fi
IFS=',' read -r -a GPU_ID_LIST <<< "$GPU_DEVICES"
FSDP_DEVICES="${#GPU_ID_LIST[@]}"
declare -A SEEN_GPU_IDS=()
for gpu_id in "${GPU_ID_LIST[@]}"; do
  if [[ -n "${SEEN_GPU_IDS[$gpu_id]:-}" ]]; then
    echo "GPU ids must be distinct, found duplicate: $gpu_id" >&2
    exit 2
  fi
  SEEN_GPU_IDS[$gpu_id]=1
done
EXP_NAME="${3:-${BENCHMARK}_relative_smoke_$(date '+%Y%m%d_%H%M%S')_$$}"

case "$BENCHMARK" in
  metaworld)
    CONFIG_NAME="phy_metaworld_full_finetune"
    DATASET_REL="data/lerobot/metaworld_mt50"
    ALAM_REL="evaluation/checkpoints/alam/alam_pretrain_latent_action_tokenizer"
    POLICY_REL="evaluation/checkpoints/metaworld/alam_plus_pi_metaworld_mt50_step30000"
    ;;
  libero)
    CONFIG_NAME="phy_libero_full_finetune"
    DATASET_REL="data/lerobot/libero_real"
    ALAM_REL="evaluation/checkpoints/alam/alam_pretrain_latent_action_tokenizer"
    POLICY_REL="evaluation/checkpoints/libero/alam_plus_pi_libero_step30000"
    ;;
esac

PI0_PYTHON="$ENV_ROOT/pi0/bin/python"
[[ -x "$PI0_PYTHON" ]] || { echo "Missing pi0 environment: $PI0_PYTHON" >&2; exit 1; }
for path in "$DATASET_REL" "$ALAM_REL" "$POLICY_REL" "$POLICY_REL/params" "$POLICY_REL/assets"; do
  [[ -e "$REPO_ROOT/$path" ]] || { echo "Missing relative smoke asset: $path" >&2; exit 1; }
done

CHECKPOINT_ROOT="outputs/tests/post_training_smoke"
SAVED_REL="$CHECKPOINT_ROOT/$CONFIG_NAME/$EXP_NAME/0"
cd "$REPO_ROOT"
echo "pi0 post-training smoke: benchmark=$BENCHMARK gpus=$GPU_DEVICES fsdp_devices=$FSDP_DEVICES dataset=$DATASET_REL alam=$ALAM_REL base_params=$POLICY_REL/params"

CUDA_VISIBLE_DEVICES="$GPU_DEVICES" \
XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}" \
XLA_PYTHON_CLIENT_ALLOCATOR="${XLA_PYTHON_CLIENT_ALLOCATOR:-platform}" \
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
  "$PI0_PYTHON" workflows/pi0_post_training/train.py "$BENCHMARK" \
    --exp-name "$EXP_NAME" \
    --dataset-home data/lerobot \
    --alam-checkpoint "$ALAM_REL" \
    --assets-dir "$POLICY_REL/assets" \
    --base-params "$POLICY_REL/params" \
    --checkpoint-root "$CHECKPOINT_ROOT" \
    --steps 1 \
    --batch-size "$FSDP_DEVICES" \
    --workers 1 \
    --fsdp-devices "$FSDP_DEVICES" \
    --episode-limit 1 \
    --overwrite

"$PI0_PYTHON" workflows/pi0_post_training/tests/test_checkpoint.py \
  --benchmark "$BENCHMARK" \
  --checkpoint "$SAVED_REL" \
  --allow-train-state
echo "pi0 relative-path post-training smoke PASS: $SAVED_REL"
