#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$REPO_ROOT/workflows/common/read_env.sh"
load_project_env "$REPO_ROOT/.env"
ENV_ROOT="${ALAM_ENV_ROOT:-$REPO_ROOT/.venvs}"
[[ "$ENV_ROOT" == /* ]] || ENV_ROOT="$REPO_ROOT/$ENV_ROOT"
ACCELERATE="$ENV_ROOT/alam/bin/accelerate"
[[ -x "$ACCELERATE" ]] || {
  echo "Missing ALAM environment. Run: bash workflows/alam_pretraining/install.sh" >&2
  exit 1
}

if [[ "${1:-}" == "--dry-run" ]]; then
  shift
  exec "$ENV_ROOT/alam/bin/python" "$REPO_ROOT/workflows/alam_pretraining/train.py" \
    --config "$REPO_ROOT/configs/lam/alam_pretrain.yaml" --dry-run "$@"
fi

NUM_MACHINES="${ALAM_NUM_MACHINES:-1}"
if [[ -n "${ALAM_NUM_PROCESSES:-}" ]]; then
  NUM_PROCESSES="$ALAM_NUM_PROCESSES"
elif command -v nvidia-smi >/dev/null 2>&1; then
  NUM_PROCESSES="$(nvidia-smi --list-gpus | wc -l)"
else
  echo "Set ALAM_NUM_PROCESSES to the total GPU process count." >&2
  exit 1
fi

LAUNCH=("$ACCELERATE" launch --mixed_precision no \
  --num_machines "$NUM_MACHINES" --num_processes "$NUM_PROCESSES")
if (( NUM_PROCESSES > 1 )); then
  LAUNCH+=(--multi_gpu)
fi
if (( NUM_MACHINES > 1 )); then
  : "${MACHINE_RANK:?MACHINE_RANK is required for multi-node pretraining}"
  : "${MAIN_PROCESS_IP:?MAIN_PROCESS_IP is required for multi-node pretraining}"
  : "${MAIN_PROCESS_PORT:?MAIN_PROCESS_PORT is required for multi-node pretraining}"
  LAUNCH+=(--machine_rank "$MACHINE_RANK" --main_process_ip "$MAIN_PROCESS_IP" \
    --main_process_port "$MAIN_PROCESS_PORT")
fi

cd "$REPO_ROOT"
exec "${LAUNCH[@]}" workflows/alam_pretraining/train.py \
  --config configs/lam/alam_pretrain.yaml "$@"
