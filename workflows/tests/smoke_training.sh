#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$REPO_ROOT/workflows/common/read_env.sh"
load_project_env "$REPO_ROOT/.env"
ENV_ROOT="${ALAM_ENV_ROOT:-$REPO_ROOT/.venvs}"
[[ "$ENV_ROOT" == /* ]] || ENV_ROOT="$REPO_ROOT/$ENV_ROOT"

GPU_DEVICES="${ALAM_SMOKE_GPUS:-0,1}"
COMPONENT=all
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu) GPU_DEVICES="$2"; shift 2 ;;
    --component) COMPONENT="$2"; shift 2 ;;
    -h|--help)
      echo "Usage: $0 [--gpu IDS] [--component all|alam|metaworld|libero]  # IDS example: 0,1"
      exit 0
      ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
case "$COMPONENT" in all|alam|metaworld|libero) ;; *) echo "Unknown component: $COMPONENT" >&2; exit 2 ;; esac

cd "$REPO_ROOT"
ALAM_GPU="${GPU_DEVICES%%,*}"
"$ENV_ROOT/alam/bin/python" workflows/alam_pretraining/test_datasets.py
"$ENV_ROOT/pi0/bin/python" workflows/pi0_post_training/tests/test_datasets.py

if [[ "$COMPONENT" == "all" || "$COMPONENT" == "alam" ]]; then
  CUDA_VISIBLE_DEVICES="$ALAM_GPU" "$ENV_ROOT/alam/bin/python" \
    workflows/alam_pretraining/smoke_train.py
fi
if [[ "$COMPONENT" == "all" || "$COMPONENT" == "metaworld" ]]; then
  bash workflows/pi0_post_training/smoke_train.sh metaworld "$GPU_DEVICES"
fi
if [[ "$COMPONENT" == "all" || "$COMPONENT" == "libero" ]]; then
  bash workflows/pi0_post_training/smoke_train.sh libero "$GPU_DEVICES"
fi
echo "Relative-path training smoke PASS: component=$COMPONENT gpus=$GPU_DEVICES"
