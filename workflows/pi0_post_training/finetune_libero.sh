#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$REPO_ROOT/workflows/common/read_env.sh"
load_project_env "$REPO_ROOT/.env"
ENV_ROOT="${ALAM_ENV_ROOT:-$REPO_ROOT/.venvs}"
[[ "$ENV_ROOT" == /* ]] || ENV_ROOT="$REPO_ROOT/$ENV_ROOT"
EXP_NAME="${1:-}"
if [[ -z "$EXP_NAME" || "$EXP_NAME" == "-h" || "$EXP_NAME" == "--help" ]]; then
  echo "Usage: $0 EXP_NAME [training arguments]"
  echo "Example: CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 $0 my_libero_run"
  [[ -n "$EXP_NAME" ]] && exit 0
  exit 2
fi
shift
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
[[ -x "$ENV_ROOT/pi0/bin/python" ]] || {
  echo "Missing pi0 environment. Run: bash workflows/pi0_post_training/install.sh" >&2
  exit 1
}
exec "$ENV_ROOT/pi0/bin/python" "$REPO_ROOT/workflows/pi0_post_training/train.py" \
  libero --exp-name "$EXP_NAME" "$@"
