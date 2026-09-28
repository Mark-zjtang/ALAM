#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$REPO_ROOT/workflows/common/read_env.sh"
load_project_env "$REPO_ROOT/.env"
ENV_ROOT="${ALAM_ENV_ROOT:-$REPO_ROOT/.venvs}"
[[ "$ENV_ROOT" == /* ]] || ENV_ROOT="$REPO_ROOT/$ENV_ROOT"
ALAM_PYTHON="$ENV_ROOT/alam/bin/python"

GPU=0
ARGUMENTS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu) GPU="$2"; shift 2 ;;
    -h|--help)
      echo "Usage: $0 [--gpu N] [--checkpoint PATH] [--calvin-root PATH] [--samples N] [--output PATH]"
      exit 0
      ;;
    *) ARGUMENTS+=("$1"); shift ;;
  esac
done
[[ -x "$ALAM_PYTHON" ]] || {
  echo "Missing ALAM environment. Run: bash workflows/alam_pretraining/install.sh" >&2
  exit 1
}

cd "$REPO_ROOT"
CUDA_VISIBLE_DEVICES="$GPU" "$ALAM_PYTHON" \
  workflows/alam_pretraining/evaluate_validation.py "${ARGUMENTS[@]}"
