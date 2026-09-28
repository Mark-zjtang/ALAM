#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$REPO_ROOT/workflows/common/read_env.sh"
load_project_env "$REPO_ROOT/.env"
ENV_ROOT="${ALAM_ENV_ROOT:-$REPO_ROOT/.venvs}"
[[ "$ENV_ROOT" == /* ]] || ENV_ROOT="$REPO_ROOT/$ENV_ROOT"
DEVICE="${1:-cpu}"
if [[ "$DEVICE" != "cpu" && "$DEVICE" != "cuda" ]]; then
  echo "Usage: $0 [cpu|cuda] [test_checkpoint.py arguments]" >&2
  exit 2
fi
shift || true
exec "$ENV_ROOT/alam/bin/python" "$REPO_ROOT/workflows/alam_pretraining/test_checkpoint.py" \
  --device "$DEVICE" "$@"
