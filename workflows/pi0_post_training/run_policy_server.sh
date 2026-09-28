#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$REPO_ROOT/workflows/common/read_env.sh"
load_project_env "$REPO_ROOT/.env"
ENV_ROOT="${ALAM_ENV_ROOT:-$REPO_ROOT/.venvs}"
if [[ "$ENV_ROOT" != /* ]]; then
  ENV_ROOT="$REPO_ROOT/$ENV_ROOT"
fi
PI0_PYTHON="${PI0_PYTHON:-$ENV_ROOT/pi0/bin/python}"

if [[ ! -x "$PI0_PYTHON" ]]; then
  echo "Missing pi0 environment: $PI0_PYTHON" >&2
  echo "Run: bash workflows/install/install_environments.sh --components pi0" >&2
  exit 1
fi

cd "$REPO_ROOT"
exec "$PI0_PYTHON" workflows/pi0_post_training/serve_policy.py "$@"
