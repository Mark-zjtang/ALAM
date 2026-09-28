#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$REPO_ROOT/workflows/common/read_env.sh"
load_project_env "$REPO_ROOT/.env"
ENV_ROOT="${ALAM_ENV_ROOT:-$REPO_ROOT/.venvs}"
[[ "$ENV_ROOT" == /* ]] || ENV_ROOT="$REPO_ROOT/$ENV_ROOT"
COMPONENTS=all
TEST_MODE=cpu
HF_ORG=""
GPU=0
DOWNLOAD_MODE=1
while [[ $# -gt 0 ]]; do
  case "$1" in
    --components) COMPONENTS="$2"; shift 2 ;;
    --test) TEST_MODE="$2"; shift 2 ;;
    --org) HF_ORG="$2"; shift 2 ;;
    --gpu) GPU="$2"; shift 2 ;;
    --skip-download) DOWNLOAD_MODE=0; shift ;;
    -h|--help)
      echo "Usage: $0 [--components LIST] [--org HF_ORG] [--skip-download] [--test cpu|gpu|e2e] [--gpu N]"
      exit 0
      ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

cd "$REPO_ROOT"
bash workflows/install/install_environments.sh --components "$COMPONENTS"
if [[ "$DOWNLOAD_MODE" -eq 1 ]]; then
  DOWNLOAD_CMD=("$ENV_ROOT/publish/bin/python" workflows/publishing/download_huggingface.py)
  [[ -z "$HF_ORG" ]] || DOWNLOAD_CMD+=(--org "$HF_ORG")
  "${DOWNLOAD_CMD[@]}"
fi
GPU_ID="$GPU" bash workflows/tests/test_all.sh "$TEST_MODE"
