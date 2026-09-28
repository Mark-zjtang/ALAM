#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$REPO_ROOT/workflows/common/read_env.sh"
load_project_env "$REPO_ROOT/.env"
ENV_ROOT="${ALAM_ENV_ROOT:-$REPO_ROOT/.venvs}"
[[ "$ENV_ROOT" == /* ]] || ENV_ROOT="$REPO_ROOT/$ENV_ROOT"
PUBLISH_PYTHON="$ENV_ROOT/publish/bin/python"
MODE=models
ORG_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --models-only) MODE=models; shift ;;
    --all) MODE=all; shift ;;
    --org) ORG_ARGS=(--org "$2"); shift 2 ;;
    -h|--help)
      echo "Usage: $0 [--models-only|--all] [--org OWNER]"
      exit 0
      ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ -x "$PUBLISH_PYTHON" ]] || {
  echo "Missing publisher environment. Run: bash workflows/install/install_environments.sh --components publish" >&2
  exit 1
}

artifacts=(alam_pretrain alam_plus_pi_metaworld_mt50 alam_plus_pi_libero)
if [[ "$MODE" == "all" ]]; then
  artifacts+=(calvin_task_abc_d oxe_mix10_datasets metaworld_mt50_lerobot libero_real_lerobot)
fi
arguments=()
for artifact in "${artifacts[@]}"; do
  arguments+=(--artifact "$artifact")
done

cd "$REPO_ROOT"
"$PUBLISH_PYTHON" workflows/publishing/download_huggingface.py \
  "${arguments[@]}" "${ORG_ARGS[@]}"
python3 workflows/audit/validate.py --require-weights
if [[ "$MODE" == "all" ]]; then
  "$ENV_ROOT/alam/bin/python" workflows/alam_pretraining/test_datasets.py
  "$ENV_ROOT/pi0/bin/python" workflows/pi0_post_training/tests/test_datasets.py
fi
echo "Hugging Face release download PASS: mode=$MODE"
