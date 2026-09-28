#!/usr/bin/env bash
# Publish only the three ALAM model repositories. Never upload datasets.
set -Eeuo pipefail
export PYTHONDONTWRITEBYTECODE=1

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${HF_PUBLISH_PYTHON:-$ROOT/.venvs/publish/bin/python}"
mode="${1:---dry-run}"
if [[ $mode == --dry-run && ! -x $PYTHON ]]; then
  PYTHON="$(command -v python3)"
fi
[[ -x "$PYTHON" ]] || {
  echo 'Install the publish environment first: bash workflows/install/install_environments.sh --components publish' >&2
  exit 2
}

case "$mode" in
  --dry-run|--verify-only) [[ $# -eq 1 || $# -eq 0 ]] ;;
  --execute) [[ $# -eq 2 && $2 == --acknowledge-rights ]] || {
    echo 'Usage: upload_models.sh [--dry-run|--verify-only|--execute --acknowledge-rights]' >&2
    exit 64
  } ;;
  *) echo 'Usage: upload_models.sh [--dry-run|--verify-only|--execute --acknowledge-rights]' >&2; exit 64 ;;
esac

artifacts=(alam_pretrain alam_plus_pi_metaworld_mt50 alam_plus_pi_libero)
args=()
for artifact in "${artifacts[@]}"; do
  args+=(--artifact "$artifact")
done

cd "$ROOT"
if [[ $mode == --dry-run ]]; then
  exec "$PYTHON" workflows/publishing/publish_huggingface.py "${args[@]}"
fi

echo 'Verifying all 46 local model files against evaluation/WEIGHTS_MANIFEST.sha256 ...'
(cd evaluation && sha256sum --quiet -c WEIGHTS_MANIFEST.sha256)

if [[ $mode == --execute ]]; then
  if [[ -z ${HF_TOKEN:-} ]]; then
    [[ -t 0 ]] || { echo 'Run from a terminal to enter the Hugging Face write token.' >&2; exit 2; }
    read -r -s -p 'Hugging Face write token: ' HF_TOKEN
    printf '\n' >&2
    [[ -n $HF_TOKEN ]] || { echo 'Empty token.' >&2; exit 2; }
    export HF_TOKEN
    trap 'unset HF_TOKEN' EXIT
  fi
  "$PYTHON" workflows/publishing/publish_huggingface.py \
    "${args[@]}" --execute --acknowledge-rights
fi

for artifact in "${artifacts[@]}"; do
  "$PYTHON" workflows/publishing/verify_huggingface.py --artifact "$artifact"
done
echo 'Three model repositories verified. No dataset was uploaded.'
