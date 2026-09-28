#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_ROOT="${ALAM_ENV_ROOT:-$REPO_ROOT/.venvs}"
[[ "$ENV_ROOT" == /* ]] || ENV_ROOT="$REPO_ROOT/$ENV_ROOT"
PUBLISH_PYTHON="${HF_PUBLISH_PYTHON:-$ENV_ROOT/publish/bin/python}"
export PYTHONDONTWRITEBYTECODE=1
EXECUTE=0
ACKNOWLEDGE_RIGHTS=0
ORG_ARGS=()

usage() {
  echo "Usage: $0 [--execute --acknowledge-rights] [--org OWNER]"
  echo "Creates if needed, uploads, and verifies only the MetaWorld MT50 and LIBERO LeRobot fine-tuning datasets."
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --execute) EXECUTE=1; shift ;;
    --acknowledge-rights) ACKNOWLEDGE_RIGHTS=1; shift ;;
    --org) ORG_ARGS=(--org "$2"); shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ ! -x "$PUBLISH_PYTHON" ]]; then
  echo "Missing publisher environment: $PUBLISH_PYTHON" >&2
  echo "Run: bash workflows/install/install_environments.sh --components publish" >&2
  exit 1
fi

cd "$REPO_ROOT"
echo "[downstream-data] validate source metadata and repository mapping"
python3 -B workflows/audit/validate.py --source-datasets

PUBLISH_ARGS=(
  --artifact metaworld_mt50_lerobot
  --artifact libero_real_lerobot
  "${ORG_ARGS[@]}"
)
if [[ "$ACKNOWLEDGE_RIGHTS" -eq 1 ]]; then
  PUBLISH_ARGS+=(--acknowledge-rights)
fi

if [[ "$EXECUTE" -eq 0 ]]; then
  "$PUBLISH_PYTHON" workflows/publishing/publish_huggingface.py "${PUBLISH_ARGS[@]}"
  echo "Dry run only. Execute mode creates both named repositories if needed, then uploads them largest-first."
  exit 0
fi

if [[ "$ACKNOWLEDGE_RIGHTS" -ne 1 ]]; then
  echo "Execute mode requires --acknowledge-rights after reviewing both dataset licenses." >&2
  exit 2
fi
if [[ -z "${HF_TOKEN:-}" ]] && ! "$PUBLISH_PYTHON" -c \
  'from huggingface_hub import get_token; raise SystemExit(0 if get_token() else 1)'; then
  echo "No Hugging Face token found. Run 'hf auth login' or export HF_TOKEN." >&2
  exit 2
fi

echo "[downstream-data] compute exact upload sizes and upload largest-first"
while IFS=$'\t' read -r artifact bytes; do
  echo "[downstream-data] uploading $artifact ($bytes bytes)"
  "$PUBLISH_PYTHON" workflows/publishing/publish_huggingface.py \
    --artifact "$artifact" --execute --acknowledge-rights "${ORG_ARGS[@]}"
  echo "[downstream-data] verify remote file set, sizes, and hashes for $artifact"
  "$PUBLISH_PYTHON" workflows/publishing/verify_huggingface.py \
    --artifact "$artifact" "${ORG_ARGS[@]}"
done < <(
  "$PUBLISH_PYTHON" workflows/publishing/artifact_upload_order.py \
    --artifact metaworld_mt50_lerobot \
    --artifact libero_real_lerobot \
    --format shell
)

echo "Both downstream LeRobot dataset repositories uploaded and verified: PASS"
