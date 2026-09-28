#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_ROOT="${ALAM_ENV_ROOT:-$REPO_ROOT/.venvs}"
[[ "$ENV_ROOT" == /* ]] || ENV_ROOT="$REPO_ROOT/$ENV_ROOT"
PUBLISH_PYTHON="${HF_PUBLISH_PYTHON:-$ENV_ROOT/publish/bin/python}"
STATUS_ROOT="$REPO_ROOT/.runtime/publishing"
STATUS_FILE="$STATUS_ROOT/upload_created_repositories.status"
EXECUTE=0
ACKNOWLEDGE_RIGHTS=0
RESUME=0
STATUS_TRACKING=0
CURRENT_STAGE="initializing"
ORG_ARGS=()

usage() {
  echo "Usage: $0 [--execute --acknowledge-rights] [--resume] [--org OWNER]"
  echo "Dry-run is the default. Execute mode audits, stages, uploads largest-first, and verifies every repository."
  echo "Use --resume only after a prior run produced verified ARCHIVE_MANIFEST.sha256 and DATASET_LAYOUT.json files."
}

write_status() {
  local status="$1"
  local stage="$2"
  local temporary
  mkdir -p "$STATUS_ROOT"
  temporary="$STATUS_FILE.tmp.$$"
  {
    printf 'status=%s\n' "$status"
    printf 'stage=%s\n' "$stage"
    printf 'pid=%s\n' "$$"
    printf 'updated_at=%s\n' "$(date '+%Y-%m-%d %H:%M:%S %Z')"
  } > "$temporary"
  mv -f -- "$temporary" "$STATUS_FILE"
}

record_exit() {
  local code="$1"
  if [[ "$STATUS_TRACKING" -eq 1 ]]; then
    if [[ "$code" -eq 0 ]]; then
      write_status "PASS" "completed"
    else
      write_status "FAILED" "$CURRENT_STAGE (exit=$code)"
    fi
  fi
}

clean_generated_source_files() {
  find "$REPO_ROOT" \
    \( -path "$REPO_ROOT/.cache" \
       -o -path "$REPO_ROOT/.runtime" \
       -o -path "$REPO_ROOT/.tools" \
       -o -path "$REPO_ROOT/.venvs" \
       -o -path "$REPO_ROOT/data" \
       -o -path "$REPO_ROOT/hf_staging" \
       -o -path "$REPO_ROOT/outputs" \
       -o -path "$REPO_ROOT/evaluation/checkpoints" \) -prune \
    -o -type d -name __pycache__ -exec rm -rf -- {} +
  find "$REPO_ROOT" \
    \( -path "$REPO_ROOT/.cache" \
       -o -path "$REPO_ROOT/.runtime" \
       -o -path "$REPO_ROOT/.tools" \
       -o -path "$REPO_ROOT/.venvs" \
       -o -path "$REPO_ROOT/data" \
       -o -path "$REPO_ROOT/hf_staging" \
       -o -path "$REPO_ROOT/outputs" \
       -o -path "$REPO_ROOT/evaluation/checkpoints" \) -prune \
    -o -type f \( -name '*.pyc' -o -name '*.pyo' -o -name '.DS_Store' \) -exec rm -f -- {} +
}

main() {
  # Keep the complete long-running workflow inside one function. Bash parses
  # the whole function before execution, so later edits to the source file
  # cannot corrupt a run that is already packaging or uploading.
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --execute)
        EXECUTE=1
        shift
        ;;
      --acknowledge-rights)
        ACKNOWLEDGE_RIGHTS=1
        shift
        ;;
      --resume)
        RESUME=1
        shift
        ;;
      --org)
        ORG_ARGS=(--org "$2")
        shift 2
        ;;
      -h|--help)
        usage
        return 0
        ;;
      *)
        echo "Unknown argument: $1" >&2
        usage >&2
        return 2
        ;;
    esac
  done

  if [[ ! -x "$PUBLISH_PYTHON" ]]; then
    echo "Missing publisher environment: $PUBLISH_PYTHON" >&2
    echo "Run: bash workflows/install/install_environments.sh --components publish" >&2
    return 1
  fi

  cd "$REPO_ROOT"
  clean_generated_source_files
  CURRENT_STAGE="base-audit"
  python3 workflows/audit/validate.py

  if [[ "$EXECUTE" -eq 0 ]]; then
    "$PUBLISH_PYTHON" workflows/publishing/prepare_dataset_archives.py \
      --artifact oxe_mix10_datasets
    "$PUBLISH_PYTHON" workflows/publishing/publish_huggingface.py \
      --artifact alam_pretrain \
      --artifact alam_plus_pi_libero \
      --artifact alam_plus_pi_metaworld_mt50 \
      --artifact oxe_mix10_datasets \
      --acknowledge-rights \
      "${ORG_ARGS[@]}"
    echo "Dry run only. Execute mode creates OXE tar shards, computes exact sizes, and uploads largest-first."
    return 0
  fi

  STATUS_TRACKING=1
  write_status "RUNNING" "$CURRENT_STAGE"
  trap 'record_exit "$?"' EXIT

  if [[ "$ACKNOWLEDGE_RIGHTS" -ne 1 ]]; then
    echo "Execute mode requires --acknowledge-rights after reviewing all OXE licenses." >&2
    return 2
  fi
  if [[ -z "${HF_TOKEN:-}" ]] && ! "$PUBLISH_PYTHON" -c \
    'from huggingface_hub import get_token; raise SystemExit(0 if get_token() else 1)'; then
    echo "No Hugging Face token found. Run 'hf auth login' or export HF_TOKEN." >&2
    return 2
  fi

  if [[ "$RESUME" -eq 1 ]]; then
    CURRENT_STAGE="resume-audit"
    write_status "RUNNING" "$CURRENT_STAGE"
    python3 workflows/audit/validate.py --hash-weights
    CURRENT_STAGE="reuse-verified-dataset-archives"
    write_status "RUNNING" "$CURRENT_STAGE"
    "$PUBLISH_PYTHON" workflows/publishing/prepare_dataset_archives.py \
      --artifact oxe_mix10_datasets --execute
  else
    CURRENT_STAGE="full-source-audit"
    write_status "RUNNING" "$CURRENT_STAGE"
    python3 workflows/audit/validate.py --hash-weights --source-code --source-models
    CURRENT_STAGE="build-and-verify-dataset-archives"
    write_status "RUNNING" "$CURRENT_STAGE"
    "$PUBLISH_PYTHON" workflows/publishing/prepare_dataset_archives.py \
      --artifact oxe_mix10_datasets --execute --verify-existing
  fi

  export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"
  CURRENT_STAGE="compute-upload-order"
  write_status "RUNNING" "$CURRENT_STAGE"
  local order_file="$STATUS_ROOT/upload_order.tsv"
  local order_temporary="$order_file.tmp.$$"
  "$PUBLISH_PYTHON" workflows/publishing/artifact_upload_order.py \
    --artifact oxe_mix10_datasets \
    --artifact alam_plus_pi_libero \
    --artifact alam_plus_pi_metaworld_mt50 \
    --artifact alam_pretrain \
    --format shell > "$order_temporary"
  mv -f -- "$order_temporary" "$order_file"
  while IFS=$'\t' read -r artifact bytes; do
    CURRENT_STAGE="upload:$artifact"
    write_status "RUNNING" "$CURRENT_STAGE"
    echo "Uploading $artifact ($bytes bytes)"
    "$PUBLISH_PYTHON" workflows/publishing/publish_huggingface.py \
      --artifact "$artifact" \
      --execute \
      --acknowledge-rights \
      "${ORG_ARGS[@]}"
    CURRENT_STAGE="verify:$artifact"
    write_status "RUNNING" "$CURRENT_STAGE"
    "$PUBLISH_PYTHON" workflows/publishing/verify_huggingface.py \
      --artifact "$artifact" \
      "${ORG_ARGS[@]}"
  done < "$order_file"

  CURRENT_STAGE="completed"
  echo "All four bulk Hugging Face repositories uploaded and verified: PASS"
}

main "$@"
