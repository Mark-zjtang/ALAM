#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
METAWORLD_GPU=0
LIBERO_GPU=1
OUTPUT_ROOT=outputs/reproduction
while [[ $# -gt 0 ]]; do
  case "$1" in
    --metaworld-gpu) METAWORLD_GPU="$2"; shift 2 ;;
    --libero-gpu) LIBERO_GPU="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    -h|--help)
      echo "Usage: $0 [--metaworld-gpu N] [--libero-gpu N] [--output-root PATH]"
      exit 0
      ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
cd "$REPO_ROOT"
resolved_output="$OUTPUT_ROOT"
[[ "$resolved_output" == /* ]] || resolved_output="$REPO_ROOT/$resolved_output"
mkdir -p "$REPO_ROOT/.runtime"
exec 9>"$REPO_ROOT/.runtime/local_trained_policy_eval.serial.lock"
if ! flock -n 9; then
  echo "Another formal evaluation holds the serial lock." >&2
  exit 1
fi
if [[ -e "$resolved_output" || -L "$resolved_output" ]]; then
  echo "Refusing to reuse an evaluation output directory: $resolved_output" >&2
  exit 2
fi
mkdir -p "$resolved_output/launcher_logs"

stage_number=0
run_stage() {
  local stage="$1"
  shift
  stage_number=$((stage_number + 1))
  printf '%d\t%s\tSTART\t%s\n' "$stage_number" "$stage" "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" >>"$resolved_output/stage_order.tsv"
  if "$@" >"$resolved_output/launcher_logs/$stage.log" 2>&1; then
    printf '%d\t%s\tCOMPLETE\t%s\n' "$stage_number" "$stage" "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" >>"$resolved_output/stage_order.tsv"
  else
    local status=$?
    printf '%d\t%s\tFAILED(%d)\t%s\n' "$stage_number" "$stage" "$status" "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" >>"$resolved_output/stage_order.tsv"
    echo "Evaluation stage $stage failed; inspect $resolved_output/launcher_logs/$stage.log" >&2
    return "$status"
  fi
}

run_stage metaworld env ALAM_OUTPUT_ROOT="$OUTPUT_ROOT" XLA_PYTHON_CLIENT_MEM_FRACTION=0.65 \
  bash workflows/metaworld_evaluation/evaluate_mt50.sh \
    --gpu "$METAWORLD_GPU" --port 8200 --camera-x 0.70

for suite in spatial object goal long; do
  run_stage "libero_$suite" env ALAM_OUTPUT_ROOT="$OUTPUT_ROOT" XLA_PYTHON_CLIENT_MEM_FRACTION=0.65 \
    bash workflows/libero_evaluation/evaluate_libero.sh "$suite" \
      --gpu "$LIBERO_GPU" --port 8301 --trials 50
done

python3 workflows/tests/verify_live_reproduction.py all \
  --output-root "$OUTPUT_ROOT" \
  --summary "$OUTPUT_ROOT/reproduction_summary.json"
echo "Full trained-model reproduction PASS: $resolved_output/reproduction_summary.json"
