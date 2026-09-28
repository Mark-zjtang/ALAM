#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GPU=0
PORT=8000
TRIALS=50
DRY_RUN=0
MODEL_ARGS=()
OUTPUT_DIR=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --policy-checkpoint|--alam-checkpoint|--output-dir)
      [[ -n ${2:-} && $2 != --* ]] || { echo "$1 requires a path" >&2; exit 2; }
      if [[ $1 == --output-dir ]]; then OUTPUT_DIR="$2"; else MODEL_ARGS+=("$1" "$2"); fi
      shift 2 ;;
    --gpu) GPU="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --trials) TRIALS="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help)
      echo "Usage: $0 [--policy-checkpoint PATH] [--alam-checkpoint PATH] [--output-dir PATH] [--gpu N] [--port N] [--trials N] [--dry-run]"
      exit 0
      ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

for suite in spatial object goal long; do
  command=(bash "$REPO_ROOT/workflows/libero_evaluation/evaluate_libero.sh" "$suite" \
    --gpu "$GPU" --port "$PORT" --trials "$TRIALS")
  command+=("${MODEL_ARGS[@]}")
  [[ -z "$OUTPUT_DIR" ]] || command+=(--output-dir "$OUTPUT_DIR/$suite")
  (( DRY_RUN == 0 )) || command+=(--dry-run)
  "${command[@]}"
done
