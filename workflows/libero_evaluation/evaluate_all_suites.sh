#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GPU=0
PORT=8000
TRIALS=50
DRY_RUN=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu) GPU="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --trials) TRIALS="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help)
      echo "Usage: $0 [--gpu N] [--port N] [--trials N] [--dry-run]"
      exit 0
      ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

for suite in spatial object goal long; do
  command=(bash "$REPO_ROOT/workflows/libero_evaluation/evaluate_libero.sh" "$suite" \
    --gpu "$GPU" --port "$PORT" --trials "$TRIALS")
  (( DRY_RUN == 0 )) || command+=(--dry-run)
  "${command[@]}"
done
