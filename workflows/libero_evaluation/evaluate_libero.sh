#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SUITE="${1:-}"
if [[ -z "$SUITE" || "$SUITE" == "-h" || "$SUITE" == "--help" ]]; then
  echo "Usage: $0 {spatial|object|goal|long} [--gpu N] [--port N] [--trials N] [--host HOST] [--client-only] [--dry-run]"
  [[ -n "$SUITE" ]] && exit 0
  exit 2
fi
shift
exec bash "$REPO_ROOT/workflows/pi0_post_training/evaluate_libero.sh" "$SUITE" "$@"
