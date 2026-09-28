#!/usr/bin/env bash
# Best observed: 475/500, independent r8; H18, replan12, original 18-step sampler.
set -Eeuo pipefail
P="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
R="$(cd "$P/.." && pwd)/alam_runtime_20260901"
[[ $# -eq 0 || ( $# -eq 1 && $1 == --dry-run ) ]] || { echo "usage: $0 [--dry-run]" >&2; exit 64; }
exec bash "$P/workflows/best_observed_evaluation/support/libero_runner.bash" long 3 19914 "$R/libero_python38_history146.sh" "$R/outputs/libero_four_suite_independent_r8_20260903/libero_four_suite_independent_summary.json" 475 "$@"
