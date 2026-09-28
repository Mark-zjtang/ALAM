#!/usr/bin/env bash
# Best observed: 495/500, independent r8; H14, replan5, original 14-step sampler.
set -Eeuo pipefail
P="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
R="$(cd "$P/.." && pwd)/alam_runtime_20260901"
[[ $# -eq 0 || ( $# -eq 1 && $1 == --dry-run ) ]] || { echo "usage: $0 [--dry-run]" >&2; exit 64; }
exec bash "$P/workflows/best_observed_evaluation/support/libero_runner.bash" spatial 0 19911 "$R/libero_python38_history135.sh" "$R/outputs/libero_four_suite_independent_r8_20260903/libero_four_suite_independent_summary.json" 495 "$@"
