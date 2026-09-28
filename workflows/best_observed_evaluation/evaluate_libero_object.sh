#!/usr/bin/env bash
# Best observed: 496/500, historical-client r6c; H14, replan10, original 14-step sampler.
set -Eeuo pipefail
P="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
R="$(cd "$P/.." && pwd)/alam_runtime_20260901"
[[ $# -eq 0 || ( $# -eq 1 && $1 == --dry-run ) ]] || { echo "usage: $0 [--dry-run]" >&2; exit 64; }
exec bash "$P/workflows/best_observed_evaluation/support/libero_runner.bash" object 1 19912 "$R/libero_python38_history135.sh" "$R/outputs/libero_four_suite_historical_client_r6c_20260902/libero_four_suite_summary.json" 496 "$@"
