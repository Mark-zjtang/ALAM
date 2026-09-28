#!/usr/bin/env bash
set -euo pipefail
ulimit -c 0 2>/dev/null || true
export PYTHONDONTWRITEBYTECODE=1

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$REPO_ROOT/workflows/common/read_env.sh"
load_project_env "$REPO_ROOT/.env"
ENV_ROOT="${ALAM_ENV_ROOT:-$REPO_ROOT/.venvs}"
[[ "$ENV_ROOT" == /* ]] || ENV_ROOT="$REPO_ROOT/$ENV_ROOT"
MODE="${1:-cpu}"
GPU_ID="${GPU_ID:-0}"
EGL_GPU_ID="${ALAM_EGL_GPU:-0}"
MIN_FREE_MIB="${ALAM_TEST_MIN_FREE_MIB:-60000}"
cd "$REPO_ROOT"

require_executable() {
  [[ -x "$1" ]] || { echo "Missing executable: $1" >&2; exit 1; }
}

check_gpu_capacity() {
  command -v nvidia-smi >/dev/null 2>&1 || { echo "nvidia-smi is required" >&2; exit 1; }
  local free_mib
  free_mib="$(nvidia-smi -i "$GPU_ID" --query-gpu=memory.free --format=csv,noheader,nounits)"
  if (( free_mib < MIN_FREE_MIB )); then
    echo "GPU $GPU_ID has ${free_mib} MiB free; ${MIN_FREE_MIB} MiB is required. Set GPU_ID or ALAM_TEST_MIN_FREE_MIB." >&2
    exit 1
  fi
}

run_cpu() {
  require_executable "$ENV_ROOT/alam/bin/python"
  require_executable "$ENV_ROOT/pi0/bin/python"
  require_executable "$ENV_ROOT/metaworld/bin/python"
  require_executable "$ENV_ROOT/libero/bin/python"
  bash workflows/tests/test_entrypoints.sh
  python3 workflows/tests/test_evaluation_paths.py
  python3 workflows/tests/test_github_preflight.py
  python3 workflows/tests/test_checkpoint_preparation.py
  python3 workflows/tests/test_wait_for_port.py
  python3 workflows/tests/test_evaluation_client_adapter.py
  python3 workflows/audit/validate.py --require-weights
  python3 workflows/tests/test_live_verifier.py
  python3 workflows/tests/test_archive_roundtrip.py
  "$ENV_ROOT/alam/bin/python" workflows/alam_pretraining/test_imports.py
  "$ENV_ROOT/alam/bin/python" workflows/alam_pretraining/test_datasets.py
  "$ENV_ROOT/alam/bin/python" workflows/alam_pretraining/test_video_backend.py
  "$ENV_ROOT/pi0/bin/python" workflows/pi0_post_training/tests/test_alam_imports.py
  "$ENV_ROOT/pi0/bin/python" workflows/pi0_post_training/tests/test_datasets.py
  "$ENV_ROOT/alam/bin/python" workflows/alam_pretraining/test_checkpoint.py --device cpu
  "$ENV_ROOT/alam/bin/python" workflows/alam_pretraining/test_checkpoint.py \
    --checkpoint evaluation/checkpoints/alam/libero_epoch16_step49024 --device cpu
  "$ENV_ROOT/pi0/bin/python" workflows/pi0_post_training/tests/test_checkpoint.py --benchmark metaworld
  "$ENV_ROOT/pi0/bin/python" workflows/pi0_post_training/tests/test_checkpoint.py --benchmark libero
  python3 workflows/pi0_post_training/tests/test_libero_assets.py
  "$ENV_ROOT/metaworld/bin/python" -c \
    'import metaworld; import openpi_client; print("MetaWorld evaluation environment import: PASS")'
  LIBERO_CONFIG_PATH="$REPO_ROOT/configs/libero" \
    PYTHONPATH="$REPO_ROOT/evaluation/third_party/libero" \
    "$ENV_ROOT/libero/bin/python" \
    workflows/pi0_post_training/tests/test_libero_imports.py
}

run_gpu() {
  require_executable "$ENV_ROOT/alam/bin/python"
  require_executable "$ENV_ROOT/pi0/bin/python"
  require_executable "$ENV_ROOT/libero/bin/python"
  check_gpu_capacity
  CUDA_VISIBLE_DEVICES="$GPU_ID" "$ENV_ROOT/alam/bin/python" workflows/alam_pretraining/test_checkpoint.py --device cuda
  CUDA_VISIBLE_DEVICES="$GPU_ID" "$ENV_ROOT/pi0/bin/python" workflows/pi0_post_training/tests/test_checkpoint.py --benchmark libero --gpu
  CUDA_VISIBLE_DEVICES="$EGL_GPU_ID" MUJOCO_GL=egl LIBERO_CONFIG_PATH="$REPO_ROOT/configs/libero" PYTHONPATH="$REPO_ROOT/evaluation/third_party/libero" "$ENV_ROOT/libero/bin/python" workflows/pi0_post_training/tests/test_libero_assets.py --render
}

run_e2e() {
  run_gpu
  local run_root="$REPO_ROOT/outputs/tests/libero_e2e"
  mkdir -p "$run_root"
  python3 workflows/common/check_port_free.py --host 127.0.0.1 --port 8000
  CUDA_VISIBLE_DEVICES="$GPU_ID" "$ENV_ROOT/pi0/bin/python" workflows/pi0_post_training/serve_policy.py libero --suite spatial --port 8000 >"$run_root/server.log" 2>&1 &
  local server_pid=$!
  cleanup_e2e() {
    if kill -0 "$server_pid" 2>/dev/null; then
      kill "$server_pid"
      wait "$server_pid" 2>/dev/null || true
    fi
  }
  trap cleanup_e2e EXIT INT TERM
  python3 workflows/common/wait_for_port.py \
    --host 127.0.0.1 --port 8000 --timeout 600 --pid "$server_pid"
  CUDA_VISIBLE_DEVICES="$EGL_GPU_ID" MUJOCO_GL=egl LIBERO_CONFIG_PATH="$REPO_ROOT/configs/libero" PYTHONPATH="$REPO_ROOT/evaluation/third_party/libero" "$ENV_ROOT/libero/bin/python" workflows/pi0_post_training/tests/test_libero_policy.py --suite spatial --host 127.0.0.1 --port 8000
  cleanup_e2e
  trap - EXIT INT TERM
}

case "$MODE" in
  cpu) run_cpu ;;
  gpu) run_gpu ;;
  e2e) run_cpu; run_e2e ;;
  *) echo "Usage: $0 {cpu|gpu|e2e}" >&2; exit 2 ;;
esac

echo "Release test mode '$MODE': PASS"
