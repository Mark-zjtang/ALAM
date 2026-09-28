#!/usr/bin/env bash
# Shared mechanics only. The independently executable suite launchers freeze inputs.
set -Eeuo pipefail
ulimit -c 0 2>/dev/null || true

[[ $# -eq 6 || ( $# -eq 7 && $7 == --dry-run ) ]] || {
  echo "usage: $0 SUITE GPU PORT CLIENT_PY REFERENCE EXPECTED [--dry-run]" >&2
  exit 64
}
suite=$1 gpu=$2 port=$3 client_py=$4 reference=$5 expected=$6
dry_run=${7:-}
P="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
W="$(cd "$P/.." && pwd)"
R="$W/alam_runtime_20260901"
case "$suite" in
  spatial) [[ $gpu == 0 && $port == 19911 && $client_py == "$R/libero_python38_history135.sh" && $expected == 495 ]] ;;
  object)  [[ $gpu == 1 && $port == 19912 && $client_py == "$R/libero_python38_history135.sh" && $expected == 496 ]] ;;
  long)    [[ $gpu == 3 && $port == 19914 && $client_py == "$R/libero_python38_history146.sh" && $expected == 475 ]] ;;
  *) echo "Unsupported release suite: $suite" >&2; exit 64 ;;
esac
[[ -f $reference && -f $client_py && -f $R/evaluate_libero_frozen_openpi_client_cwd_overlay.sh ]]
python3 - "$reference" "$suite" "$expected" <<'PY'
import json, sys
from pathlib import Path
report = json.loads(Path(sys.argv[1]).read_text())
rows = [row for row in report['suites'] if row['suite'] == sys.argv[2]]
assert report['execution_status'] == 'COMPLETE' and len(rows) == 1
assert rows[0]['validation_status'] == 'STRUCTURALLY_VALID'
assert rows[0]['episodes'] == 500 and rows[0]['successes'] == int(sys.argv[3])
PY

policy="$P/evaluation/checkpoints/libero/phy_libero_full_finetune/phylam_gpu8_ah21_l1_loss_steps30001_0418_phy2_mix_11_larger_dataset_32_train_fp32_mix_11_larger_dataset_continue_from_epoch_24_step_73536_add_saved_epoch_16_step_49024/30000"
lam="$P/evaluation/checkpoints/alam/libero_epoch16_step49024"
[[ -f $policy/params/_METADATA && -f $policy/assets/libero_real/norm_stats.json && -f $lam/pytorch_model.bin ]]

unset PYTHONHOME PYTHONHASHSEED FIXED_SEED_TRACE FIXED_SEED_HISTORY_PYTHON LIBERO_DIAGNOSTIC_TRACE XLA_FLAGS JAX_COMPILATION_CACHE_DIR JAX_ENABLE_COMPILATION_CACHE
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export ALAM_LIBERO_POLICY="$policy" ALAM_LIBERO_TOKENIZER="$lam"
export PI0_PYTHON="$R/pi0_python311_relinked.sh" LIBERO_PYTHON="$client_py"
export ALAM_EGL_GPU=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 ALAM_SERVER_START_TIMEOUT=1800
export ALAM_TORCH_HOME="$W/models"
export OPENPI_DATA_HOME="$W/openpi_base"
export HF_HOME="$R/cache/huggingface" HF_LEROBOT_HOME="$W"
export PYTHONPATH="$P/evaluation/third_party/libero:$P/evaluation/packages/openpi-client/src"

base="$P/outputs/best_observed_evaluation"
if [[ $dry_run == --dry-run ]]; then
  export ALAM_RUNTIME_ROOT="$base/dry_run_$suite/runtime" ALAM_OUTPUT_ROOT="$base/dry_run_$suite/eval"
  printf 'reference=%s\nrecorded_successes=%s/500\npolicy=%s\nlam=%s\nXLA_FLAGS=<unset>\n' "$reference" "$expected" "$policy" "$lam"
  exec bash "$R/evaluate_libero_frozen_openpi_client_cwd_overlay.sh" "$suite" --gpu "$gpu" --port "$port" --trials 50 --dry-run
fi

run_id="${suite}_$(date +%Y%m%d_%H%M%S)_${BASHPID}"
mkdir -p "$base"
S="$base/$run_id"
mkdir "$S" # exclusive; never append to or overwrite historical results
exec >"$S/runner.log" 2>&1
stage=PRECHECK
trap 'code=$?; printf "FAILED stage=%s exit=%s\n" "$stage" "$code" >"$S/status"; date -Is >"$S/finished_at"' ERR
date -Is >"$S/started_at"
printf '%s\n' "$BASHPID" >"$S/runner.pid"
printf 'suite=%s\nreference=%s\nrecorded_successes=%s/500\npolicy=%s\nlam=%s\ngpu=%s\negl_gpu=0\nport=%s\nclient_py=%s\ntrials_per_task=50\nevaluator_seed=7\npolicy_rng=continuous_key0\nxla_flags=unset\nxla_memory_fraction=0.4\n' "$suite" "$reference" "$expected" "$policy" "$lam" "$gpu" "$port" "$client_py" >"$S/protocol.txt"
export ALAM_RUNTIME_ROOT="$S/runtime" ALAM_OUTPUT_ROOT="$S/eval"
sha256sum -c "$(dirname "$reference")/source_hashes.sha256" >"$S/reference_source_precheck.txt"
cmp -s "$P/evaluation/examples/libero/main.py" "$W/physics_latent_world/examples/libero/main.py"
cmp -s "$P/evaluation/packages/openpi-client/src/openpi_client/websocket_client_policy.py" "$W/physics_latent_world/packages/openpi-client/src/openpi_client/websocket_client_policy.py"
"$client_py" -VV >"$S/client_environment.txt" 2>&1
"$client_py" -c 'import importlib.metadata as m; print({x:m.version(x) for x in ("mujoco","robosuite","numpy","torch","torchvision","websockets","gym")})' >>"$S/client_environment.txt" 2>&1
rg -q 'Python 3\.8\.20' "$S/client_environment.txt"
rg -q "'mujoco': '3.2.3'" "$S/client_environment.txt"
rg -q "'websockets': '13.1'" "$S/client_environment.txt"
free=$(nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits)
[[ "$free" -ge 25000 ]]
python3 "$P/workflows/common/check_port_free.py" --host 127.0.0.1 --port "$port"
env CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MPLCONFIGDIR="$S/matplotlib" LIBERO_CONFIG_PATH="$P/configs/libero" "$client_py" "$P/workflows/pi0_post_training/tests/test_libero_assets.py" --suite spatial --render >"$S/egl_preflight.log" 2>&1
rg -q 'LIBERO EGL PASS' "$S/egl_preflight.log"
sha256sum "$P/evaluation/examples/libero/main.py" "$R/evaluate_libero_frozen_openpi_client_cwd_overlay.sh" "$client_py" "$P/workflows/best_observed_evaluation/support/libero_runner.bash" "$P/workflows/best_observed_evaluation/evaluate_libero_$suite.sh" "$P/workflows/best_observed_evaluation/support/validate_libero.py" >"$S/source_before.sha256"
stage=EVALUATING
printf '%s\n' "$stage" >"$S/status"
bash "$R/evaluate_libero_frozen_openpi_client_cwd_overlay.sh" "$suite" --gpu "$gpu" --port "$port" --trials 50 >"$S/evaluation.stdout.log" 2>&1
stage=VALIDATING
printf '%s\n' "$stage" >"$S/status"
sha256sum -c "$S/source_before.sha256" >"$S/source_after.txt"
python3 -B "$P/workflows/best_observed_evaluation/support/validate_libero.py" "$S" "$suite"
printf 'COMPLETE_STRUCTURALLY_VALID see=result.json\n' >"$S/status"
date -Is >"$S/finished_at"
