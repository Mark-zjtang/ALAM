#!/usr/bin/env bash
# Best observed: 493/500, independent 36-step Goal experiment (not the historical 18-step protocol).
set -Eeuo pipefail
ulimit -c 0 2>/dev/null || true
[[ $# -eq 0 || ( $# -eq 1 && $1 == --dry-run ) ]] || { echo "usage: $0 [--dry-run]" >&2; exit 64; }

P="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
W="$(cd "$P/.." && pwd)"
R="$W/alam_runtime_20260901"
D="$R/libero_goal_integrator36_20260923_2225"
F="$R/libero_aligned_trace_v1_20260910"
REF="$R/outputs/libero_default_cached_20260918_1104/goal_gpu8"
RECORD="$R/outputs/libero_goal_integrator36_20260923_2225/goal_gpu8/final_result.json"
POLICY="$W/physics_va_flow_checkpoints/phy_libero_full_finetune/phylam_gpu8_ah21_l1_loss_steps30001_0418_phy2_mix_11_larger_dataset_32_train_fp32_mix_11_larger_dataset_continue_from_epoch_24_step_73536_add_saved_epoch_16_step_49024/30000"
LAM="$W/UniWorldModel-256x4-skipframe10_high_speed_fp32/logs/ctl_v3_lam_tokenizer_trained_on_calvin/ctl_v3_lam_calvin_0407_64gpu_batch_32_train_fp32_mix_11_larger_dataset_continue_from_epoch_24_step_73536/saved_epoch_16_step_49024"
GPU=2 # reference used physical GPU3; GPU2 is the same A100 and frees GPU3 for Long.
PORT=19913
[[ -f "$RECORD" && -f "$REF/autotune_cache.textproto" && -f "$POLICY/params/_METADATA" && -f "$LAM/pytorch_model.bin" ]]
python3 - "$RECORD" <<'PY'
import json, sys
from pathlib import Path
record = json.loads(Path(sys.argv[1]).read_text())
assert record['suite'] == 'goal' and record['episodes'] == 500
assert record['successes'] == 493 and record['integration_steps'] == 36
assert record['structural_validation'] == 'PASS'
PY

unset PYTHONHOME PYTHONHASHSEED FIXED_SEED_TRACE FIXED_SEED_HISTORY_PYTHON LIBERO_DIAGNOSTIC_TRACE JAX_COMPILATION_CACHE_DIR XLA_FLAGS
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 JAX_ENABLE_COMPILATION_CACHE=false
export CUDA_VISIBLE_DEVICES="$GPU" XLA_PYTHON_CLIENT_MEM_FRACTION=0.4
export ALAM_TORCH_HOME="$W/models"
export OPENPI_DATA_HOME="$W/openpi_base"
export HF_HOME="$R/cache/huggingface" HF_LEROBOT_HOME="$W"
export PYTHONPATH="$P/evaluation/third_party/libero:$P/evaluation/packages/openpi-client/src"
export ALAM_LIBERO_POLICY="$POLICY" ALAM_LIBERO_TOKENIZER="$LAM"
export XLA_FLAGS="--xla_gpu_load_autotune_results_from=$REF/autotune_cache.textproto --xla_gpu_require_complete_aot_autotune_results=true"

base="$P/outputs/best_observed_evaluation"
if [[ ${1:-} == --dry-run ]]; then
  export INTEGRATOR_OUTPUT="$base/dry_run_goal" LIBERO_AUDIT_ROOT="$base/dry_run_goal/trace"
  export LIBERO_AUDIT_HISTORY_PYTHON="$R/libero_python38_history146.sh"
  export ALAM_RUNTIME_ROOT="$base/dry_run_goal/runtime" ALAM_OUTPUT_ROOT="$base/dry_run_goal/eval"
  printf 'reference=%s\nrecorded_successes=493/500\npolicy=%s\nlam=%s\nintegration_steps=36\nXLA_FLAGS=%s\n' "$RECORD" "$POLICY" "$LAM" "$XLA_FLAGS"
  exec env INTEGRATOR_MODE=serve PI0_PYTHON="$D/server_python.sh" LIBERO_PYTHON="$F/client_python.sh" ALAM_EGL_GPU=0 ALAM_SERVER_START_TIMEOUT=3600 bash "$R/evaluate_libero_frozen_openpi_client_cwd_overlay.sh" goal --gpu "$GPU" --port "$PORT" --trials 50 --dry-run
fi

mkdir -p "$base"
S="$base/goal_$(date +%Y%m%d_%H%M%S)_${BASHPID}"
mkdir "$S" # exclusive output; source and prior scores stay untouched
exec >"$S/runner.log" 2>&1
stage=PRECHECK
trap 'code=$?; printf "FAILED stage=%s exit=%s\n" "$stage" "$code" >"$S/status"; date -Is >"$S/finished_at"' ERR
date -Is >"$S/started_at"
printf '%s\n' "$BASHPID" >"$S/runner.pid"
printf 'suite=goal\nreference=%s\nrecorded_successes=493/500\npolicy=%s\nlam=%s\nconfig=phy_libero_full_finetune\ninfer_h=18\nreplan=7\nintegration_steps=36\nreference_gpu=3\nassigned_gpu=%s\negl_gpu=0\nport=%s\ntrials_per_task=50\nevaluator_seed=7\npolicy_rng=continuous_key0\nxla_flags=%s\nxla_memory_fraction=0.4\n' "$RECORD" "$POLICY" "$LAM" "$GPU" "$PORT" "$XLA_FLAGS" >"$S/protocol.txt"
cp "$D/PROTOCOL.md" "$S/PROTOCOL.reference.md"
printf '%s\n' "$stage" >"$S/status"
export INTEGRATOR_OUTPUT="$S" LIBERO_AUDIT_ROOT="$S/trace"
export LIBERO_AUDIT_HISTORY_PYTHON="$R/libero_python38_history146.sh"
export ALAM_RUNTIME_ROOT="$S/runtime" ALAM_OUTPUT_ROOT="$S/eval"
printf '%s\n' "$XLA_FLAGS" >"$S/xla_flags.txt"
nvidia-smi >"$S/gpu_before.txt"
free=$(nvidia-smi -i "$GPU" --query-gpu=memory.free --format=csv,noheader,nounits)
[[ "$free" -ge 25000 ]]
python3 "$P/workflows/common/check_port_free.py" --host 127.0.0.1 --port "$PORT"
sha256sum -c "$REF/source_before.sha256" >"$S/reference_sources_checked.txt"
sha256sum -c "$REF/autotune_cache.sha256" >"$S/cache_before.txt"
python3 "$F/preflight.py" "$S"
cmp "$S/checkpoint_identity.json" "$REF/checkpoint_identity.json"
sha256sum "$D"/*.py "$D"/*.sh "$D/PROTOCOL.md" "$P/workflows/best_observed_evaluation/evaluate_libero_goal.sh" "$R/evaluate_libero_frozen_openpi_client_cwd_overlay.sh" "$P/evaluation/examples/libero/main.py" >"$S/source_before.sha256"
cd "$P"
stage=ENVIRONMENT_GATES; printf '%s\n' "$stage" >"$S/status"
for label in a b; do
  env CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl LIBERO_CONFIG_PATH="$P/configs/libero" bash "$R/libero_python38_history146.sh" "$R/libero_default_cached_20260918_1104/diagnostics.py" environment goal "$REF/replay_inputs.json" "$S/env_$label.json" >"$S/env_$label.log" 2>&1
done
cmp "$S/env_a.json" "$S/env_b.json"
cmp "$S/env_a.json" "$REF/env_a.json"
stage=OFFLINE_GATES; printf '%s\n' "$stage" >"$S/status"
env INTEGRATOR_MODE=offline timeout 3600 bash "$R/pi0_python311_relinked.sh" "$D/server_gate.py" >"$S/offline.log" 2>&1
stage=SERVING_GATE_THEN_500_EPISODES; printf '%s\n' "$stage" >"$S/status"
env INTEGRATOR_MODE=serve PI0_PYTHON="$D/server_python.sh" LIBERO_PYTHON="$F/client_python.sh" ALAM_EGL_GPU=0 ALAM_SERVER_START_TIMEOUT=3600 timeout 86400 bash "$R/evaluate_libero_frozen_openpi_client_cwd_overlay.sh" goal --gpu "$GPU" --port "$PORT" --trials 50 >"$S/evaluation.stdout.log" 2>&1
printf '0\n' >"$S/evaluator_exit_code"
stage=VALIDATING; printf '%s\n' "$stage" >"$S/status"
sha256sum -c "$S/source_before.sha256" >"$S/source_after.txt"
sha256sum -c "$REF/source_before.sha256" >"$S/reference_sources_after.txt"
sha256sum -c "$REF/autotune_cache.sha256" >"$S/cache_after.txt"
python3 "$D/finish.py" "$S"
date -Is >"$S/finished_at"
printf 'COMPLETE_VALIDATED_INDEPENDENT_GOAL36 see=final_result.json\n' >"$S/status"
