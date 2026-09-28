#!/usr/bin/env bash
# Selected completed local MT50 evaluation: 432/500.
set -Eeuo pipefail
ulimit -c 0 2>/dev/null || true
[[ $# -eq 0 || ( $# -eq 1 && $1 == --dry-run ) ]] || { echo "usage: $0 [--dry-run]" >&2; exit 64; }

P="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
W="$(cd "$P/.." && pwd)"
R="$W/alam_runtime_20260901"
H="$W/physics_latent_world"
GPU=3
PORT=19915
CAM_X=0.71
PY="$R/python311/bin/python"
PI0_SITE="$P/.venvs/pi0/lib/python3.11/site-packages"
CLIENT_SITE="$H/examples/metaworld/.venv146/lib/python3.11/site-packages"
OPENPI_CLIENT="$H/packages/openpi-client/src"
SOURCE_OVERLAY="$R/metaworld_apr13_source_overlay_r1/src"
TRANSPORT_OVERLAY="$R/metaworld_client_transport_overlay_r1"
SERVER="$W/physics_latent_world_xingyun_training_latent/scripts/serve_policy.py"
EVALUATOR="$H/examples/metaworld/eval_metaworld_policy_client_0409.py"
POLICY="$W/physics_va_flow_checkpoints/phy_metaworld_full_finetune/phylam_gpu1_ah6_l1_loss_steps30001_0412_phy2_mix_11_larger_dataset_epoch19_step_58216/30000"
LAM="$W/UniWorldModel-256x4-skipframe10_high_speed_fp32/logs/ctl_v3_lam_tokenizer_trained_on_calvin/ctl_v3_lam_calvin_0407_64gpu_batch_32_train_fp32_mix_11_larger_dataset/saved_epoch_19_step_58216"
REFERENCE="$R/outputs/metaworld_camx_primary10_transport_recovery3_user_directed_r1_20260902/camx_071/metaworld_camx_summary.json"
REFERENCE_PROTOCOL="${REFERENCE%/*}/protocol.txt"
MANIFEST="$P/evaluation/WEIGHTS_MANIFEST.sha256"
SUMMARIZER="$P/workflows/best_observed_evaluation/support/summarize_metaworld.py"
BASE="$P/outputs/best_observed_evaluation"
MODEL_NAME=alam_pi0_mt50_camx071_h5_r5

[[ -x "$PY" && -f "$SERVER" && -f "$EVALUATOR" && -f "$REFERENCE" && -f "$REFERENCE_PROTOCOL" && -f "$POLICY/params/_METADATA" && -f "$LAM/pytorch_model.bin" ]]
printf '%s  %s\n' \
  8c97fb66651f19f7cba5869e70775fe9841d48fc5e63704c498cc43de9fa37a3 "$REFERENCE" \
  4017290184fa498cfe5319d368b47eb1d1adf53bb939001ce9900da26ef97dec "$REFERENCE_PROTOCOL" \
  | sha256sum --quiet -c -
python3 - "$REFERENCE" <<'PY'
import hashlib, json, sys
from pathlib import Path
d = json.loads(Path(sys.argv[1]).read_text())
assert d['execution_status'] == 'COMPLETE' and d['validation_status'] == 'STRUCTURALLY_VALID'
assert d['camera']['x'] == 0.71 and d['successes'] == 432 and d['episodes_count'] == 500
assert d['replan_steps'] == 5 and d['episodes_per_task'] == 10 and d['tasks_count'] == 50
assert d['episode_weighted_percent'] == 86.4
assert abs(d['difficulty_macro_percent'] - 85.36688311688312) < 1e-9
raw_log = Path(d['log'])
assert raw_log.is_file() and hashlib.sha256(raw_log.read_bytes()).hexdigest() == d['log_sha256']
PY

SERVER_PYTHONPATH="$SOURCE_OVERLAY:$OPENPI_CLIENT:$PI0_SITE"
CLIENT_PYTHONPATH="$CLIENT_SITE:$TRANSPORT_OVERLAY:$OPENPI_CLIENT"
if [[ ${1:-} == --dry-run ]]; then
  output="$BASE/dry_run_metaworld"
else
  mkdir -p "$BASE"
  output="$BASE/metaworld_$(date +%Y%m%d_%H%M%S)_${BASHPID}"
fi
PRIMARY_ROOT="$output/primary_10ep"
SERVER_CMD=(env -u XLA_PYTHON_CLIENT_PREALLOCATE -u XLA_CLIENT_MEM_FRACTION -u XLA_PYTHON_CLIENT_ALLOCATOR -u LEROBOT_HOME
  "CUDA_VISIBLE_DEVICES=$GPU" XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
  "PYTHONPATH=$SERVER_PYTHONPATH" HF_ENDPOINT=https://hf-mirror.com "HF_HOME=$R/cache/huggingface"
  "TRANSFORMERS_CACHE=$R/cache/huggingface/transformers" "OPENPI_DATA_HOME=$W/openpi_base" "HF_LEROBOT_HOME=$W/"
  "TORCH_HOME=$W/models" MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
  "$PY" -B "$SERVER" "--port=$PORT" policy:checkpoint --policy.config=phy_metaworld_full_finetune "--policy.dir=$POLICY")
CLIENT_CMD=(env "CUDA_VISIBLE_DEVICES=$GPU" MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
  "PYTHONPATH=$CLIENT_PYTHONPATH" PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
  NO_PROXY=0.0.0.0,127.0.0.1,localhost no_proxy=0.0.0.0,127.0.0.1,localhost
  "$PY" -B "$EVALUATOR" --port "$PORT" --chunk_size 5 --cam_pos_x "$CAM_X"
  --model_name "$MODEL_NAME" --output_dir "$PRIMARY_ROOT")
if [[ ${1:-} == --dry-run ]]; then
  printf 'reference=%s\nrecorded_successes=432/500\npolicy=%s\nlam=%s\ncamera=corner2@(0.71,0.075,0.70)\nexplicit_seed_cli=none\n' "$REFERENCE" "$POLICY" "$LAM"
  printf 'server_command='; printf '%q ' "${SERVER_CMD[@]}"; printf '\n'
  printf 'client_command='; printf '%q ' "${CLIENT_CMD[@]}"; printf '\n'
  exit 0
fi

mkdir "$output"
mkdir "$output/server" "$PRIMARY_ROOT"
exec >"$output/runner.log" 2>&1
status=PRECHECK
server_pid='' client_pid='' complete=0
cleanup() {
  code=$?
  trap - EXIT INT TERM
  for pid in "$client_pid" "$server_pid"; do
    [[ -n $pid ]] || continue
    if kill -0 "$pid" 2>/dev/null; then kill -TERM "$pid" 2>/dev/null || true; fi
    wait "$pid" 2>/dev/null || true
  done
  if [[ $code -eq 0 && $complete -eq 1 ]]; then
    printf 'COMPLETE_STRUCTURALLY_VALID see=metaworld_summary.json\n' >"$output/status"
  else
    printf 'FAILED stage=%s exit=%s\n' "$status" "$code" >"$output/status"
  fi
  date -Is >"$output/finished_at"
  exit "$code"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
date -Is >"$output/started_at"
printf '%s\n' "$BASHPID" >"$output/runner.pid"
printf 'suite=metaworld_mt50\nreference=%s\nrecorded_successes=432/500\ncamera_x=0.71\ncamera_y=0.075\ncamera_z=0.70\npolicy=%s\nlam=%s\nconfig=phy_metaworld_full_finetune\nraw_h=6\neffective_h=5\nchunk=5\nepisodes_per_task=10\nexplicit_seed_cli=none\nserver_gpu=3\nclient_gpu=3\nport=%s\n' "$REFERENCE" "$POLICY" "$LAM" "$PORT" >"$output/protocol.txt"

status=VERIFYING_SOURCE_AND_WEIGHTS; printf '%s\n' "$status" >"$output/status"
printf '%s  %s\n' \
  43a8a0e3bed3b322a785e51dfad35d959113c7daa89e6f93adfdb597c640d10b "$SERVER" \
  a82365f5cd67a4ecbfaffeb9034ffd50ac044d35af8d91f84e324aa748f6c590 "$EVALUATOR" \
  8a3b55fca6ad3c0ea6bca11e1b3bec02ea239406ae45b28e55fe50446e210246 "$H/examples/metaworld/mt50_eval_instruction.py" \
  f96009f787d6ccdebde077cc4f11829b89fcce677532af44b3d206a93d53d5b4 "$OPENPI_CLIENT/openpi_client/websocket_client_policy.py" \
  6371e4ebdda95749b5257503891c195485c3dbc647138e53b5d118c873adacd5 "$SOURCE_OVERLAY/openpi/training/config.py" \
  a43a47451515ae5634b1c6a4f36937fe47aad81a6a3dd1c63364d45458ad9e9b "$TRANSPORT_OVERLAY/openpi_client/websocket_client_policy.py" \
  8986bb4f423f07f8c7f70d0dbe3526fb2316056c17bae71b1ea975e77a168fc6 "$W/openpi_base/big_vision/paligemma_tokenizer.model" \
  397923af8e79cdbb6a7127f12361acd7a2f83e06b05044ddf496e83de57a5bf0 "$W/models/hub/checkpoints/vgg16-397923af.pth" \
  a78928a0af1e5f0fcb1f3b9e8f8c3a2a5a3de244d830ad5c1feddc79b8432868 "$PI0_SITE/lpips/weights/v0.1/vgg.pth" \
  64db8ac842c3835780846d58656b0aad9eebc6028fd1b10497b56fc1a8414cdc "$LAM/config.yaml" \
  bd99de14a552776c95251a9094398909444b81edbf1712c8c36a3137fc5dd845 "$LAM/pytorch_model.bin" \
  9e0e7dc6c06a09d9eeb7a729f9d48582a7a283705ae30c88b93b8e4829d9010c "$MANIFEST" \
  >"$output/selected_sources.sha256"
sha256sum -c "$output/selected_sources.sha256" >"$output/source_precheck.txt"
cmp -s "$SUMMARIZER" "$R/summarize_metaworld_camx_single.py"
[[ "$(env PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 "$PY" -B -c 'import platform,sys;print(sys.implementation.name+"|"+platform.python_version())')" == 'cpython|3.11.16' ]]
tree_hash="$(cd "$SOURCE_OVERLAY" && find openpi -type f ! -path '*/__pycache__/*' ! -name '*.pyc' -print0 | LC_ALL=C sort -z | xargs -0 sha256sum | sha256sum | awk '{print $1}')"
[[ "$tree_hash" == 2994c6b2bee65d96d6e82c1375e62e0abe6e9b51914429500eb1874b19ace9c7 ]]
rg -Fq "phy_lam_ckpt=\"$LAM\"" "$SOURCE_OVERLAY/openpi/training/config.py"
awk -v prefix='checkpoints/metaworld/phy_metaworld_full_finetune/phylam_gpu1_ah6_l1_loss_steps30001_0412_phy2_mix_11_larger_dataset_epoch19_step_58216/30000/' -v root="$POLICY/" 'index($2,prefix)==1 {sub(prefix,root,$2);print $1 "  " $2}' "$MANIFEST" >"$output/policy_manifest.sha256"
[[ $(wc -l <"$output/policy_manifest.sha256") -eq 19 ]]
sha256sum --quiet -c "$output/policy_manifest.sha256"
server_runtime="$(env PYTHONNOUSERSITE=1 PYTHONPATH="$SERVER_PYTHONPATH" "$PY" -B -c 'import importlib.metadata as m; print("|".join(f"{x}={m.version(x)}" for x in ("jax","jaxlib","flax","orbax-checkpoint","torch","torchvision","numpy","websockets","tyro","mujoco")))')"
[[ "$server_runtime" == 'jax=0.5.3|jaxlib=0.5.3|flax=0.10.2|orbax-checkpoint=0.11.13|torch=2.7.0|torchvision=0.22.0|numpy=1.26.4|websockets=15.0.1|tyro=0.9.22|mujoco=2.3.7' ]]
client_runtime="$(
  env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 \
    PYTHONPATH="$CLIENT_PYTHONPATH" "$PY" -B - <<'PY'
import importlib.metadata as metadata
import torch

versions = (
    f"metaworld={metadata.version('metaworld')}",
    f"mujoco={metadata.version('mujoco')}",
    f"gymnasium={metadata.version('gymnasium')}",
    f"numpy={metadata.version('numpy')}",
    f"torch={torch.__version__}",
    f"websockets={metadata.version('websockets')}",
)
print("|".join(versions))
PY
)"
[[ "$client_runtime" == 'metaworld=2.0.0|mujoco=3.6.0|gymnasium=1.2.2|numpy=1.26.4|torch=2.2.0+cu121|websockets=15.0.1' ]]
free="$(nvidia-smi -i "$GPU" --query-gpu=memory.free --format=csv,noheader,nounits)"
[[ "$free" -ge 25000 ]]
python3 "$P/workflows/common/check_port_free.py" --host 127.0.0.1 --port "$PORT"

status=SERVING; printf '%s\n' "$status" >"$output/status"
(cd "$H" && exec "${SERVER_CMD[@]}") >"$output/server/server.log" 2>&1 & server_pid=$!
ready=0
for ((i=0;i<1800;i++)); do
  kill -0 "$server_pid" 2>/dev/null || { echo 'server exited before ready' >&2; exit 1; }
  if rg -q "server listening on 0.0.0.0:$PORT" "$output/server/server.log"; then ready=1; break; fi
  sleep 1
done
[[ $ready -eq 1 ]]
rg -Fq "pretrained_path $LAM" "$output/server/server.log"
rg -Fq "load $LAM/pytorch_model.bin" "$output/server/server.log"

status=EVALUATING_500; printf '%s\n' "$status" >"$output/status"
(cd "$H" && exec timeout --signal=TERM --kill-after=30s 64800s "${CLIENT_CMD[@]}") >"$output/client.stdout.log" 2>&1 & client_pid=$!
finished_pid=''
set +e
wait -n -p finished_pid "$server_pid" "$client_pid"
wait_status=$?
set -e
[[ "$finished_pid" == "$client_pid" && $wait_status -eq 0 ]]
client_pid=''
kill -0 "$server_pid" 2>/dev/null

status=VALIDATING; printf '%s\n' "$status" >"$output/status"
mapfile -d '' logs < <(find "$PRIMARY_ROOT" -type f -name '*_episodes_10.log' -print0)
[[ ${#logs[@]} -eq 1 ]]
"$PY" -B "$SUMMARIZER" --log "${logs[0]}" --output "$output/metaworld_summary.json" --expected-port "$PORT" --camera-x "$CAM_X" --expected-model-name "$MODEL_NAME" >"$output/summary.stdout.log" 2>&1
sha256sum -c "$output/selected_sources.sha256" >"$output/source_postcheck.txt"
complete=1
