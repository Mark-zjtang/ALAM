#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$REPO_ROOT/workflows/common/read_env.sh"
load_project_env "$REPO_ROOT/.env"
ENV_ROOT="${ALAM_ENV_ROOT:-$REPO_ROOT/.venvs}"
[[ "$ENV_ROOT" == /* ]] || ENV_ROOT="$REPO_ROOT/$ENV_ROOT"
PI0_PYTHON="${PI0_PYTHON:-$ENV_ROOT/pi0/bin/python}"
METAWORLD_PYTHON="${METAWORLD_PYTHON:-$ENV_ROOT/metaworld/bin/python}"

GPU=0
PORT=8000
EPISODES=10
CAMERA_X="${ALAM_METAWORLD_CAMERA_X:-0.71}"
CAMERA_Y="${ALAM_METAWORLD_CAMERA_Y:-0.075}"
CAMERA_Z="${ALAM_METAWORLD_CAMERA_Z:-0.70}"
CLIENT_ONLY=0
DRY_RUN=0
HOST=127.0.0.1
SERVER_START_TIMEOUT="${ALAM_SERVER_START_TIMEOUT:-1800}"
usage() {
  echo "Usage: $0 [--gpu N] [--port N] [--episodes N] [--camera-x X] [--camera-y Y] [--camera-z Z] [--host HOST] [--client-only] [--dry-run]"
}
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu) GPU="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --episodes) EPISODES="$2"; shift 2 ;;
    --camera-x) CAMERA_X="$2"; shift 2 ;;
    --camera-y) CAMERA_Y="$2"; shift 2 ;;
    --camera-z) CAMERA_Z="$2"; shift 2 ;;
    --host) HOST="$2"; shift 2 ;;
    --client-only) CLIENT_ONLY=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done
if [[ ! "$SERVER_START_TIMEOUT" =~ ^[1-9][0-9]*$ ]]; then
  echo "ALAM_SERVER_START_TIMEOUT must be a positive integer, found: $SERVER_START_TIMEOUT" >&2
  exit 2
fi
if [[ ! "$EPISODES" =~ ^[1-9][0-9]*$ ]]; then
  echo "--episodes must be a positive integer, found: $EPISODES" >&2
  exit 2
fi
for camera_coordinate in "$CAMERA_X" "$CAMERA_Y" "$CAMERA_Z"; do
  if [[ ! "$camera_coordinate" =~ ^-?[0-9]+([.][0-9]+)?$ ]]; then
    echo "Camera coordinates must be decimal numbers, found: $camera_coordinate" >&2
    exit 2
  fi
done

OUTPUT_ROOT="${ALAM_OUTPUT_ROOT:-$REPO_ROOT/outputs}"
[[ "$OUTPUT_ROOT" == /* ]] || OUTPUT_ROOT="$REPO_ROOT/$OUTPUT_ROOT"
RUN_ROOT="$OUTPUT_ROOT/evaluation/metaworld_mt50"
CAMERA_TAG="${CAMERA_X//./p}"
CAMERA_TAG="${CAMERA_TAG//-/m}"
if [[ "$CAMERA_X" == "0.71" ]]; then
  PROFILE_ID="oss_release_x071"
elif [[ "$CAMERA_X" == "0.70" ]]; then
  PROFILE_ID="historical_evidence_x070"
else
  PROFILE_ID="custom_camera_x_${CAMERA_TAG}"
fi
LOCAL_NO_PROXY="${NO_PROXY:-},0.0.0.0,127.0.0.1,localhost"
LOCAL_NO_PROXY_LOWER="${no_proxy:-},0.0.0.0,127.0.0.1,localhost"
SERVER_CMD=(env "CUDA_VISIBLE_DEVICES=$GPU" "PYTHONUNBUFFERED=1" "$PI0_PYTHON" "$REPO_ROOT/workflows/pi0_post_training/serve_policy.py" metaworld --port "$PORT" --single-client)
CLIENT_CMD=(env "CUDA_VISIBLE_DEVICES=$GPU" "MUJOCO_GL=egl" "PYOPENGL_PLATFORM=egl" "MUJOCO_EGL_DEVICE_ID=0" "NO_PROXY=$LOCAL_NO_PROXY" "no_proxy=$LOCAL_NO_PROXY_LOWER" "$METAWORLD_PYTHON" "$REPO_ROOT/workflows/pi0_post_training/run_evaluation_client.py" "$REPO_ROOT/evaluation/examples/metaworld/eval_metaworld_policy_client_0409.py" --host "$HOST" --port "$PORT" --device cpu --max_episodes "$EPISODES" --chunk_size 5 --action_horzion 5 --cam_pos_x "$CAMERA_X" --cam_pos_y "$CAMERA_Y" --cam_pos_z "$CAMERA_Z" --seed 10 --eval_max_steps 200 --model_name "alam_plus_pi_metaworld_mt50_H5_replan5_camx_${CAMERA_TAG}" --output_dir "$RUN_ROOT")

print_cmd() {
  printf '+'
  printf ' %q' "$@"
  printf '\n'
}

echo "MetaWorld MT50: profile=$PROFILE_ID mode=single_client_serial raw_H=6 effective_H=5 replan=5 camera=corner2@($CAMERA_X,$CAMERA_Y,$CAMERA_Z) episodes_per_task=$EPISODES"
if [[ "$DRY_RUN" -eq 1 ]]; then
  [[ "$CLIENT_ONLY" -eq 1 ]] || print_cmd "${SERVER_CMD[@]}"
  print_cmd "${CLIENT_CMD[@]}"
  exit 0
fi
[[ -x "$METAWORLD_PYTHON" ]] || { echo "Missing environment executable: $METAWORLD_PYTHON" >&2; exit 1; }
if [[ "$CLIENT_ONLY" -eq 0 && ! -x "$PI0_PYTHON" ]]; then
  echo "Missing environment executable: $PI0_PYTHON" >&2
  exit 1
fi

mkdir -p "$RUN_ROOT/server"
"$METAWORLD_PYTHON" "$REPO_ROOT/workflows/metaworld_evaluation/record_runtime.py" \
  --role client --output "$RUN_ROOT/client_runtime.json" --profile "$PROFILE_ID" \
  --camera-x "$CAMERA_X" --camera-y "$CAMERA_Y" --camera-z "$CAMERA_Z" \
  --episodes "$EPISODES" --port "$PORT"
if [[ "$CLIENT_ONLY" -eq 0 ]]; then
  "$PI0_PYTHON" "$REPO_ROOT/workflows/metaworld_evaluation/record_runtime.py" \
    --role server --output "$RUN_ROOT/server_runtime.json" --profile "$PROFILE_ID" \
    --camera-x "$CAMERA_X" --camera-y "$CAMERA_Y" --camera-z "$CAMERA_Z" \
    --episodes "$EPISODES" --port "$PORT"
fi
SERVER_PID=""
cleanup() {
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID"
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

if [[ "$CLIENT_ONLY" -eq 0 ]]; then
  python3 "$REPO_ROOT/workflows/common/check_port_free.py" --host "$HOST" --port "$PORT"
  "${SERVER_CMD[@]}" >"$RUN_ROOT/server/server.log" 2>&1 &
  SERVER_PID=$!
  python3 "$REPO_ROOT/workflows/common/wait_for_port.py" \
    --host "$HOST" --port "$PORT" --timeout "$SERVER_START_TIMEOUT" --pid "$SERVER_PID"
fi
"${CLIENT_CMD[@]}"
