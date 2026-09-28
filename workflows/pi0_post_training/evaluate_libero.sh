#!/usr/bin/env bash
set -euo pipefail
ulimit -c 0 2>/dev/null || true

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$REPO_ROOT/workflows/common/read_env.sh"
load_project_env "$REPO_ROOT/.env"
cd "$REPO_ROOT"
ENV_ROOT="${ALAM_ENV_ROOT:-$REPO_ROOT/.venvs}"
[[ "$ENV_ROOT" == /* ]] || ENV_ROOT="$REPO_ROOT/$ENV_ROOT"
PI0_PYTHON="${PI0_PYTHON:-$ENV_ROOT/pi0/bin/python}"
LIBERO_PYTHON="${LIBERO_PYTHON:-$ENV_ROOT/libero/bin/python}"

SUITE="${1:-}"
usage() {
  echo "Usage: $0 {spatial|object|goal|long} [--policy-checkpoint PATH] [--alam-checkpoint PATH] [--output-dir PATH] [--gpu N] [--port N] [--trials N] [--host HOST] [--client-only] [--dry-run]"
}
if [[ -z "$SUITE" || "$SUITE" == "-h" || "$SUITE" == "--help" ]]; then
  usage
  [[ -n "$SUITE" ]] && exit 0
  exit 2
fi
shift

GPU=0
EGL_GPU="${ALAM_EGL_GPU:-0}"
PORT=8000
TRIALS=50
CLIENT_ONLY=0
DRY_RUN=0
HOST=127.0.0.1
SERVER_START_TIMEOUT="${ALAM_SERVER_START_TIMEOUT:-1800}"
MODEL_ARGS=()
OUTPUT_DIR=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --policy-checkpoint|--alam-checkpoint|--output-dir)
      [[ -n ${2:-} && $2 != --* ]] || { echo "$1 requires a path" >&2; exit 2; }
      if [[ $1 == --output-dir ]]; then OUTPUT_DIR="$2"; else MODEL_ARGS+=("$1" "$2"); fi
      shift 2 ;;
    --gpu) GPU="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --trials) TRIALS="$2"; shift 2 ;;
    --host) HOST="$2"; shift 2 ;;
    --client-only) CLIENT_ONLY=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help)
      usage
      exit 0
      ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
if (( CLIENT_ONLY && ${#MODEL_ARGS[@]} )); then
  echo 'Checkpoint options require a local policy server; omit --client-only.' >&2
  exit 2
fi
if [[ ! "$SERVER_START_TIMEOUT" =~ ^[1-9][0-9]*$ ]]; then
  echo "ALAM_SERVER_START_TIMEOUT must be a positive integer, found: $SERVER_START_TIMEOUT" >&2
  exit 2
fi

case "$SUITE" in
  spatial) CLI_SUITE=libero_spatial; INFER_H=14; REPLAN=5 ;;
  object) CLI_SUITE=libero_object; INFER_H=14; REPLAN=10 ;;
  goal) CLI_SUITE=libero_goal; INFER_H=18; REPLAN=7 ;;
  long) CLI_SUITE=libero_10; INFER_H=18; REPLAN=12 ;;
  *) echo "Unknown LIBERO suite: $SUITE" >&2; exit 2 ;;
esac

OUTPUT_ROOT="${ALAM_OUTPUT_ROOT:-$REPO_ROOT/outputs}"
[[ "$OUTPUT_ROOT" == /* ]] || OUTPUT_ROOT="$REPO_ROOT/$OUTPUT_ROOT"
RUN_ROOT="${OUTPUT_DIR:-$OUTPUT_ROOT/evaluation/libero/$SUITE}"
[[ "$RUN_ROOT" == /* ]] || RUN_ROOT="$REPO_ROOT/$RUN_ROOT"
SERVER_CMD=(env "CUDA_VISIBLE_DEVICES=$GPU" "PYTHONUNBUFFERED=1" "$PI0_PYTHON" "$REPO_ROOT/workflows/pi0_post_training/serve_policy.py" libero --suite "$SUITE" --port "$PORT" --single-client)
SERVER_CMD+=("${MODEL_ARGS[@]}")
# The released LIBERO/robosuite stack addresses EGL by the visible-device id.
# Keep rendering on its separately configurable, known-good device while the
# policy server remains on --gpu. On this validated host, EGL_GPU=0 is required.
CLIENT_SOURCE_PATH="$REPO_ROOT/evaluation/third_party/libero:$REPO_ROOT/evaluation/packages/openpi-client/src${PYTHONPATH:+:$PYTHONPATH}"
CLIENT_CMD=(env "CUDA_VISIBLE_DEVICES=$EGL_GPU" "MUJOCO_GL=egl" "LIBERO_CONFIG_PATH=$REPO_ROOT/configs/libero" "PYTHONPATH=$CLIENT_SOURCE_PATH" "$LIBERO_PYTHON" "$REPO_ROOT/workflows/pi0_post_training/run_evaluation_client.py" --hard-exit-on-success "$REPO_ROOT/evaluation/examples/libero/main.py" --args.host "$HOST" --args.port "$PORT" --args.task-suite-name "$CLI_SUITE" --args.replan_steps "$REPLAN" --args.num_trials_per_task "$TRIALS" --args.video_out_path "$RUN_ROOT/videos" --args.checkpoint_dir "$RUN_ROOT/logs" --args.policy_config alam_plus_pi_libero --args.model_name "alam_plus_pi_libero_trainH20_inferH${INFER_H}_replan${REPLAN}")

print_cmd() {
  printf '+'
  printf ' %q' "$@"
  printf '\n'
}

echo "LIBERO evaluation: mode=single_client_serial suite=$SUITE training_H=20 inference_H=$INFER_H replan=$REPLAN trials_per_task=$TRIALS policy_gpu=$GPU egl_gpu=$EGL_GPU"
if [[ "$DRY_RUN" -eq 1 ]]; then
  [[ "$CLIENT_ONLY" -eq 1 ]] || print_cmd "${SERVER_CMD[@]}"
  print_cmd "${CLIENT_CMD[@]}"
  exit 0
fi
for executable in "$LIBERO_PYTHON"; do
  [[ -x "$executable" ]] || { echo "Missing environment executable: $executable" >&2; exit 1; }
done
if [[ "$CLIENT_ONLY" -eq 0 && ! -x "$PI0_PYTHON" ]]; then
  echo "Missing environment executable: $PI0_PYTHON" >&2
  exit 1
fi

mkdir -p "$RUN_ROOT/server" "$RUN_ROOT/videos" "$RUN_ROOT/logs"
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
