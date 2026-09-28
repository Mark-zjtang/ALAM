#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
source "$REPO_ROOT/workflows/common/read_env.sh"
load_project_env "$REPO_ROOT/.env"

COMPONENTS="all"
DRY_RUN=0
ENV_ROOT="${ALAM_ENV_ROOT:-$REPO_ROOT/.venvs}"
ALAM_TORCH_INDEX_URL="${ALAM_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
LIBERO_TORCH_INDEX_URL="${LIBERO_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu113}"
PI0_INSTALL_MODE="${ALAM_PI0_INSTALL_MODE:-portable}"
PI0_INDEX_URL="${ALAM_PI0_INDEX_URL:-https://pypi.org/simple}"
PI0_WHEEL_DIR="${ALAM_PI0_WHEEL_DIR:-}"

# Large CUDA wheels occasionally hit transient proxy/CDN failures. These uv
# settings keep one-command installs resumable while preserving its cache.
export UV_HTTP_RETRIES="${UV_HTTP_RETRIES:-10}"
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-600}"
export UV_HTTP_CONNECT_TIMEOUT="${UV_HTTP_CONNECT_TIMEOUT:-30}"
export UV_CONCURRENT_DOWNLOADS="${UV_CONCURRENT_DOWNLOADS:-4}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$REPO_ROOT/.cache/uv}"
# The shared uv cache and this workspace can be different filesystems on DSW.
# Explicit copy mode avoids noisy hardlink fallback warnings and is portable.
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
INSTALL_RETRIES="${ALAM_INSTALL_RETRIES:-5}"

usage() {
  echo "Usage: $0 [--components LIST] [--env-root PATH] [--dry-run]"
  echo "Components: all, alam, pi0, metaworld, libero, publish"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --components)
      COMPONENTS="$2"
      shift 2
      ;;
    --env-root)
      ENV_ROOT="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ ! "$INSTALL_RETRIES" =~ ^[1-9][0-9]*$ ]]; then
  echo "ALAM_INSTALL_RETRIES must be a positive integer, found: $INSTALL_RETRIES" >&2
  exit 2
fi

case "$PI0_INSTALL_MODE" in
  portable|source-lock) ;;
  *)
    echo "ALAM_PI0_INSTALL_MODE must be portable or source-lock, found: $PI0_INSTALL_MODE" >&2
    exit 2
    ;;
esac

if [[ -n "$PI0_WHEEL_DIR" && "$PI0_WHEEL_DIR" != /* ]]; then
  PI0_WHEEL_DIR="$REPO_ROOT/$PI0_WHEEL_DIR"
fi

if [[ "$ENV_ROOT" != /* ]]; then
  ENV_ROOT="$REPO_ROOT/$ENV_ROOT"
fi

run() {
  printf '+'
  printf ' %q' "$@"
  printf '\n'
  if [[ "$DRY_RUN" -eq 0 ]]; then
    "$@"
  fi
}

run_with_retries() {
  local max_attempts="$1"
  shift
  printf '+'
  printf ' %q' "$@"
  printf '\n'
  if [[ "$DRY_RUN" -eq 1 ]]; then
    return 0
  fi

  local attempt=1
  while ! "$@"; do
    if (( attempt >= max_attempts )); then
      echo "Command failed after $attempt attempts." >&2
      return 1
    fi
    local delay=$((attempt * 5))
    echo "Command failed on attempt $attempt/$max_attempts; retrying in ${delay}s." >&2
    sleep "$delay"
    attempt=$((attempt + 1))
  done
}

contains_component() {
  local wanted="$1"
  [[ ",$COMPONENTS," == *,all,* || ",$COMPONENTS," == *,$wanted,* ]]
}

IFS=',' read -r -a REQUESTED_COMPONENTS <<< "$COMPONENTS"
for component in "${REQUESTED_COMPONENTS[@]}"; do
  case "$component" in
    all|alam|pi0|metaworld|libero|publish) ;;
    *)
      echo "Unknown component: $component" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ "$DRY_RUN" -eq 1 ]]; then
  UV_BIN="${UV_BIN:-uv}"
elif [[ -n "${UV_BIN:-}" ]]; then
  command -v "$UV_BIN" >/dev/null 2>&1 || [[ -x "$UV_BIN" ]]
elif command -v uv >/dev/null 2>&1; then
  UV_BIN="$(command -v uv)"
elif [[ -x "$REPO_ROOT/.tools/uv/bin/uv" ]]; then
  UV_BIN="$REPO_ROOT/.tools/uv/bin/uv"
else
  python3 -m venv "$REPO_ROOT/.tools/uv"
  "$REPO_ROOT/.tools/uv/bin/python" -m pip install --upgrade pip
  "$REPO_ROOT/.tools/uv/bin/python" -m pip install uv
  UV_BIN="$REPO_ROOT/.tools/uv/bin/uv"
fi

begin_venv_install() {
  local name="$1"
  local python_version="$2"
  local env_dir="$ENV_ROOT/$name"
  local state_dir="$ENV_ROOT/.alam-install-state"
  local incomplete_marker="$state_dir/$name.alam-install-in-progress"
  local resume_incomplete=0
  [[ -f "$incomplete_marker" ]] && resume_incomplete=1

  run mkdir -p "$state_dir"
  run touch "$incomplete_marker"

  # uv installs are transactional at the package-cache level, but a process
  # interruption while files are copied into site-packages can leave an
  # apparently installed yet incomplete environment. Only clear environments
  # carrying the marker written by this installer; never replace an unrelated
  # pre-existing environment merely because it lacks release metadata.
  if [[ "$resume_incomplete" -eq 1 ]]; then
    echo "Previous $name installation was interrupted; recreating its isolated environment."
    run "$UV_BIN" venv --clear --python "$python_version" "$env_dir"
  elif [[ ! -x "$env_dir/bin/python" ]]; then
    run "$UV_BIN" venv --python "$python_version" "$env_dir"
  fi
}

finish_venv_install() {
  local name="$1"
  local state_dir="$ENV_ROOT/.alam-install-state"
  run mv "$state_dir/$name.alam-install-in-progress" "$state_dir/$name.alam-install-complete"
}

install_alam() {
  begin_venv_install alam 3.10
  run_with_retries "$INSTALL_RETRIES" "$UV_BIN" pip install \
    --python "$ENV_ROOT/alam/bin/python" \
    --extra-index-url "$ALAM_TORCH_INDEX_URL" \
    --index-strategy unsafe-best-match \
    -r "$REPO_ROOT/requirements/alam-runtime.txt"
  run python3 "$REPO_ROOT/workflows/install/check_environment.py" \
    --uv "$UV_BIN" \
    --python "$ENV_ROOT/alam/bin/python" \
    --allow-decord-wheel-tag
  run python3 "$REPO_ROOT/workflows/install/check_installed_files.py" \
    --python "$ENV_ROOT/alam/bin/python"
  finish_venv_install alam
}

install_pi0() {
  begin_venv_install pi0 3.11
  if [[ "$PI0_INSTALL_MODE" == "source-lock" ]]; then
    # Compatibility mode for the regional registry URLs embedded by the
    # source machine. The public default below does not require that mirror.
    run_with_retries "$INSTALL_RETRIES" env \
      GIT_CONFIG_COUNT=1 \
      GIT_CONFIG_KEY_0=http.version \
      GIT_CONFIG_VALUE_0=HTTP/1.1 \
      UV_PROJECT_ENVIRONMENT="$ENV_ROOT/pi0" \
      "$UV_BIN" sync --project "$REPO_ROOT/evaluation" --frozen --no-dev --python "$ENV_ROOT/pi0/bin/python"
  else
    # Keep uv.lock byte-identical while removing its regional download
    # dependency. The frozen export retains every version, platform marker,
    # artifact hash, and immutable Git revision, then resolves those exact
    # artifacts from a public or user-selected index.
    local exported_requirements="$REPO_ROOT/.runtime/pi0-frozen-requirements.txt"
    if [[ "$DRY_RUN" -eq 0 ]]; then
      mkdir -p "$REPO_ROOT/.runtime"
    fi
    run "$UV_BIN" export \
      --project "$REPO_ROOT/evaluation" \
      --frozen \
      --no-dev \
      --no-emit-project \
      --no-emit-workspace \
      --format requirements-txt \
      --output-file "$exported_requirements" \
      --quiet

    local -a install_args=(
      env
      GIT_CONFIG_COUNT=1
      GIT_CONFIG_KEY_0=http.version
      GIT_CONFIG_VALUE_0=HTTP/1.1
      "$UV_BIN" pip install
      --python "$ENV_ROOT/pi0/bin/python"
      --no-deps
      --default-index "$PI0_INDEX_URL"
      --requirements "$exported_requirements"
    )
    if [[ -n "$PI0_WHEEL_DIR" ]]; then
      if [[ "$DRY_RUN" -eq 0 && ! -d "$PI0_WHEEL_DIR" ]]; then
        echo "ALAM_PI0_WHEEL_DIR does not exist: $PI0_WHEEL_DIR" >&2
        exit 2
      fi
      install_args+=(--find-links "$PI0_WHEEL_DIR")
    fi
    run_with_retries "$INSTALL_RETRIES" "${install_args[@]}"
    run "$UV_BIN" pip install \
      --python "$ENV_ROOT/pi0/bin/python" \
      --no-deps \
      --editable "$REPO_ROOT/evaluation" \
      --editable "$REPO_ROOT/evaluation/packages/openpi-client"
  fi
  run_with_retries "$INSTALL_RETRIES" "$UV_BIN" pip install \
    --python "$ENV_ROOT/pi0/bin/python" \
    -r "$REPO_ROOT/evaluation/requirements-alam.txt"
  run "$UV_BIN" pip check --python "$ENV_ROOT/pi0/bin/python"
  run python3 "$REPO_ROOT/workflows/install/check_installed_files.py" \
    --python "$ENV_ROOT/pi0/bin/python"
  finish_venv_install pi0
}

install_metaworld() {
  begin_venv_install metaworld 3.11
  run_with_retries "$INSTALL_RETRIES" "$UV_BIN" pip install \
    --python "$ENV_ROOT/metaworld/bin/python" \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    --index-strategy unsafe-best-match \
    -r "$REPO_ROOT/evaluation/examples/metaworld/requirements-eval.txt"
  run_with_retries "$INSTALL_RETRIES" "$UV_BIN" pip install --python "$ENV_ROOT/metaworld/bin/python" -e "$REPO_ROOT/evaluation/packages/openpi-client"
  run "$UV_BIN" pip check --python "$ENV_ROOT/metaworld/bin/python"
  run python3 "$REPO_ROOT/workflows/install/check_installed_files.py" \
    --python "$ENV_ROOT/metaworld/bin/python"
  finish_venv_install metaworld
}

install_libero() {
  begin_venv_install libero 3.8
  run_with_retries "$INSTALL_RETRIES" "$UV_BIN" pip install \
    --python "$ENV_ROOT/libero/bin/python" \
    --extra-index-url "$LIBERO_TORCH_INDEX_URL" \
    --index-strategy unsafe-best-match \
    -r "$REPO_ROOT/evaluation/examples/libero/requirements-eval.txt"
  run_with_retries "$INSTALL_RETRIES" "$UV_BIN" pip install --python "$ENV_ROOT/libero/bin/python" -e "$REPO_ROOT/evaluation/packages/openpi-client"
  run_with_retries "$INSTALL_RETRIES" "$UV_BIN" pip install --python "$ENV_ROOT/libero/bin/python" -e "$REPO_ROOT/evaluation/third_party/libero"
  run "$UV_BIN" pip check --python "$ENV_ROOT/libero/bin/python"
  run python3 "$REPO_ROOT/workflows/install/check_installed_files.py" \
    --python "$ENV_ROOT/libero/bin/python"
  finish_venv_install libero
}

install_publish() {
  begin_venv_install publish 3.11
  run_with_retries "$INSTALL_RETRIES" "$UV_BIN" pip install --python "$ENV_ROOT/publish/bin/python" "huggingface_hub>=1.0,<2"
  run "$UV_BIN" pip check --python "$ENV_ROOT/publish/bin/python"
  run python3 "$REPO_ROOT/workflows/install/check_installed_files.py" \
    --python "$ENV_ROOT/publish/bin/python"
  finish_venv_install publish
}

if [[ "$DRY_RUN" -eq 0 ]]; then
  mkdir -p "$ENV_ROOT"
fi
contains_component alam && install_alam
contains_component pi0 && install_pi0
contains_component metaworld && install_metaworld
contains_component libero && install_libero
contains_component publish && install_publish

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "Environment installation plan complete (no files created): $ENV_ROOT"
else
  echo "Environment installation complete: $ENV_ROOT"
fi
