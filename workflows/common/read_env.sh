#!/usr/bin/env bash

# Load only the documented workflow variables. Values are assigned literally;
# this file never evals or executes .env content.
load_project_env() {
  local env_file="$1"
  [[ -f "$env_file" ]] || return 0
  local line key value
  while IFS= read -r line || [[ -n "$line" ]]; do
    line="${line%$'\r'}"
    [[ -z "$line" || "$line" == \#* || "$line" != *=* ]] && continue
    key="${line%%=*}"
    value="${line#*=}"
    case "$key" in
      ALAM_CALVIN_ROOT|ALAM_OXE_VIDEO_ROOT|HF_LEROBOT_HOME|HF_HOME|OPENPI_DATA_HOME|ALAM_TORCH_HOME|ALAM_ENV_ROOT|ALAM_RUNTIME_ROOT|ALAM_OUTPUT_ROOT|ALAM_METAWORLD_TOKENIZER|ALAM_LIBERO_TOKENIZER|ALAM_METAWORLD_POLICY|ALAM_LIBERO_POLICY|ALAM_PI0_INSTALL_MODE|ALAM_PI0_INDEX_URL|ALAM_PI0_WHEEL_DIR|ALAM_INSTALL_RETRIES|ALAM_SERVER_START_TIMEOUT|ALAM_EGL_GPU) ;;
      *) continue ;;
    esac
    if [[ "$value" == \"*\" && "$value" == *\" ]]; then
      value="${value:1:${#value}-2}"
    elif [[ "$value" == \'*\' && "$value" == *\' ]]; then
      value="${value:1:${#value}-2}"
    fi
    if [[ -z "${!key+x}" ]]; then
      printf -v "$key" '%s' "$value"
      export "$key"
    fi
  done < "$env_file"
}
