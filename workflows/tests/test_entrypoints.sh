#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

assert_contains() {
  local label="$1"
  local output="$2"
  local expected="$3"
  [[ "$output" == *"$expected"* ]] || {
    echo "$label did not contain expected text: $expected" >&2
    exit 1
  }
}

assert_not_contains() {
  local label="$1"
  local output="$2"
  local forbidden="$3"
  [[ "$output" != *"$forbidden"* ]] || {
    echo "$label unexpectedly contained: $forbidden" >&2
    exit 1
  }
}

assert_portable_output() {
  local label="$1"
  local output="$2"
  local forbidden
  # Build the former home path from two tokens so this negative test does not
  # itself place an internal absolute path literal in the public source tree.
  for forbidden in UniWorldModel-256x4 physics_va_flow_checkpoints "/home/"fortress; do
    [[ "$output" != *"$forbidden"* ]] || {
      echo "$label leaked a historical internal runtime path: $forbidden" >&2
      exit 1
    }
  done
}

pi0_install="$(UV_BIN=uv bash workflows/install/install_environments.sh \
  --components pi0 --dry-run)"
assert_contains pi0-install "$pi0_install" 'uv export --project'
assert_contains pi0-install "$pi0_install" '--frozen --no-dev --no-emit-project --no-emit-workspace'
assert_contains pi0-install "$pi0_install" '--default-index https://pypi.org/simple'
assert_not_contains pi0-install "$pi0_install" ' uv sync '
assert_contains pi0-install "$pi0_install" '.alam-install-in-progress'
assert_contains pi0-install "$pi0_install" '.alam-install-complete'
assert_portable_output pi0-install "$pi0_install"
rg -qx 'ipython==9.2.0' evaluation/requirements-alam.txt
rg -qx 'jedi==0.19.2' evaluation/requirements-alam.txt
rg -qx 'traitlets==5.14.3' evaluation/requirements-alam.txt
rg -qx 'packaging==25.0' evaluation/examples/metaworld/requirements-eval.txt
rg -qx 'gymnasium==1.2.2' evaluation/examples/metaworld/requirements-eval.txt
rg -qx 'mujoco==3.6.0' evaluation/examples/metaworld/requirements-eval.txt
rg -q -- '--episode-limit 1' workflows/pi0_post_training/smoke_train.sh
rg -q 'ping_timeout=None' workflows/pi0_post_training/serve_policy.py
rg -q 'ping_interval=None' workflows/pi0_post_training/run_evaluation_client.py
rg -Fq 'cd "$REPO_ROOT"' workflows/pi0_post_training/evaluate_libero.sh
rg -Fq 'evaluation/packages/openpi-client/src' workflows/pi0_post_training/evaluate_libero.sh
rg -q -- '--hard-exit-on-success' workflows/pi0_post_training/evaluate_libero.sh
rg -Fq 'os._exit(0)' workflows/pi0_post_training/run_evaluation_client.py
rg -Fq 'os._exit(0)' workflows/pi0_post_training/tests/test_libero_imports.py

pretrain="$(bash workflows/alam_pretraining/pretrain.sh --dry-run \
  --batch-size 1 --workers 1 --epochs 1)"
assert_contains pretrain "$pretrain" \
  "Algebraic_latent_action_model.latent_action_model.models.ctl_latent_action_tokenizer_v3_portable.LatentActionTokenizer"
assert_contains pretrain "$pretrain" "video_dir: data/alam/oxe_videos"
assert_contains pretrain "$pretrain" "npz_dir: data/alam/calvin"
assert_portable_output pretrain "$pretrain"

metaworld_train="$(bash workflows/pi0_post_training/finetune_metaworld.sh portable_dry_run --dry-run)"
assert_contains metaworld-train "$metaworld_train" '"raw_horizon": 6'
assert_contains metaworld-train "$metaworld_train" '"effective_horizon": 5'
assert_contains metaworld-train "$metaworld_train" 'data/lerobot/metaworld_mt50'
assert_portable_output metaworld-train "$metaworld_train"

libero_train="$(bash workflows/pi0_post_training/finetune_libero.sh portable_dry_run --dry-run)"
assert_contains libero-train "$libero_train" '"raw_horizon": 21'
assert_contains libero-train "$libero_train" '"effective_horizon": 20'
assert_contains libero-train "$libero_train" 'data/lerobot/libero_real'
assert_portable_output libero-train "$libero_train"

metaworld_eval="$(bash workflows/metaworld_evaluation/evaluate_mt50.sh --dry-run)"
assert_contains metaworld-eval "$metaworld_eval" 'raw_H=6 effective_H=5 replan=5'
assert_contains metaworld-eval "$metaworld_eval" '--action_horzion 5'
assert_contains metaworld-eval "$metaworld_eval" 'profile=oss_release_x071'
assert_contains metaworld-eval "$metaworld_eval" '--cam_pos_x 0.71'
assert_portable_output metaworld-eval "$metaworld_eval"

metaworld_historical_eval="$(bash workflows/metaworld_evaluation/evaluate_mt50.sh \
  --camera-x 0.70 --dry-run)"
assert_contains metaworld-historical-eval "$metaworld_historical_eval" \
  'profile=historical_evidence_x070'
assert_contains metaworld-historical-eval "$metaworld_historical_eval" '--cam_pos_x 0.70'
assert_portable_output metaworld-historical-eval "$metaworld_historical_eval"

for spec in 'spatial 14 5' 'object 14 10' 'goal 18 7' 'long 18 12'; do
  read -r suite horizon replan <<< "$spec"
  libero_eval="$(bash workflows/libero_evaluation/evaluate_libero.sh \
    "$suite" --dry-run)"
  assert_contains "libero-$suite" "$libero_eval" \
    "training_H=20 inference_H=$horizon replan=$replan"
  assert_contains "libero-$suite" "$libero_eval" \
    'policy_gpu=0 egl_gpu=0'
  assert_contains "libero-$suite" "$libero_eval" \
    "--args.replan_steps $replan"
  assert_portable_output "libero-$suite" "$libero_eval"
done

echo "Portable training/evaluation entrypoint dry runs: PASS"
