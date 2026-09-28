# ALAM

本仓库是 ALAM 预训练及 ALAM + π0 下游训练/评估的开源发布版；下文把安装、路径、数据、权重、训练、评估、上传和审计统一放在这一个入口文档中。The repository is intentionally documented from this single root README.

This open-source package contains ALAM tokenizer pretraining and the ALAM + π0 downstream stack for MetaWorld MT50 and LIBERO. Portable entry points are grouped by stage under `workflows/`; they configure paths, cache locations, checkpoints, and evaluation settings at runtime.

Five date-free, evidence-bound local best-observed evaluation entry points are
staged under [`workflows/best_observed_evaluation/`](workflows/best_observed_evaluation/README.md).
They pin the completed MetaWorld evaluation and four independent LIBERO
settings. These launchers still depend on frozen historical sibling assets and
are not a self-contained public reproduction package; see their README for the
score, protocol, and publication boundary.

Code repository: [Mark-zjtang/ALAM](https://github.com/Mark-zjtang/ALAM).
This release publishes three model repositories on Hugging Face, but does not
publish or mirror training datasets. Download datasets from the original
projects listed below. Checkpoints and data are ignored by Git.

ALAM-authored code is dual-licensed under MIT or Apache-2.0 at the user's
option; see [LICENSE](LICENSE). Bundled third-party material retains its own
license. Redistribution rights for model and dataset artifacts must be
confirmed separately. The model-only upload script never transfers datasets.

## What is included

```text
ALAM/
├── configs/lam/alam_pretrain.yaml  # sole public config; byte-identical renamed YAML
├── Algebraic_latent_action_model/  # preserved ALAM model/data/trainer
├── train_lam.py                    # preserved ALAM training implementation
├── evaluation/
│   ├── src/openpi/                 # preserved ALAM + pi0 adaptation
│   ├── scripts/                    # preserved pi0 train/norm/server entry points
│   ├── examples/{metaworld,libero}/
│   ├── third_party/libero/         # simulator code, BDDL, init states, assets
│   ├── requirements-alam.txt       # exact ALAM-in-pi0 integration pins
│   └── checkpoints/                # local/HF model placement; ignored by Git
├── workflows/
│   ├── alam_pretraining/           # install, pretrain, checkpoint evaluation/test
│   ├── pi0_post_training/          # install and two downstream fine-tuning jobs
│   ├── metaworld_evaluation/       # isolated install + MT50 evaluation
│   ├── libero_evaluation/          # isolated install + four-suite evaluation
│   ├── install/                    # environments and one-command acceptance
│   ├── publishing/                 # GitHub/Hugging Face preparation
│   └── audit/                      # multi-pass validation
└── requirements/                   # curated per-environment installation inputs
```

Training and evaluation are separated as follows:

- ALAM pretraining: `workflows/alam_pretraining/pretrain.sh` calls the portable wrapper, then preserved `train_lam.py`.
- π0 downstream fine-tuning: `finetune_metaworld.sh` / `finetune_libero.sh` call preserved `evaluation/scripts/train.py` through the portable wrapper.
- Policy inference: `workflows/pi0_post_training/serve_policy.py` loads the published checkpoints.
- Simulator evaluation: `workflows/metaworld_evaluation/evaluate_mt50.sh` and `workflows/libero_evaluation/evaluate_libero.sh` run the preserved clients.

`train_lam.py` is not a second public launcher: its `main(cfg)` contains the byte-preserved dataset/dataloader/trainer implementation and is called by `workflows/alam_pretraining/train.py`. Its historical direct-CLI block points to a removed legacy config and must not be invoked directly. Unused imports and comments are retained deliberately because editing them would violate source identity.

The root `scripts/data_preprocessing/` files are byte-preserved provenance
snapshots, not public launchers; some retain historical machine defaults.
Download datasets from their original projects and configure the local paths
through `.env` or the documented CLI flags.

### One-command entry points

| Stage | Isolated installation | One-command run | Dependency source |
| --- | --- | --- | --- |
| ALAM pretraining | `bash workflows/alam_pretraining/install.sh` | `bash workflows/alam_pretraining/pretrain.sh` | `requirements/alam-runtime.txt` |
| ALAM checkpoint strict structure/load test | same ALAM environment | `bash workflows/alam_pretraining/evaluate_checkpoint.sh cpu` | same as above |
| ALAM real validation evaluation | same ALAM environment + one GPU | `bash workflows/alam_pretraining/evaluate_validation.sh --gpu 0` | same as above |
| π0 + ALAM MetaWorld fine-tuning | `bash workflows/pi0_post_training/install.sh` | `bash workflows/pi0_post_training/finetune_metaworld.sh EXP_NAME` | hash-locked export from `evaluation/uv.lock` + `evaluation/requirements-alam.txt` |
| π0 + ALAM LIBERO fine-tuning | same π0 environment | `bash workflows/pi0_post_training/finetune_libero.sh EXP_NAME` | same as above |
| MetaWorld MT50 evaluation | `bash workflows/metaworld_evaluation/install.sh` | `bash workflows/metaworld_evaluation/evaluate_mt50.sh --gpu 0` | `evaluation/examples/metaworld/requirements-eval.txt` + π0 server |
| LIBERO evaluation | `bash workflows/libero_evaluation/install.sh` | `bash workflows/libero_evaluation/evaluate_libero.sh SUITE --gpu 0` | `evaluation/examples/libero/requirements-eval.txt` + π0 server |
| LIBERO all-suite evaluation | same π0 + LIBERO environments | `bash workflows/libero_evaluation/evaluate_all_suites.sh --gpu 0` | same as above |
| Real relative-path training smoke | all training environments | `bash workflows/tests/smoke_training.sh --gpu 0,1` | ALAM + π0 environments |

Use `bash workflows/alam_pretraining/pretrain.sh --dry-run` and append `--dry-run` to either fine-tuning/evaluation command to resolve every path and parameter without starting a GPU workload.

## Quick start

Copy and edit the public path file. Relative values are resolved from the repository root.

```bash
cp .env.example .env
```

Install all isolated environments:

```bash
bash workflows/install/install_environments.sh --components all
```

Download the three model repositories. Training datasets are obtained from
their original projects, not from this release:

```bash
bash workflows/publishing/download_release.sh --models-only
bash workflows/tests/test_all.sh cpu
```

On an 80 GiB A100 with at least 60 GiB free, run CUDA and one real ALAM + π0 + LIBERO action:

```bash
GPU_ID=0 bash workflows/tests/test_all.sh gpu
GPU_ID=0 bash workflows/tests/test_all.sh e2e
```

`evaluate_checkpoint.sh cpu` strictly restores all 458 ALAM tensors and checks
the instantiated parameter/buffer structure without executing the encoder. The
preserved encoder contains an internal `.cuda()` call, so real encoder execution
is deliberately tested by `test_all.sh gpu` on CUDA; the core model file is not
edited to conceal that historical device assumption.

The install, model download, and E2E test can be started with one command
after the required datasets are prepared separately:

```bash
bash workflows/install/setup_and_test.sh --test e2e --gpu 0
```

For maintainers who already have the four relative dataset roots and four
released checkpoints, the strongest bounded training acceptance is:

```bash
bash workflows/tests/smoke_training.sh --gpu 0,1
```

This is not a configuration-only test. It decodes real CALVIN and OXE samples,
strictly restores 458 ALAM tensors, executes the preserved ALAM loss, backward
pass, and optimizer update, then runs one real MetaWorld and one real LIBERO π0
fine-tuning step from the released policy parameters using two-way FSDP. Each downstream smoke
saves an Orbax checkpoint and restores its parameter tree. Generated evidence
is written below `outputs/tests/`; full training defaults are not changed.

For a standalone ALAM validation check, the following command strictly restores
the 458-tensor checkpoint and reports preserved-model losses over real CALVIN
validation samples. This is a bounded release acceptance metric, not a claimed
paper benchmark:

```bash
bash workflows/alam_pretraining/evaluate_validation.sh --gpu 0 --samples 10
```

The GPU tests are intentionally guarded by a free-memory check. Override it only when appropriate with `ALAM_TEST_MIN_FREE_MIB`.

The environments are deliberately isolated: `.venvs/alam` (Python 3.10), `.venvs/pi0` (3.11), `.venvs/metaworld` (3.11), `.venvs/libero` (3.8), and `.venvs/publish` (3.11). Install just one component when desired, for example:

```bash
bash workflows/install/install_environments.sh --components alam
bash workflows/install/install_environments.sh --components pi0
bash workflows/install/install_environments.sh --components metaworld
bash workflows/install/install_environments.sh --components libero
bash workflows/install/install_environments.sh --components publish
```

The legacy `decord==0.6.0` distribution has a known packaging inconsistency:
its filename is tagged for Python 3 but its embedded `WHEEL` metadata retains a
CPython 3.6 tag. The installer waives only that exact `uv pip check` message,
then imports decord; CPU acceptance additionally decodes a real OXE MP4 frame.
Every other dependency-check finding remains fatal.

The installer prefers an existing `uv`; otherwise it bootstraps `uv` under `.tools/uv`. It never modifies a global Conda environment. Use `--dry-run` to inspect every command without installing anything.

The managed environment root carries per-environment installation-state
markers. If a transfer or package-copy step is interrupted, rerunning the same
install command recreates only that marked, isolated environment before
retrying. Environments that were not marked by this installer are never cleared
automatically. Keeping the marker outside the environment also covers an
interruption while `uv venv --clear` is rebuilding it. This prevents a partial
`site-packages` tree from being mistaken for a valid resumed installation.
After dependency resolution, the installer checks both dependency compatibility
and every path listed by installed wheel `RECORD` metadata; a missing module or
shared library is therefore fatal even when a package manager can still read
the distribution's version metadata.

By default uv's wheel cache is also repository-local at `.cache/uv`, so a small
home filesystem is not silently filled. Set `UV_CACHE_DIR` before installation
if a faster local cache volume is available; the environments themselves remain
under `ALAM_ENV_ROOT`. Package transfers retry transient network failures up to
five times (`ALAM_INSTALL_RETRIES` overrides this). The π0 installer forces
Git HTTP/1.1, while every retry still fetches the exact LeRobot commit recorded
in `uv.lock`.

The preserved π0 lock was generated on a source machine whose 348 registry
entries embed a regional mirror. The public installer therefore defaults to
`ALAM_PI0_INSTALL_MODE=portable`: it runs `uv export --frozen`, preserving every
resolved version, environment marker, distribution SHA-256, and immutable Git
revision, then installs that exported lock from
`ALAM_PI0_INDEX_URL=https://pypi.org/simple`. This changes only where identical
artifacts are fetched; it does not regenerate or edit `evaluation/uv.lock`.
Maintainers on the original mirror can request its embedded URLs with
`ALAM_PI0_INSTALL_MODE=source-lock`. For an offline or resumable install, place
exact wheels in a directory and set `ALAM_PI0_WHEEL_DIR`; their hashes must
still match the exported lock.

The π0 lock supplies Torch, OmegaConf, Einops, and Transformers. Loading the
preserved ALAM tokenizer adds two missing direct runtime packages:
`hydra-core==1.3.2` for config instantiation and `lpips==0.1.4` for its
constructor. The preserved source also imports IPython; although IPython is
recorded in the development lock, the public `--no-dev` export omits it.
`evaluation/requirements-alam.txt` therefore pins IPython 9.2.0 and its exact
lock-recorded transitive set as well. Install this small compatibility layer
after the hash-locked π0 dependencies; do not merge the complete ALAM
pretraining environment into π0.

The root `requirements.txt` remains byte-identical to ALAM. The separate public
`requirements/alam-runtime.txt` pins its formerly unpinned Torch, torchvision,
xFormers, IPython, and Diffusers direct requirements to the locally audited
Python 3.10/CUDA 12.8 resolution.

On its first ALAM construction, LPIPS/torchvision may download the standard
`vgg16-397923af.pth` auxiliary backbone (about 528 MiB) into
`.cache/torch/hub/checkpoints/`. Set `ALAM_TORCH_HOME` in `.env` to a preseeded
cache when running offline. This is a standard LPIPS dependency, not an ALAM
checkpoint; the two ALAM and two π0 releases remain the model artifacts listed
below.

The main `.env` fields and their public defaults are:

| Variable | Default | Meaning |
| --- | --- | --- |
| `ALAM_CALVIN_ROOT` | `data/alam/calvin` | CALVIN NPZ root |
| `ALAM_OXE_VIDEO_ROOT` | `data/alam/oxe_videos` | converted OXE video root |
| `HF_LEROBOT_HOME` | `data/lerobot` | parent of both downstream datasets |
| `ALAM_ENV_ROOT` | `.venvs` | isolated environments |
| `ALAM_RUNTIME_ROOT` | `.runtime` | generated path-only runtime configs |
| `ALAM_OUTPUT_ROOT` | `outputs` | all generated train/eval output |
| `ALAM_PI0_INSTALL_MODE` | `portable` | frozen hash export; use `source-lock` only for the original mirror |
| `ALAM_PI0_INDEX_URL` | `https://pypi.org/simple` | package index for the portable π0 install |
| `ALAM_PI0_WHEEL_DIR` | unset | optional offline/resumable exact-wheel directory |
| `ALAM_INSTALL_RETRIES` | `5` | retry count for dependency-transfer commands |
| `ALAM_SERVER_START_TIMEOUT` | `1800` | seconds allowed for a large policy restore before evaluator startup fails |
| `ALAM_METAWORLD_CAMERA_X/Y/Z` | `0.71/0.075/0.70` | public `corner2` camera position in absolute MuJoCo world coordinates |
| `ALAM_EGL_GPU` | `0` | visible GPU used by the preserved LIBERO/robosuite EGL renderer; independent of the policy `--gpu` |
| `ALAM_*_TOKENIZER` | `evaluation/checkpoints/alam/...` | downloaded ALAM checkpoints |
| `ALAM_*_POLICY` | `evaluation/checkpoints/...` | downloaded π0 checkpoints |

Shell environment variables override `.env`; explicit CLI flags override both. Portable workflows accept absolute user paths but default to repository-relative locations. The five separately documented, evidence-bound local evaluation launchers require their frozen sibling runtime.

No Python source edit is required for another machine. If data or weights are
stored elsewhere, copy `.env.example` to `.env` and edit only the right-hand
side. A relative value is always anchored at the repository root. In
particular, π0 resolves `repo_id=metaworld_mt50` as
`$HF_LEROBOT_HOME/metaworld_mt50`, and `repo_id=libero_real` as
`$HF_LEROBOT_HOME/libero_real`.

## Released model mapping

| Artifact | Original read-only path | Public local path | Hugging Face target |
| --- | --- | --- | --- |
| MetaWorld ALAM tokenizer | `/mnt/workspace/tangzuojin.tzj/UniWorldModel-256x4-skipframe10_high_speed_fp32/logs/ctl_v3_lam_tokenizer_trained_on_calvin/ctl_v3_lam_calvin_0407_64gpu_batch_32_train_fp32_mix_11_larger_dataset/saved_epoch_19_step_58216` | `evaluation/checkpoints/alam/metaworld_epoch19_step58216` | `Mark-ZJTang/alam_pretrain` |
| LIBERO ALAM tokenizer | `/mnt/workspace/tangzuojin.tzj/UniWorldModel-256x4-skipframe10_high_speed_fp32/logs/ctl_v3_lam_tokenizer_trained_on_calvin/ctl_v3_lam_calvin_0407_64gpu_batch_32_train_fp32_mix_11_larger_dataset_continue_from_epoch_24_step_73536/saved_epoch_16_step_49024` | `evaluation/checkpoints/alam/libero_epoch16_step49024` | `Mark-ZJTang/alam_pretrain` |
| MetaWorld π0 policy | `/mnt/workspace/tangzuojin.tzj/physics_va_flow_checkpoints/phy_metaworld_full_finetune/phylam_gpu1_ah6_l1_loss_steps30001_0412_phy2_mix_11_larger_dataset_epoch19_step_58216/30000` | `evaluation/checkpoints/metaworld/alam_plus_pi_metaworld_mt50_step30000` | `Mark-ZJTang/alam_plus_pi_metaworld_mt50` |
| LIBERO π0 policy | `/mnt/workspace/tangzuojin.tzj/physics_va_flow_checkpoints/phy_libero_full_finetune/phylam_gpu8_ah21_l1_loss_steps30001_0418_phy2_mix_11_larger_dataset_32_train_fp32_mix_11_larger_dataset_continue_from_epoch_24_step_73536_add_saved_epoch_16_step_49024/30000` | `evaluation/checkpoints/libero/alam_plus_pi_libero_step30000` | `Mark-ZJTang/alam_plus_pi_libero` |

The tokenizer files are exact, CPU-loadable state dicts with 458 tensor entries each:

| Public file | Bytes | SHA-256 |
| --- | ---: | --- |
| `metaworld_epoch19_step58216/config.yaml` | 428 | `64db8ac842c3835780846d58656b0aad9eebc6028fd1b10497b56fc1a8414cdc` |
| `metaworld_epoch19_step58216/pytorch_model.bin` | 861,738,935 | `bd99de14a552776c95251a9094398909444b81edbf1712c8c36a3137fc5dd845` |
| `libero_epoch16_step49024/config.yaml` | 428 | `64db8ac842c3835780846d58656b0aad9eebc6028fd1b10497b56fc1a8414cdc` |
| `libero_epoch16_step49024/pytorch_model.bin` | 861,738,935 | `9ea24d58a9eb444a23efd387ec94fd2937148fcba56fb376809245ea23943dcc` |

The MetaWorld policy package has 19 files / 12,075,867,227 bytes; the LIBERO package has 23 files / 12,076,381,073 bytes. All four model artifacts total 46 files / 25,875,727,026 bytes and are covered by [`evaluation/WEIGHTS_MANIFEST.sha256`](evaluation/WEIGHTS_MANIFEST.sha256).

The two π0 releases contain inference `params`, normalization `assets`, and `_CHECKPOINT_METADATA`; optimizer `train_state` is intentionally excluded. Their retained files were byte-compared to the original checkpoints.

## Dataset sources (not mirrored)

Training datasets are not bundled with this code and are not uploaded by the
model-publishing script. Obtain them from the original projects and follow
their license and download instructions:

| Input | Original project | Expected local layout |
| --- | --- | --- |
| CALVIN ABC→D | [CALVIN dataset](https://github.com/mees/calvin/blob/main/dataset/README.md) | `data/alam/calvin` after preprocessing |
| Open X-Embodiment mixture | [Google DeepMind Open X-Embodiment](https://github.com/google-deepmind/open_x_embodiment) | `data/alam/oxe_videos` after conversion |
| MetaWorld MT50 | [Farama MetaWorld](https://github.com/Farama-Foundation/Metaworld) | `data/lerobot/metaworld_mt50` for derived training trajectories |
| LIBERO | [LIBERO datasets](https://libero-project.github.io/datasets) | `data/lerobot/libero_real` after conversion |

The links point to upstream sources, not byte-identical copies of the
preprocessed training data. In particular, the MetaWorld project supplies the
benchmark environments, not this release's derived LeRobot trajectories.
To reproduce training, convert the upstream data into the layouts expected by
the loaders. Policy evaluation does not require the training datasets.

The OXE mixture uses the ten dataset roots listed in
[`scripts/data_preprocessing/oxe_dataset_configs.py`](scripts/data_preprocessing/oxe_dataset_configs.py).

## Training

ALAM pretraining (recommended public entry):

```bash
bash workflows/alam_pretraining/pretrain.sh
```

To continue from a downloaded tokenizer checkpoint, pass its repository-relative
directory explicitly; the preserved trainer reads `pytorch_model.bin` from that
directory:

```bash
bash workflows/alam_pretraining/pretrain.sh \
  --resume-checkpoint evaluation/checkpoints/alam/metaworld_epoch19_step58216 \
  --output-dir outputs/alam/resumed_run
```

The owner-specified full pretraining allocation is **128 NVIDIA H20 GPUs**, one process per GPU. The preserved per-GPU batch is 32 with gradient accumulation 2, giving a nominal effective global batch of 8,192. GPU count and multi-node rendezvous are launcher settings rather than YAML fields. Historical checkpoint paths containing `64gpu` remain provenance labels and are not the public hardware specification.

For example, a 16-node × 8-GPU topology runs the following on every node, with rank/address/port supplied by the cluster:

```bash
ALAM_NUM_MACHINES=16 ALAM_NUM_PROCESSES=128 \
MACHINE_RANK="$MACHINE_RANK" \
MAIN_PROCESS_IP="$MAIN_PROCESS_IP" \
MAIN_PROCESS_PORT="$MAIN_PROCESS_PORT" \
bash workflows/alam_pretraining/pretrain.sh
```

For another node topology, change only the launcher machine/process/rendezvous arguments, not `alam_pretrain.yaml`.

The launcher reads `ALAM_CALVIN_ROOT`, `ALAM_OXE_VIDEO_ROOT`, and `ALAM_OUTPUT_ROOT` from `.env`. CLI arguments such as `--calvin-root`, `--oxe-video-root`, and `--output-dir` override them.

`configs/lam/alam_pretrain.yaml` is the original `ctl_v3_lam_calvin_mix_data.yaml` under a public name. Both files hash to `2f4b98ebafec559b9cb0d0b51eed5bcc88d1a23f3c51af821b58b8860016c653`; no YAML value was edited. The workflow applies portable data/output/resume paths in memory.

Downstream π0 fine-tuning:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  bash workflows/pi0_post_training/finetune_metaworld.sh my_metaworld_run

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  bash workflows/pi0_post_training/finetune_libero.sh my_libero_run
```

The public full downstream run requires **8 GPUs**, and the wrapper defaults to `--fsdp-devices 8`. The MetaWorld training raw horizon is 6 (effective H=5). The shared LIBERO checkpoint was trained with raw horizon 21 (effective H=20). Both launchers default to 30,001 steps and use normalization statistics included with the released policy checkpoint. The preserved config's upstream π0 base-parameter URI remains the default; use `--base-params` for a local mirror.

Use `--dry-run` to print resolved paths and parameters without importing the GPU stack.

For a one-step check before committing an 8-GPU run:

```bash
bash workflows/tests/smoke_training.sh --gpu 0,1 --component metaworld
bash workflows/tests/smoke_training.sh --gpu 0,1 --component libero
```

These smoke commands require two idle 80 GiB GPUs, initialize from the already-trained released policy
`params`, use the corresponding released ALAM tokenizer and normalization
assets, load the relative LeRobot dataset, perform one optimizer step, save
step 0, and restore the saved Orbax tree. Each smoke explicitly selects the
first real episode with `--episode-limit 1`, avoiding an unnecessary Arrow
conversion of the complete dataset before a single-step check. Normal
fine-tuning does not set this acceptance-only option and still uses every
episode. Publication training remains 30,001 steps on 8 GPUs.

## Evaluation protocols

The selected, completed local evaluations have five independent launchers:

```bash
bash workflows/best_observed_evaluation/evaluate_metaworld.sh
bash workflows/best_observed_evaluation/evaluate_libero_spatial.sh
bash workflows/best_observed_evaluation/evaluate_libero_object.sh
bash workflows/best_observed_evaluation/evaluate_libero_goal.sh
bash workflows/best_observed_evaluation/evaluate_libero_long.sh
```

Their results, settings, and runtime requirements are in
[`workflows/best_observed_evaluation/README.md`](workflows/best_observed_evaluation/README.md).
These launchers require the frozen sibling runtime and are not yet
fresh-machine entry points. Append `--dry-run` to inspect a command. Each real
run writes to a new output directory; report it separately from its selected
reference result. Raw evidence and source hashes remain in the audit manifests.

The portable MetaWorld and LIBERO evaluation entry points are:

```bash
bash workflows/metaworld_evaluation/evaluate_mt50.sh --gpu 0
```

The open-source default fixes raw/effective horizon `6/5`, replan 5, camera
`corner2@(0.71, 0.075, 0.70)`, seed argument 10, 200 steps, 10 episodes per
task, and all 50 tasks. The completed release-profile observation is 432/500 =
86.4% episode-weighted and 85.366883% difficulty macro. Runtime versions,
code hashes, and transport behavior are in
[`workflows/metaworld_evaluation/RELEASE_PROFILE.md`](workflows/metaworld_evaluation/RELEASE_PROFILE.md)
and its machine-readable JSON contract.

LIBERO:

```bash
bash workflows/libero_evaluation/evaluate_libero.sh spatial --gpu 0
bash workflows/libero_evaluation/evaluate_libero.sh object  --gpu 0
bash workflows/libero_evaluation/evaluate_libero.sh goal    --gpu 0
bash workflows/libero_evaluation/evaluate_libero.sh long    --gpu 0

```

These portable LIBERO commands follow the documented Table 9 protocol. The
selected local outcomes and their exact independent settings are listed in the
best-observed evaluation README above; do not interchange those results with a
fresh portable run. Full protocol details are in
[`evaluation/README.md`](evaluation/README.md).

Each full suite evaluates 10 tasks × 50 trials. Use `--trials 1` for a short
simulator check. The scripts start and stop a local policy server by default;
add `--client-only --host HOST --port PORT` for an existing server. `--gpu`
selects the policy GPU. The preserved LIBERO/robosuite version addresses EGL by
its visible-device id, so rendering is isolated on `ALAM_EGL_GPU=0` by default;
change that variable only after
`CUDA_VISIBLE_DEVICES=N MUJOCO_GL=egl LIBERO_CONFIG_PATH=configs/libero PYTHONPATH=evaluation/third_party/libero .venvs/libero/bin/python workflows/pi0_post_training/tests/test_libero_assets.py --render`
passes on the target host. Policy and renderer may safely use different GPUs.

For WebSockets 15, the portable MetaWorld client adapter disables its keepalive
ping while the synchronous server may be blocked by first-call JAX compilation;
the server also disables its keepalive timeout. This matches the completed
x=0.71 run and prevents a healthy cold start from being closed after 20 seconds.
WebSockets 13 in the pinned LIBERO client has no sync keepalive option, so it
keeps native behavior. Port readiness, process-liveness, evaluator exceptions,
and completed-outcome checks remain enabled.

The pinned Python 3.8 LIBERO/robosuite native stack can also report
`free(): invalid pointer` during interpreter teardown after all evaluator code
has returned successfully. The LIBERO adapter therefore flushes output and uses
an immediate successful process exit only after a normal evaluator return.
Evaluator exceptions and nonzero `SystemExit` values still propagate unchanged;
the workaround does not turn a failed or incomplete rollout into a pass.

Checkpoint provenance caveat: Object, Goal, and Long evidence supports the packaged GPU8/epoch16 shared checkpoint. The Spatial client log does not record the server's loaded path, and the historical server comments conflict; Spatial's exact checkpoint attribution remains unproven by the retained log. The release uses the paper's shared-checkpoint configuration and states this uncertainty rather than fabricating evidence.

## Code and model publication

Code: [Mark-zjtang/ALAM](https://github.com/Mark-zjtang/ALAM).
Model weights are published in three Hugging Face repositories:

| Artifact | Hugging Face |
| --- | --- |
| MetaWorld and LIBERO ALAM tokenizers | [Mark-ZJTang/alam_pretrain](https://huggingface.co/Mark-ZJTang/alam_pretrain) |
| MetaWorld π0 policy | [Mark-ZJTang/alam_plus_pi_metaworld_mt50](https://huggingface.co/Mark-ZJTang/alam_plus_pi_metaworld_mt50) |
| LIBERO π0 policy shared by four suites | [Mark-ZJTang/alam_plus_pi_libero](https://huggingface.co/Mark-ZJTang/alam_plus_pi_libero) |

The exact local placement and file hashes are recorded in
[`huggingface_manifest.json`](workflows/publishing/huggingface_manifest.json)
and [`WEIGHTS_MANIFEST.sha256`](evaluation/WEIGHTS_MANIFEST.sha256).
To download only the models:

```bash
bash workflows/publishing/download_release.sh --models-only
```

The maintainer can inspect the upload plan, verify existing remote models, or
upload the three model repositories after confirming redistribution rights:

```bash
bash workflows/install/install_environments.sh --components publish
bash workflows/publishing/upload_models.sh --dry-run
bash workflows/publishing/upload_models.sh --verify-only
bash workflows/publishing/upload_models.sh --execute --acknowledge-rights
```

The execute command prompts for a Hugging Face write token without echoing or
saving it in the repository. The script checks all 46 local weight files before
upload and verifies the remote files afterward. It never uploads datasets.
For datasets, use the original-project links in
[Dataset sources](#dataset-sources-not-mirrored).

## Integrity and red line

Three audit layers are available:

```bash
# Preserved source and public runtime paths
python3 workflows/audit/validate.py

# All 25.9 GB of packaged model files
python3 workflows/audit/validate.py --hash-weights

# On this internal machine only: original code, evidence, data, and model byte identity
python3 workflows/audit/validate.py --source-code --source-datasets --source-models
```

The first check runs both source manifests. The original repositories and original model/data roots are read-only inputs; release changes stay in this repository.

Completed local audit evidence as of 2026-08-27 (GPU/live rows are updated only
after their corresponding run finishes):

| Independent pass | Result |
| --- | --- |
| Public structure, relative runtime paths, JSON/YAML, Python AST, shell syntax, symlinks | PASS |
| Root ALAM direct comparison | 25/25 byte-identical |
| Downstream release source integrity | PASS (1,345 files checked against the release manifest and source audit) |
| Packaged weight manifest/source comparison | 46/46 files, 25,875,727,026 bytes |
| Historical evaluation-log recalculation | MetaWorld 434/500; LIBERO 496/498/495/472 out of 500 |
| ALAM checkpoint loading | strict 458-tensor restore plus encoder execution is required by the test |
| Dataset relative paths | CALVIN, ten OXE roots, MetaWorld LeRobot, LIBERO LeRobot PASS |

Release gates that are intentionally still open:

1. Confirm redistribution rights and license metadata for all model and dataset artifacts.
2. Rebuild the expected training-data layouts from upstream sources and test them on a clean machine; upstream downloads alone are not byte-identical to the historical converted data.
3. Confirm the LIBERO Spatial checkpoint attribution from a server log or a controlled rerun; the retained client log alone does not prove the loaded server path.
4. When an A100 is idle, run `workflows/tests/test_all.sh gpu` and `e2e`. Static/CPU success must not be reported as simulator/GPU success.
