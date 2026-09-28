# ALAM + π0 downstream release

This directory contains the ALAM + π0 downstream source and assets for training, policy serving, MetaWorld MT50 evaluation, and LIBERO evaluation. Published files are protected by `SOURCE_MANIFEST.sha256`.

Portable public entry points live one level above under `workflows/`, separated into ALAM pretraining and π0 post-training. They apply repository-relative paths and publication parameters at runtime, then call the preserved implementations.

## Training and evaluation separation

| Purpose | Public entry | Preserved implementation |
| --- | --- | --- |
| π0 downstream training | `../workflows/pi0_post_training/train.py` | `scripts/train.py` |
| normalization statistics | documented preserved entry | `scripts/compute_norm_stats.py` |
| policy server | `../workflows/pi0_post_training/serve_policy.py` | openpi policy/config/server modules |
| MetaWorld MT50 client | `../workflows/metaworld_evaluation/evaluate_mt50.sh` | `examples/metaworld/eval_metaworld_policy_client_0409.py` |
| LIBERO client | `../workflows/libero_evaluation/evaluate_libero.sh` | `examples/libero/main.py` |

The additive portable ALAM adapters only prevent preserved constructors from forcing an internal `TORCH_HOME`; the inherited model computation and state-dict keys are unchanged. Runtime checkpoint configs are generated beneath `.runtime/` and are not published as original source.

## Published inference models

Friendly paths:

```text
checkpoints/alam/metaworld_epoch19_step58216
checkpoints/alam/libero_epoch16_step49024
checkpoints/metaworld/alam_plus_pi_metaworld_mt50_step30000
checkpoints/libero/alam_plus_pi_libero_step30000
```

The two policy packages contain only `params`, normalization `assets`, and `_CHECKPOINT_METADATA`; optimizer `train_state` is excluded. All four model artifacts are covered by `WEIGHTS_MANIFEST.sha256`. Original machine paths and Hugging Face targets are in the root [`README.md`](../README.md) and [`../workflows/publishing/huggingface_manifest.json`](../workflows/publishing/huggingface_manifest.json).

## MetaWorld MT50 protocol

The open-source default profile is:

- config: `phy_metaworld_full_finetune`;
- policy: `checkpoints/metaworld/alam_plus_pi_metaworld_mt50_step30000`;
- ALAM: `checkpoints/alam/metaworld_epoch19_step58216`;
- raw action horizon 6, effective inference H=5;
- replan/chunk 5;
- camera `corner2@(0.71, 0.075, 0.70)`;
- seed 10, max 200 steps;
- 50 tasks × 10 episodes.

```bash
cd ..
bash workflows/metaworld_evaluation/evaluate_mt50.sh --gpu 0
```

The completed x=0.71 observation is 432/500 (86.4% episode-weighted) and
85.366883% difficulty macro. The full environment, source hashes, and transport
setting are recorded in
`../workflows/metaworld_evaluation/RELEASE_PROFILE.md` and
`release_profile_x071.json` beside it.

## LIBERO Table 9 protocol

All suites use the same policy candidate and a training horizon of 20. The public server converts the paper's effective inference H to raw model horizon by adding one frame transition.

| Suite argument | LIBERO suite | Effective H | Raw H | Replan | Success |
| --- | --- | ---: | ---: | ---: | ---: |
| `spatial` | `libero_spatial` | 14 | 15 | 5 | 99.2 |
| `object` | `libero_object` | 14 | 15 | 10 | 99.6 |
| `goal` | `libero_goal` | 18 | 19 | 7 | 99.0 |
| `long` | `libero_10` | 18 | 19 | 12 | 94.4 |
| Average | — | — | — | — | 98.1 |

```bash
cd ..
bash workflows/libero_evaluation/evaluate_libero.sh spatial --gpu 0
```

Use `--trials 1` for a short simulator run. A full suite is 10 tasks × 50 trials.

Provenance limitation: Object, Goal, and Long sweep evidence points to the packaged GPU8/epoch16 policy/ALAM combination. The Spatial=99.2 client log does not record the server checkpoint, while historical commented server commands conflict. Thus the release implements the paper's shared-checkpoint configuration but does not claim the retained Spatial client log independently proves that checkpoint.

The historical values also came from parallel replan sweeps sharing policy
servers. The released one-server/one-client topology changes policy RNG request
ordering. It is safer and auditable, but it does not reconstruct the unrecorded
historical interleaving; a fresh complete score may differ even when the fixed
init states, client packages, checkpoint, H, and replan values match.

## Environment boundary

The π0 server uses the root `.venvs/pi0` environment. MetaWorld and LIBERO clients use `.venvs/metaworld` and `.venvs/libero` respectively. Install them with:

```bash
cd ..
bash workflows/install/install_environments.sh --components pi0,metaworld,libero
```

Do not install all preserved requirement snapshots into a single interpreter.

## Integrity

From this directory:

```bash
sha256sum -c SOURCE_MANIFEST.sha256
sha256sum -c WEIGHTS_MANIFEST.sha256
```

From the repository root, `python3 workflows/audit/validate.py` adds public-path and structure checks. Internal source-machine audits can also validate dataset metadata and byte-compare all 46 model files to their originals.

Exact public commands, source paths, provenance caveats, and the multi-pass audit are consolidated in the root [`README.md`](../README.md).
