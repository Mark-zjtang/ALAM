# ALAM + π0 post-training

We initialize π0 from its base checkpoint and train it on action-labeled demonstrations while keeping the Mix-11-pretrained ALAM encoder frozen. This produces the MetaWorld and LIBERO policy weights released on Hugging Face.

Set `HF_LEROBOT_HOME` in the root `.env`. Training expects converted LeRobot datasets under `metaworld_mt50/` and `libero_real/` inside that directory.

From the repository root:

```bash
bash workflows/pi0_post_training/install.sh
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash workflows/pi0_post_training/finetune_metaworld.sh my_metaworld_run
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash workflows/pi0_post_training/finetune_libero.sh my_libero_run
```

Our paper runs use **8 H20 GPUs**. The launchers default to 30,001 iterations, with the released policy saved at step 30,000. Add `--dry-run` after the run name to inspect settings. `--base-params` selects the policy initialization; by default this is the upstream π0 base checkpoint. ALAM weights are selected independently with `--alam-checkpoint`.

## Fine-tuning on your own dataset

1. Convert demonstrations to the LeRobot format used by the existing loaders, under `<dataset-home>/<repo_id>/`. Include RGB observations, robot state, actions, and task instructions; preserve episode boundaries for temporal sampling.
2. In [training/config.py](../../evaluation/src/openpi/training/config.py), start from `phy_libero_full_finetune` or `phy_metaworld_full_finetune` and register a new config such as `my_robot_alam`. Adapt the data transforms for your camera keys, state/action dimensions, units, and absolute/delta action convention. Set the image/action sequence keys and training horizon for your robot.
3. Add a matching `my_robot` entry to `SPECS` in [train.py](train.py), following the existing entries: config name, dataset `repo_id`, raw/effective horizons, ALAM checkpoint, and reference policy path. The `policy` entry supplies default assets and must point to an existing checkpoint; `--base-params` controls which policy weights initialize training. New environments require these registrations before using the command below.
4. Compute normalization statistics from your new dataset using [compute_norm_stats.py](../../evaluation/scripts/compute_norm_stats.py), then pass the resulting assets directory to training. It must contain `<repo_id>/norm_stats.json`.

After registering `my_robot_alam` and `my_robot`, run from the repository root:

```bash
HF_LEROBOT_HOME=/path/to/lerobot \
  .venvs/pi0/bin/python evaluation/scripts/compute_norm_stats.py \
  --config-name my_robot_alam

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  .venvs/pi0/bin/python workflows/pi0_post_training/train.py my_robot \
  --exp-name my_robot_run \
  --dataset-home /path/to/lerobot \
  --assets-dir /path/to/generated/assets \
  --alam-checkpoint /path/to/pretrained_alam \
  --checkpoint-root outputs/pi0/checkpoints
```

To initialize from compatible post-trained policy weights, add `--base-params /path/to/policy_checkpoint/params` and use the ALAM tokenizer that was paired with that policy. Set `--fsdp-devices` and `--batch-size` for your available GPUs. Evaluation on a new robot also requires an environment client and matching server configuration.
