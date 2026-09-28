# ALAM + π0 post-training

Set `HF_LEROBOT_HOME` in the root `.env`. Training expects converted LeRobot datasets under `metaworld_mt50/` and `libero_real/` inside that directory.

From the repository root:

```bash
bash workflows/pi0_post_training/install.sh
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash workflows/pi0_post_training/finetune_metaworld.sh my_metaworld_run
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash workflows/pi0_post_training/finetune_libero.sh my_libero_run
```

Each full run uses eight GPUs and defaults to 30,001 steps. Add `--dry-run` after the run name to inspect settings before training. The launchers use the preserved π0 trainer and the ALAM checkpoints downloaded from Hugging Face.
