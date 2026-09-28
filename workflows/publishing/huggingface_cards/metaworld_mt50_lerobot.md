# MetaWorld MT50 LeRobot post-training data

Repository ID: `Mark-ZJTang/metaworld_mt50`.

LeRobot v2-format downstream fine-tuning dataset with 2,500 episodes, 204,806
frames, and 49 task IDs. This is a training dataset; MetaWorld MT50 simulator
evaluation itself does not read it.

- Split: `train` (`0:2500`), 80 FPS, three chunks.
- `observation.image`: image, `[3, 480, 480]`.
- `observation.state`: float32, `[4]`.
- `observation.environment_state`: float32, `[39]`.
- `action`: float32, `[4]` (`x`, `y`, `z`, `gripper`).
- Per-frame reward/success, task, episode, frame, and global indices are retained.

Restore it to the ALAM repository's loader-compatible relative path and verify
both downstream datasets with:

```bash
.venvs/publish/bin/python workflows/publishing/download_huggingface.py \
  --artifact metaworld_mt50_lerobot
.venvs/pi0/bin/python workflows/pi0_post_training/tests/test_datasets.py
```

The resulting root is `data/lerobot/metaworld_mt50`; the public fine-tuning
wrapper supplies its parent as `HF_LEROBOT_HOME`.

Before publication, verify the provenance and redistribution rights of generated
trajectories and bundled task assets, then add a license and generation details.
