# LIBERO LeRobot post-training data

Repository ID: `Mark-ZJTang/libero_real`.

LeRobot v2-format downstream fine-tuning dataset with 1,693 episodes, 273,465
frames, and 40 task IDs. It contains agent-view and wrist-view images, state, and
7D actions. LIBERO simulator evaluation uses bundled BDDL/init/assets and does
not read these demonstrations.

- Split: `train` (`0:1693`), 10 FPS, two chunks.
- `image` and `wrist_image`: image, `[256, 256, 3]` each.
- `state`: float32, `[8]`.
- `actions`: float32, `[7]`.
- Task, episode, frame, and global indices are retained.

Restore it to the ALAM repository's loader-compatible relative path and verify
both downstream datasets with:

```bash
.venvs/publish/bin/python workflows/publishing/download_huggingface.py \
  --artifact libero_real_lerobot
.venvs/pi0/bin/python workflows/pi0_post_training/tests/test_datasets.py
```

The resulting root is `data/lerobot/libero_real`; the public fine-tuning wrapper
supplies its parent as `HF_LEROBOT_HOME`.

Before publication, verify LIBERO dataset redistribution rights and upstream
attribution, then add the correct license and data-generation details.
