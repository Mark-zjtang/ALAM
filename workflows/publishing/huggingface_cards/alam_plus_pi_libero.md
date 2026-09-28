---
tags:
- robotics
- libero
- pi0
- alam
library_name: jax
---

# ALAM + pi0 post-trained for LIBERO

Repository ID: `Mark-ZJTang/alam_plus_pi_libero`.

Inference-only step-30,000 Orbax checkpoint used as the shared checkpoint for the
LIBERO Table 9 release configuration. Training effective horizon is 20. At
inference, Spatial/Object use H=14 and Goal/Long use H=18; replan steps are 5,
10, 7, and 12 respectively.

The public full post-training resource contract is 8 GPUs. Inference and the
documented CUDA/EGL acceptance use one sufficiently large idle GPU.

The checkpoint includes `params`, `_CHECKPOINT_METADATA`, and
`assets/libero_real/norm_stats.json`; optimizer `train_state` is intentionally
excluded. The matching ALAM tokenizer is `libero_epoch16_step49024`.

The available historical evidence supports the shared-checkpoint release
configuration, but the Spatial server identity was not recorded in its client
log; this provenance limitation is documented in the GitHub release. License
metadata must be completed before public publication.

From the matching GitHub code checkout:

```bash
.venvs/publish/bin/python workflows/publishing/download_huggingface.py \
  --artifact alam_pretrain \
  --artifact alam_plus_pi_libero
bash workflows/libero_evaluation/evaluate_all_suites.sh --gpu 0
```

Each suite uses 10 tasks x 50 trials. Recorded successes are Spatial 496/500
(99.2%), Object 498/500 (99.6%), Goal 495/500 (99.0%), and Long 472/500
(94.4%), for a 98.05% mean reported as 98.1%.
