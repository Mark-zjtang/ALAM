---
tags:
- robotics
- latent-action-model
- alam
library_name: pytorch
---

# ALAM pretrained tokenizer checkpoints

Repository ID: `Mark-ZJTang/alam_pretrain`.

This repository contains the two ALAM v3 tokenizer checkpoints used by the
released downstream evaluations. Both use 7 latent-action slots, a 256-entry
codebook, and 128-dimensional latent vectors.

| Directory | Downstream use | Training epoch/step |
| --- | --- | --- |
| `metaworld_epoch19_step58216` | MetaWorld MT50 | epoch 19, step 58,216 |
| `libero_epoch16_step49024` | LIBERO Table 9 shared-checkpoint configuration | epoch 16, step 49,024 |

Each directory contains `config.yaml` and `pytorch_model.bin`. Verify the files
against `evaluation/WEIGHTS_MANIFEST.sha256` in the GitHub code release. License
metadata must be completed by the copyright owner before public publication.

From the matching GitHub code checkout, restore both checkpoints and run strict
CPU structure checks with:

```bash
.venvs/publish/bin/python workflows/publishing/download_huggingface.py \
  --artifact alam_pretrain
bash workflows/alam_pretraining/evaluate_checkpoint.sh cpu
bash workflows/alam_pretraining/evaluate_checkpoint.sh cpu \
  --checkpoint evaluation/checkpoints/alam/libero_epoch16_step49024
```

Use `evaluate_checkpoint.sh cuda` for real encoder execution; the preserved
encoder has a historical internal CUDA device assumption.

The project-owner resource specification for full ALAM pretraining is 128
NVIDIA H20 GPUs (one process per GPU), with per-GPU batch 32 and gradient
accumulation 2. Historical directory labels containing `64gpu` are provenance
names, not the public resource contract.
