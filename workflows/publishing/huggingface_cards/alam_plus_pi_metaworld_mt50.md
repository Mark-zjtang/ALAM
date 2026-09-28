---
tags:
- robotics
- metaworld
- pi0
- alam
library_name: jax
---

# ALAM + pi0 post-trained for MetaWorld MT50

Repository ID: `Mark-ZJTang/alam_plus_pi_metaworld_mt50`.

Inference-only step-30,000 Orbax checkpoint for the released MetaWorld MT50
evaluation. The open-source default uses raw action horizon 6, effective
inference horizon 5, replan step 5, and camera `corner2` at absolute MuJoCo
position `(0.71, 0.075, 0.70)`.

The public full post-training resource contract is 8 GPUs. Inference and the
documented simulator acceptance use one sufficiently large idle GPU.

The checkpoint includes `params`, `_CHECKPOINT_METADATA`, and
`assets/metaworld_mt50/norm_stats.json`; optimizer `train_state` is intentionally
excluded. The matching external ALAM tokenizer is
`metaworld_epoch19_step58216` from the ALAM tokenizer model repository.

Verify all files against `evaluation/WEIGHTS_MANIFEST.sha256` in the GitHub code
release. License metadata must be completed before public publication.

From the matching GitHub code checkout:

```bash
.venvs/publish/bin/python workflows/publishing/download_huggingface.py \
  --artifact alam_pretrain \
  --artifact alam_plus_pi_metaworld_mt50
bash workflows/metaworld_evaluation/evaluate_mt50.sh --gpu 0
```

The 50-task protocol uses 10 episodes per task. A completed x=0.71 evaluation
observed 432/500 episode successes (86.4% episode-weighted) and 85.366883% after
macro averaging Easy/Medium/Hard/Very-Hard groups. Fresh runs may differ.
