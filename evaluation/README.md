# ALAM + π0 downstream source

This directory contains the preserved π0 integration, simulator clients, and assets. Run the portable workflows from the repository root:

| Task | Guide |
| --- | --- |
| Fine-tuning | [ALAM + π0 post-training](../workflows/pi0_post_training/README.md) |
| MetaWorld MT50 | [MetaWorld evaluation](../workflows/metaworld_evaluation/README.md) |
| LIBERO | [LIBERO evaluation](../workflows/libero_evaluation/README.md) |

Downloaded weights belong under `checkpoints/`; use the [download manifest](../workflows/publishing/huggingface_manifest.json) for paths. The three default artifacts are one shared Mix-11-pretrained ALAM tokenizer (`alam_pretrain_latent_action_tokenizer`) and one post-trained π0 policy for each benchmark.

The [selected completed evaluations](../workflows/best_observed_evaluation/README.md) are historical observations, separate from new portable runs. In particular, the retained LIBERO Spatial client log does not independently prove the loaded server checkpoint.

To check bundled source hashes and the downloaded checkpoint layout from this directory:

```bash
sha256sum -c SOURCE_MANIFEST.sha256
python3 ../workflows/audit/validate.py --require-weights
```
