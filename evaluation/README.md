# ALAM + π0 downstream source

This directory contains the preserved π0 integration, simulator clients, and assets. Run the portable workflows from the repository root:

| Task | Guide |
| --- | --- |
| Fine-tuning | [ALAM + π0 post-training](../workflows/pi0_post_training/README.md) |
| MetaWorld MT50 | [MetaWorld evaluation](../workflows/metaworld_evaluation/README.md) |
| LIBERO | [LIBERO evaluation](../workflows/libero_evaluation/README.md) |

Downloaded weights belong under `checkpoints/`; use the [download manifest](../workflows/publishing/huggingface_manifest.json) for paths and [weights manifest](WEIGHTS_MANIFEST.sha256) for hashes. The four artifacts are two ALAM tokenizers and one policy for each benchmark.

The [selected completed evaluations](../workflows/best_observed_evaluation/README.md) are historical observations, separate from new portable runs. In particular, the retained LIBERO Spatial client log does not independently prove the loaded server checkpoint.

To check bundled source and downloaded model files from this directory:

```bash
sha256sum -c SOURCE_MANIFEST.sha256
sha256sum -c WEIGHTS_MANIFEST.sha256
```
