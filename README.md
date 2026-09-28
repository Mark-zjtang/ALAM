# ALAM: Algebraically Consistent Latent Action Model for Vision-Language-Action Models

Accepted at **NeurIPS 2026**.

This repository contains ALAM pretraining and ALAM + π0 training and evaluation for MetaWorld MT50 and LIBERO.

## Get started

Run from the repository root:

```bash
cp .env.example .env
bash workflows/install/install_environments.sh --components publish
bash workflows/publishing/download_release.sh --models-only
```

Edit `.env` if your data, model, or output paths differ. Install only the environments needed for your task; each module below has its own commands.

| Module | Guide |
| --- | --- |
| ALAM pretraining | [Pretrain and validate the tokenizer](workflows/alam_pretraining/README.md) |
| ALAM + π0 post-training | [Fine-tune MetaWorld or LIBERO](workflows/pi0_post_training/README.md) |
| MetaWorld MT50 | [Evaluate the MetaWorld policy](workflows/metaworld_evaluation/README.md) |
| LIBERO | [Evaluate the four LIBERO suites](workflows/libero_evaluation/README.md) |
| Models and publication | [Download or publish model weights](workflows/publishing/README.md) |

## Model weights

| Model | Hugging Face |
| --- | --- |
| MetaWorld and LIBERO ALAM tokenizers | [Mark-ZJTang/alam_pretrain](https://huggingface.co/Mark-ZJTang/alam_pretrain) |
| MetaWorld π0 policy | [Mark-ZJTang/alam_plus_pi_metaworld_mt50](https://huggingface.co/Mark-ZJTang/alam_plus_pi_metaworld_mt50) |
| LIBERO π0 policy, shared by four suites | [Mark-ZJTang/alam_plus_pi_libero](https://huggingface.co/Mark-ZJTang/alam_plus_pi_libero) |

Model files are downloaded to `evaluation/checkpoints/`. Their paths and hashes are recorded in [the model manifest](workflows/publishing/huggingface_manifest.json).

## Dataset sources

Datasets are **not included**. Obtain them from the original projects:

| Training input | Source |
| --- | --- |
| CALVIN ABC→D | [CALVIN](https://github.com/mees/calvin/blob/main/dataset/README.md) |
| Open X-Embodiment | [Open X-Embodiment](https://github.com/google-deepmind/open_x_embodiment) |
| MetaWorld | [MetaWorld](https://github.com/Farama-Foundation/Metaworld) |
| LIBERO | [LIBERO](https://libero-project.github.io/datasets) |

Training requires the converted layouts specified in `.env.example`; upstream downloads alone are not the historical converted datasets. Policy evaluation does not require training data.

## Evaluation results

The [selected completed local evaluations](workflows/best_observed_evaluation/README.md) record one MetaWorld MT50 result and four separate LIBERO results. They are historical observations, not scores guaranteed by a fresh run. Those exact launchers require frozen local assets; use the portable evaluation guides above on a new machine.

## License

ALAM-authored code is available under [MIT or Apache-2.0](LICENSE), at your option. Bundled third-party material retains its own license.
