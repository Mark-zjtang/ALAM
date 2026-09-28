# ALAM: Algebraically Consistent Latent Action Model for Vision-Language-Action Models 

[![arXiv](https://img.shields.io/badge/arXiv-2605.10819-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2605.10819)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97-Hugging_Face-ffd21e)](https://huggingface.co/Mark-ZJTang)
[![MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

![ALAM latent-action learning and algebraic consistency](figures/alam_framework.png)

*ALAM latent-action learning.*

![ALAM and VLA policy framework](figures/overall_framework.png)

*ALAM + π0 framework.*

ALAM learns structured latent actions from video and uses them to improve vision-language-action policies. This repository provides ALAM pretraining and ALAM + π0 training and evaluation for MetaWorld MT50 and LIBERO.

## 📰 News

- 🎉🎉🎉 **September 25, 2026:** ALAM was accepted at NeurIPS 2026!

## 🛠️ Environment setup

Run commands from the repository root. Create `.env` once, then set the dataset paths needed for your stage:

```bash
cp -n .env.example .env
```

A **local path** is a directory on your machine, not a web link. Relative paths in `.env` start at this repository's root: `HF_LEROBOT_HOME=data/lerobot` means `<repository>/data/lerobot`. Keep credentials out of `.env`.

Install only the environments you need:

| Stage | Install command | Environments created | Python |
| --- | --- | --- | --- |
| ALAM pretraining | `bash workflows/alam_pretraining/install.sh` | `.venvs/alam` | 3.10 |
| ALAM + π0 post-training | `bash workflows/pi0_post_training/install.sh` | `.venvs/pi0` | 3.11 |
| MetaWorld evaluation | `bash workflows/metaworld_evaluation/install.sh` | `.venvs/pi0`, `.venvs/metaworld` | 3.11 |
| LIBERO evaluation | `bash workflows/libero_evaluation/install.sh` | `.venvs/pi0`, `.venvs/libero` | 3.11 / 3.8 |

## 📦 Datasets and model weights

Training datasets are not bundled. Download the converted MetaWorld and LIBERO datasets from Hugging Face; obtain CALVIN and Open X-Embodiment from their projects. Set their local directories in `.env`.

| Dataset | Download |
| --- | --- |
| CALVIN ABC→D | [CALVIN](https://github.com/mees/calvin/blob/main/dataset/README.md) |
| Open X-Embodiment | [Open X-Embodiment](https://github.com/google-deepmind/open_x_embodiment) |
| MetaWorld | [🤗 metaworld_mt50](https://huggingface.co/datasets/Mark-ZJTang/metaworld_mt50) |
| LIBERO | [🤗 libero](https://huggingface.co/datasets/Mark-ZJTang/libero_real) |

| Released weights | Hugging Face |
| --- | --- |
| MetaWorld and LIBERO ALAM tokenizers | [🤗 alam_pretrain](https://huggingface.co/Mark-ZJTang/alam_pretrain) |
| MetaWorld π0 policy | [🤗 alam_plus_pi_metaworld_mt50](https://huggingface.co/Mark-ZJTang/alam_plus_pi_metaworld_mt50) |
| LIBERO π0 policy, shared by four suites | [🤗 alam_plus_pi_libero](https://huggingface.co/Mark-ZJTang/alam_plus_pi_libero) |

```bash
bash workflows/publishing/download_release.sh --models-only
```

Weights are placed under `evaluation/checkpoints/`; download paths are in the [manifest](workflows/publishing/huggingface_manifest.json) and hashes in the [weights manifest](evaluation/WEIGHTS_MANIFEST.sha256). Evaluation does not need the training datasets.

## 🏋️ Training

```bash
# ALAM tokenizer pretraining (128 H20 GPUs for the paper-scale run)
bash workflows/alam_pretraining/pretrain.sh

# ALAM + π0 post-training (8 GPUs)
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash workflows/pi0_post_training/finetune_metaworld.sh my_metaworld_run
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash workflows/pi0_post_training/finetune_libero.sh my_libero_run
```

See the [pretraining](workflows/alam_pretraining/README.md) and [post-training](workflows/pi0_post_training/README.md) guides for data layout and multi-node settings.

## 🎯 Evaluation

- [MetaWorld MT50: command, protocol, and paper results](workflows/metaworld_evaluation/README.md)
- [LIBERO: four suite commands, protocol, and paper results](workflows/libero_evaluation/README.md)

For model downloads and the maintainer's GitHub code-push command, see [Downloads and code publishing](workflows/publishing/README.md).

ALAM-authored code is available under [MIT or Apache-2.0](LICENSE), at your option. Bundled third-party material retains its own license.

## 📚 Citation

```bibtex
@article{tang2026alam,
  title={ALAM: Algebraically Consistent Latent Action Model for Vision-Language-Action Models},
  author={Tang, Zuojin and Liu, Haoyun and Chang, Xinyuan and Wu, Changjie and Huo, Dongjie and Yang, Yandan and Liu, Bin and Cai, Zhejia and Xiong, Feng and Xu, Mu and others},
  journal={arXiv preprint arXiv:2605.10819},
  year={2026}
}
```
