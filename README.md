# ALAM: Algebraically Consistent Latent Action Model for Vision-Language-Action Models 

[![arXiv](https://img.shields.io/badge/arXiv-2605.10819-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2605.10819)
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

We use separate environments for ALAM pretraining and π0 post-training; the MetaWorld and LIBERO installers each add their simulator environment alongside the shared π0 policy server.

From the repository root, copy `.env.example` to `.env` and install the stages you need:

```bash
cp -n .env.example .env

bash workflows/alam_pretraining/install.sh       # ALAM pretraining
bash workflows/pi0_post_training/install.sh     # π0 post-training
bash workflows/metaworld_evaluation/install.sh  # MetaWorld evaluation
bash workflows/libero_evaluation/install.sh     # LIBERO evaluation
```

Set dataset directories in `.env`; relative paths start at the repository root.

## 📦 Datasets and model weights

We pretrain ALAM on **Mix-11: 10 Open X-Embodiment (OXE) datasets + CALVIN**, using video without action labels. We then freeze the pretrained ALAM encoder and post-train π0 on MetaWorld or LIBERO demonstrations.

| Training data | Stage | Download |
| --- | --- | --- |
| CALVIN ABC→D | ALAM pretraining | [CALVIN](https://github.com/mees/calvin/blob/main/dataset/README.md) |
| OXE, 10 selected datasets | ALAM pretraining | [Open X-Embodiment](https://github.com/google-deepmind/open_x_embodiment) |
| MetaWorld demonstrations | π0 post-training | [🤗 metaworld_mt50](https://huggingface.co/datasets/Mark-ZJTang/metaworld_mt50) |
| LIBERO demonstrations | π0 post-training | [🤗 libero](https://huggingface.co/datasets/Mark-ZJTang/libero_real) |

The OXE selection is **RT-1 (fractal20220817), BridgeData-V2, TACO-Play, JaCo-Play, Berkeley Cable Routing, RoboTurk, NYU Door-Opening, VIOLA, Berkeley AutoLab UR5, and TOTO**. Dataset identifiers and sampling weights are listed in the [pretraining guide](workflows/alam_pretraining/README.md). Download datasets separately and prepare the loader-compatible layouts described there.

**Pretrained weights**

| Model | How we train it | Paper-scale resources | Download |
| --- | --- | --- | --- |
| ALAM latent-action tokenizer | Video pretraining on Mix-11 (10 OXE sources + CALVIN) | 128 × H20 GPUs | [🤗 alam_pretrain](https://huggingface.co/Mark-ZJTang/alam_pretrain) |

**Post-trained weights**

| Model | How we train it | Paper-scale resources | Download |
| --- | --- | --- | --- |
| ALAM + π0, MetaWorld | Initialize from π0 base; freeze the Mix-11 ALAM encoder; train on MetaWorld demonstrations for 30k steps | 8 × H20 GPUs | [🤗 alam_plus_pi_metaworld_mt50](https://huggingface.co/Mark-ZJTang/alam_plus_pi_metaworld_mt50) |
| ALAM + π0, LIBERO | Initialize from π0 base; freeze the Mix-11 ALAM encoder; train on LIBERO demonstrations for 30k steps; one policy serves all four suites | 8 × H20 GPUs | [🤗 alam_plus_pi_libero](https://huggingface.co/Mark-ZJTang/alam_plus_pi_libero) |

The two checkpoints in `alam_pretrain` come from the video-pretraining pipeline; their directory names identify which downstream policy uses them. GPU counts refer to our [paper experiments](https://arxiv.org/pdf/2605.10819).

```bash
bash workflows/install/install_environments.sh --components publish
bash workflows/publishing/download_release.sh --models-only
```

Weights are placed under `evaluation/checkpoints/`; download paths are in the [manifest](workflows/publishing/huggingface_manifest.json) and hashes in the [weights manifest](evaluation/WEIGHTS_MANIFEST.sha256). Evaluation does not need the training datasets.

## 🗂️ Repository layout

```text
ALAM/
├── Algebraic_latent_action_model/  # ALAM tokenizer and trainer
├── configs/lam/                   # Mix-11 pretraining configuration
├── workflows/
│   ├── alam_pretraining/          # Video pretraining
│   ├── pi0_post_training/         # Policy post-training and serving
│   ├── metaworld_evaluation/      # MetaWorld evaluation
│   └── libero_evaluation/         # LIBERO evaluation
├── evaluation/                   # π0 implementation and simulators
│   └── checkpoints/              # Downloaded weights
│       ├── alam/                 # Mix-11 pretrained ALAM tokenizers
│       ├── metaworld/            # Post-trained MetaWorld policy
│       └── libero/               # Post-trained LIBERO policy
├── data/                         # User-downloaded training data
│   ├── alam/{calvin,oxe_videos}/  # Pretraining data
│   └── lerobot/                  # Downstream demonstrations
└── outputs/                      # Training checkpoints and evaluation results
```

## 🏋️ Training

```bash
# ALAM tokenizer pretraining (128 H20 GPUs for the paper-scale run)
bash workflows/alam_pretraining/pretrain.sh

# ALAM + π0 post-training (8 GPUs)
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash workflows/pi0_post_training/finetune_metaworld.sh my_metaworld_run
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash workflows/pi0_post_training/finetune_libero.sh my_libero_run
```

See the [pretraining](workflows/alam_pretraining/README.md) and [post-training](workflows/pi0_post_training/README.md) guides for data layout and multi-node settings.

For a new robot environment or dataset, convert demonstrations to LeRobot format, adapt observation/action mappings, and compute dataset-specific normalization statistics. Follow [Fine-tuning on your own dataset](workflows/pi0_post_training/README.md#fine-tuning-on-your-own-dataset) to register the training configuration and launch a new run.

## 🎯 Evaluation

- [MetaWorld MT50: command, protocol, and paper results](workflows/metaworld_evaluation/README.md)
- [LIBERO: four suite commands, protocol, and paper results](workflows/libero_evaluation/README.md)

Use `--policy-checkpoint PATH` to evaluate your own ALAM + π0 weights; each guide includes a custom-checkpoint example.

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
