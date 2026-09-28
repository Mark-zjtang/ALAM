# ALAM: Algebraically Consistent Latent Action Model for Vision-Language-Action Models

[![arXiv](https://img.shields.io/badge/arXiv-2605.10819-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2605.10819)[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE.txt)![Apache 2.0 License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)[![Python 3.8](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/release/python-380/)


![ALAM latent-action learning and algebraic consistency](figures/alam_framework.png)

*ALAM latent-action learning.*

![ALAM and VLA policy framework](figures/overall_framework.png)

*ALAM + π0 framework.*

ALAM learns structured latent actions from video and uses them to improve vision-language-action policies. This repository provides ALAM pretraining and ALAM + π0 training and evaluation for MetaWorld MT50 and LIBERO.

## 📰 News

- **September 25, 2026:** ALAM was accepted at NeurIPS 2026.

## 🛠️ Environment setup

From the repository root:

```bash
cp .env.example .env
bash workflows/install/install_environments.sh --components all
```

Edit `.env` for your local paths. To install only one stage, see the [pretraining](workflows/alam_pretraining/README.md), [post-training](workflows/pi0_post_training/README.md), [MetaWorld](workflows/metaworld_evaluation/README.md), or [LIBERO](workflows/libero_evaluation/README.md) guide.

## 📦 Datasets and model weights

Training datasets are not bundled. Download them from their original projects; training requires the converted layouts in `.env.example`.

| Dataset | Original source |
| --- | --- |
| CALVIN ABC→D | [CALVIN](https://github.com/mees/calvin/blob/main/dataset/README.md) |
| Open X-Embodiment | [Open X-Embodiment](https://github.com/google-deepmind/open_x_embodiment) |
| MetaWorld | [MetaWorld](https://github.com/Farama-Foundation/Metaworld) |
| LIBERO | [LIBERO](https://libero-project.github.io/datasets) |

| Released weights | Hugging Face |
| --- | --- |
| MetaWorld and LIBERO ALAM tokenizers | [🤗 alam_pretrain](https://huggingface.co/Mark-ZJTang/alam_pretrain) |
| MetaWorld π0 policy | [🤗 alam_plus_pi_metaworld_mt50](https://huggingface.co/Mark-ZJTang/alam_plus_pi_metaworld_mt50) |
| LIBERO π0 policy, shared by four suites | [🤗 alam_plus_pi_libero](https://huggingface.co/Mark-ZJTang/alam_plus_pi_libero) |

```bash
bash workflows/publishing/download_release.sh --models-only
```

Weights are placed under `evaluation/checkpoints/`; exact paths and hashes are in the [model manifest](workflows/publishing/huggingface_manifest.json). Evaluation does not need the training datasets.

## 🏋️ Training

```bash
# ALAM tokenizer pretraining
bash workflows/alam_pretraining/pretrain.sh

# ALAM + π0 post-training (8 GPUs)
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash workflows/pi0_post_training/finetune_metaworld.sh my_metaworld_run
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash workflows/pi0_post_training/finetune_libero.sh my_libero_run
```

See the [pretraining](workflows/alam_pretraining/README.md) and [post-training](workflows/pi0_post_training/README.md) guides for required data paths and options.

## 🎯 Evaluation

| Benchmark | Policy | Command |
| --- | --- | --- |
| MetaWorld MT50 | MetaWorld π0 | `bash workflows/metaworld_evaluation/evaluate_mt50.sh --gpu 0` |
| LIBERO Spatial | LIBERO π0 | `bash workflows/libero_evaluation/evaluate_libero.sh spatial --gpu 0` |
| LIBERO Object | LIBERO π0 | `bash workflows/libero_evaluation/evaluate_libero.sh object --gpu 0` |
| LIBERO Goal | LIBERO π0 | `bash workflows/libero_evaluation/evaluate_libero.sh goal --gpu 0` |
| LIBERO Long | LIBERO π0 | `bash workflows/libero_evaluation/evaluate_libero.sh long --gpu 0` |

MetaWorld runs 50 tasks × 10 episodes; each LIBERO suite runs 10 tasks × 50 trials. The [MetaWorld](workflows/metaworld_evaluation/README.md) and [LIBERO](workflows/libero_evaluation/README.md) guides describe the portable commands. [Selected completed local results](workflows/best_observed_evaluation/README.md) are historical observations, not guaranteed scores for a new run.

ALAM-authored code is available under [MIT or Apache-2.0](LICENSE), at your option. Bundled third-party material retains its own license.

## 📚 Citation

```bibtex
@misc{tang2026alamalgebraicallyconsistentlatent,
  title={ALAM: Algebraically Consistent Latent Action Model for Vision-Language-Action Models},
  author={Zuojin Tang and Haoyun Liu and Xinyuan Chang and Changjie Wu and Dongjie Huo and Yandan Yang and Bin Liu and Zhejia Cai and Feng Xiong and Mu Xu and jiachen Luo and De Ma and Zhiheng Ma and Gang Pan},
  year={2026},
  eprint={2605.10819},
  archivePrefix={arXiv},
  primaryClass={cs.RO},
  url={https://arxiv.org/abs/2605.10819}
}
```
