# ALAM pretraining

We learn ALAM latent actions from **Mix-11: 10 OXE video sources + CALVIN**, without action labels. The [released pretrained weights](https://huggingface.co/Mark-ZJTang/alam_pretrain) come from this stage; downstream policy training uses the frozen encoder.

## Data mixture

The following identifiers and relative sampling weights match [alam_pretrain.yaml](../../configs/lam/alam_pretrain.yaml). OXE sources use frame-skip 5; CALVIN uses 10.

| Dataset | Config identifier | Weight |
| --- | --- | ---: |
| RT-1 | `fractal20220817_data` | 150 |
| BridgeData-V2 | `bridge` | 50 |
| TACO-Play | `taco_play` | 5 |
| JaCo-Play | `jaco_play` | 20 |
| Berkeley Cable Routing | `berkeley_cable_routing` | 20 |
| RoboTurk | `roboturk` | 10 |
| NYU Door-Opening | `nyu_door_opening_surprising_effectiveness` | 5 |
| VIOLA | `viola` | 3 |
| Berkeley AutoLab UR5 | `berkeley_autolab_ur5` | 5 |
| TOTO | `toto` | 5 |
| CALVIN ABC→D | `calvin_dataset` | 200 |

Download [OXE](https://github.com/google-deepmind/open_x_embodiment) and [CALVIN](https://github.com/mees/calvin/blob/main/dataset/README.md), then prepare videos and metadata using the [data preprocessing scripts](../../scripts/data_preprocessing). These scripts require source/output paths to be set for your machine. Set `ALAM_CALVIN_ROOT` and `ALAM_OXE_VIDEO_ROOT` in `.env`:

```text
data/alam/
├── calvin/{training,validation}/  # NPZ episodes + npz_metadata.json
└── oxe_videos/<dataset>/<view>/   # Episode videos + video_metadata.json
```

Views are `rgb_static` for TACO-Play, `front_rgb` for RoboTurk, `agentview_rgb` for VIOLA, and `image` for the other seven OXE sources.

## Training

From the repository root:

```bash
bash workflows/alam_pretraining/install.sh
bash workflows/alam_pretraining/pretrain.sh
```

Use `pretrain.sh --dry-run` to inspect resolved paths. Our paper-scale run used **128 H20 GPUs**. Multi-node launches set `ALAM_NUM_MACHINES`, total `ALAM_NUM_PROCESSES`, `MACHINE_RANK`, `MAIN_PROCESS_IP`, and `MAIN_PROCESS_PORT` on each node.

To adapt ALAM to your own videos, copy the training config, update `dataset_config` to match your video layout, and use `--config`, `--resume-checkpoint`, and `--output-dir` with `pretrain.sh`. For downstream robot-policy adaptation, see [post-training on a new dataset](../pi0_post_training/README.md#fine-tuning-on-your-own-dataset).

To check a downloaded tokenizer or run CALVIN validation:

```bash
bash workflows/alam_pretraining/evaluate_checkpoint.sh cpu
bash workflows/alam_pretraining/evaluate_validation.sh --gpu 0 --samples 10
```

Use these workflows rather than invoking the preserved `train_lam.py` directly.

See [Latent-action evaluation](LATENT_ACTION_EVALUATION.md) for a brief overview of additivity, reversibility, and reconstruction metrics.
