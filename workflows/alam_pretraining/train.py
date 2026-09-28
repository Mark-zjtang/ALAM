#!/usr/bin/env python3
"""Repository-relative launcher for the preserved ALAM training implementation."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

WORKFLOWS_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKFLOWS_ROOT))

from common.paths import REPO_ROOT, configure_runtime_environment


PORTABLE_TARGET = (
    "Algebraic_latent_action_model.latent_action_model.models."
    "ctl_latent_action_tokenizer_v3_portable.LatentActionTokenizer"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/lam/alam_pretrain.yaml")
    parser.add_argument("--calvin-root")
    parser.add_argument("--oxe-video-root")
    parser.add_argument("--output-dir")
    parser.add_argument("--resume-checkpoint", default=None)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.chdir(REPO_ROOT)
    configure_runtime_environment()
    sys.path.insert(0, str(REPO_ROOT))

    # Import the public package once so its compatibility alias is available to
    # the byte-preserved trainer and data-loader imports.
    import Algebraic_latent_action_model  # noqa: F401

    # Resolve defaults only after .env has been loaded. CLI values always win.
    calvin_root = args.calvin_root or os.environ.get("ALAM_CALVIN_ROOT", "data/alam/calvin")
    oxe_video_root = args.oxe_video_root or os.environ.get("ALAM_OXE_VIDEO_ROOT", "data/alam/oxe_videos")
    output_dir = args.output_dir or os.path.join(os.environ["ALAM_OUTPUT_ROOT"], "alam", "main")

    from omegaconf import OmegaConf

    config = OmegaConf.load(args.config)
    config.latent_action_model_config._target_ = PORTABLE_TARGET
    dataset = config.dataset_config
    if dataset.data_type == "mix":
        dataset.video_dir = oxe_video_root
        for subdataset in dataset.sub_data_configs:
            if "npz_dir" in subdataset or subdataset.get("data_name") == "calvin_dataset":
                subdataset.npz_dir = calvin_root
    elif "npz_dir" in dataset:
        dataset.npz_dir = calvin_root
    config.training_config.save_path = output_dir
    config.training_config.resume_ckpt_path = args.resume_checkpoint
    if args.batch_size is not None:
        config.dataloader_config.bs_per_gpu = args.batch_size
    if args.workers is not None:
        config.dataloader_config.workers_per_gpu = args.workers
    if args.epochs is not None:
        config.training_config.num_epochs = args.epochs

    if args.dry_run:
        print(OmegaConf.to_yaml(config, resolve=True))
        return

    from train_lam import main as train

    train(config)


if __name__ == "__main__":
    main()
