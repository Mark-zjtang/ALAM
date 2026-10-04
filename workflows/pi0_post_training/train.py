#!/usr/bin/env python3
"""Repository-relative launcher for ALAM + pi0 downstream fine-tuning.

The preserved training implementation and named configs remain unchanged. This
launcher applies only runtime path, dataset, horizon, and run-control overrides.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import os
from pathlib import Path
import sys

WORKFLOWS_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKFLOWS_ROOT))

from common.paths import REPO_ROOT, configure_runtime_environment, repo_path, require_dir
from common.prepare_lam_checkpoint import prepare


SPECS = {
    "metaworld": {
        "config": "phy_metaworld_full_finetune",
        "repo_id": "metaworld_mt50",
        "raw_horizon": 6,
        "effective_horizon": 5,
        "alam": "evaluation/checkpoints/alam/alam_pretrain_latent_action_tokenizer",
        "policy": "evaluation/checkpoints/metaworld/alam_plus_pi_metaworld_mt50_step30000",
    },
    "libero": {
        "config": "phy_libero_full_finetune",
        "repo_id": "libero_real",
        "raw_horizon": 21,
        "effective_horizon": 20,
        "alam": "evaluation/checkpoints/alam/alam_pretrain_latent_action_tokenizer",
        "policy": "evaluation/checkpoints/libero/alam_plus_pi_libero_step30000",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("benchmark", choices=tuple(SPECS))
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--dataset-home", help="Parent containing metaworld_mt50/ and libero_real/")
    parser.add_argument("--alam-checkpoint")
    parser.add_argument("--assets-dir", help="Directory containing <repo_id>/norm_stats.json")
    parser.add_argument("--base-params", help="Override the upstream pi0 base params path or URI")
    parser.add_argument("--checkpoint-root")
    parser.add_argument("--steps", type=int, default=30_001)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--fsdp-devices", type=int, default=8)
    parser.add_argument(
        "--episode-limit",
        type=int,
        help="Use the first N real LeRobot episodes (smoke/acceptance runs only)",
    )
    parser.add_argument("--wandb", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true")
    mode.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_preserved_trainer():
    path = REPO_ROOT / "evaluation" / "scripts" / "train.py"
    spec = importlib.util.spec_from_file_location("alam_release_preserved_train", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load preserved trainer: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.main


def limit_lerobot_episodes(limit: int) -> None:
    """Limit only this process to a small, real LeRobot episode subset."""
    if limit < 1:
        raise ValueError(f"--episode-limit must be positive, found: {limit}")

    from lerobot.common.datasets import lerobot_dataset
    from openpi.training import data_loader

    original_create_dataset = data_loader.create_torch_dataset
    selected_episodes = list(range(limit))

    def create_limited_dataset(*args, **kwargs):
        # The preserved loader does not expose LeRobot's `episodes` argument.
        # Patch its constructor only while the dataset is synchronously built,
        # then restore the real class before spawned workers pickle the dataset.
        original_constructor = lerobot_dataset.LeRobotDataset

        def limited_constructor(*constructor_args, **constructor_kwargs):
            if constructor_kwargs.get("episodes") is not None:
                raise ValueError("LeRobot episodes were already selected before --episode-limit")
            constructor_kwargs["episodes"] = selected_episodes
            return original_constructor(*constructor_args, **constructor_kwargs)

        lerobot_dataset.LeRobotDataset = limited_constructor
        try:
            return original_create_dataset(*args, **kwargs)
        finally:
            lerobot_dataset.LeRobotDataset = original_constructor

    data_loader.create_torch_dataset = create_limited_dataset


def main() -> None:
    args = parse_args()
    os.chdir(REPO_ROOT)
    configure_runtime_environment()
    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(REPO_ROOT / "evaluation" / "src"))

    selected = dict(SPECS[args.benchmark])
    dataset_home = repo_path(args.dataset_home or os.environ["HF_LEROBOT_HOME"])
    dataset_dir = dataset_home / str(selected["repo_id"])
    alam_source = require_dir(
        repo_path(args.alam_checkpoint or str(selected["alam"])),
        "ALAM tokenizer checkpoint",
    )
    policy_checkpoint = require_dir(repo_path(str(selected["policy"])), "packaged policy checkpoint")
    assets_dir = repo_path(args.assets_dir) if args.assets_dir else policy_checkpoint / "assets"
    checkpoint_root = repo_path(
        args.checkpoint_root or os.path.join(os.environ["ALAM_OUTPUT_ROOT"], "pi0", "checkpoints")
    )

    summary = {
        **selected,
        "dataset_home": str(dataset_home),
        "dataset_dir": str(dataset_dir),
        "dataset_exists": dataset_dir.is_dir(),
        "alam_checkpoint": str(alam_source),
        "norm_stats": str(assets_dir / str(selected["repo_id"]) / "norm_stats.json"),
        "checkpoint_root": str(checkpoint_root),
        "exp_name": args.exp_name,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "fsdp_devices": args.fsdp_devices,
        "episode_limit": args.episode_limit,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.dry_run:
        return
    require_dir(dataset_dir, f"LeRobot dataset {selected['repo_id']}")
    require_dir(assets_dir, "normalization assets")

    os.environ["HF_LEROBOT_HOME"] = str(dataset_home)
    portable_lam = prepare(alam_source)

    from openpi.training import config as openpi_config
    from openpi.training import weight_loaders

    config = openpi_config.get_config(str(selected["config"]))
    model = dataclasses.replace(
        config.model,
        action_horizon=int(selected["raw_horizon"]),
        phy_lam_ckpt=str(portable_lam),
    )
    data = dataclasses.replace(
        config.data,
        repo_id=str(selected["repo_id"]),
        assets=openpi_config.AssetsConfig(
            assets_dir=str(assets_dir),
            asset_id=str(selected["repo_id"]),
        ),
    )
    weight_loader = config.weight_loader
    if args.base_params:
        base_params = args.base_params if "://" in args.base_params else str(repo_path(args.base_params))
        weight_loader = weight_loaders.CheckpointWeightLoader(base_params)

    config = dataclasses.replace(
        config,
        exp_name=args.exp_name,
        model=model,
        data=data,
        weight_loader=weight_loader,
        checkpoint_base_dir=str(checkpoint_root),
        num_train_steps=args.steps,
        batch_size=args.batch_size,
        num_workers=args.workers,
        fsdp_devices=args.fsdp_devices,
        wandb_enabled=args.wandb,
        resume=args.resume,
        overwrite=args.overwrite,
    )
    if args.episode_limit is not None:
        limit_lerobot_episodes(args.episode_limit)
    load_preserved_trainer()(config)


if __name__ == "__main__":
    main()
