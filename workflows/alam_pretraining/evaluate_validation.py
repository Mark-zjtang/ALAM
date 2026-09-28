#!/usr/bin/env python3
"""Evaluate a released ALAM checkpoint on real CALVIN validation samples."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace


WORKFLOWS_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKFLOWS_ROOT))

from common.paths import REPO_ROOT, configure_runtime_environment, repo_path, require_dir, require_file


PORTABLE_TARGET = (
    "Algebraic_latent_action_model.latent_action_model.models."
    "ctl_latent_action_tokenizer_v3_portable.LatentActionTokenizer"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="evaluation/checkpoints/alam/metaworld_epoch19_step58216",
    )
    parser.add_argument("--calvin-root")
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--output", default="outputs/tests/alam_validation/summary.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.samples <= 0:
        raise ValueError("--samples must be positive")
    os.chdir(REPO_ROOT)
    configure_runtime_environment()
    sys.path.insert(0, str(REPO_ROOT))
    import Algebraic_latent_action_model  # noqa: F401

    calvin_value = args.calvin_root or os.environ.get("ALAM_CALVIN_ROOT", "data/alam/calvin")
    calvin_root = require_dir(repo_path(calvin_value), "CALVIN pretraining dataset")
    checkpoint = require_dir(repo_path(args.checkpoint), "ALAM checkpoint")
    config_path = require_file(checkpoint / "config.yaml", "ALAM checkpoint config")
    weights_path = require_file(checkpoint / "pytorch_model.bin", "ALAM checkpoint weights")

    import torch
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from torchvision.transforms.v2 import InterpolationMode, Resize

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for real ALAM validation evaluation")

    from Algebraic_latent_action_model.data.data_utils import load_dataset
    from Algebraic_latent_action_model.data.img_utils import get_rgb_preprocessor
    from Algebraic_latent_action_model.latent_action_model.trainers.ctl_latent_action_tokenizer_trainer import (
        LatentActionTokenizerTrainer,
    )

    resize = Resize([200, 200], interpolation=InterpolationMode.BICUBIC, antialias=True)
    dataset_config = {
        "data_type": "video_npz_3seq",
        "npz_dir": str(calvin_root),
        "skip_frame": 10,
        "rgb_shape": [200, 200],
        "rgb_preprocessor": resize,
    }
    _, validation = load_dataset(
        dataset_config,
        {
            "sequence_length": 1,
            "do_extract_future_frames": True,
            "do_extract_action": False,
        },
    )

    model_config = OmegaConf.load(config_path)
    model_config._target_ = PORTABLE_TARGET
    model = instantiate(model_config)
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or len(state) != 458:
        raise AssertionError(f"Expected 458 checkpoint tensors, found {len(state)}")
    missing, unexpected = model.load_state_dict(state, strict=True)
    if missing or unexpected:
        raise AssertionError(f"Strict restore mismatch: missing={missing}, unexpected={unexpected}")
    del state

    pretrain_config = OmegaConf.load(REPO_ROOT / "configs/lam/alam_pretrain.yaml")
    device = torch.device("cuda")
    model = model.eval().to(device)
    rgb_preprocessor = get_rgb_preprocessor(**pretrain_config.rgb_preprocessor_config).to(device)
    trainer_view = SimpleNamespace(latent_action_tokenizer=model, rgb_preprocessor=rgb_preprocessor)

    totals: defaultdict[str, float] = defaultdict(float)
    evaluated_indices: list[int] = []
    candidate = 0
    with torch.inference_mode():
        while len(evaluated_indices) < args.samples and candidate < max(args.samples * 20, 100):
            try:
                sample = validation.obtain_item(candidate, delta_t=10)
            except (FileNotFoundError, IndexError, OSError, ValueError):
                # Source snapshots can contain a damaged item; keep a deterministic
                # bounded scan and report the exact successfully evaluated indices.
                candidate += 1
                continue
            batch = {
                key: value.unsqueeze(0).to(device) if isinstance(value, torch.Tensor) else value
                for key, value in sample.items()
            }
            losses = LatentActionTokenizerTrainer.calculate_loss(trainer_view, batch, train=False)
            scalar_losses = {key: float(value.detach().float().item()) for key, value in losses.items()}
            if not all(math.isfinite(value) for value in scalar_losses.values()):
                raise FloatingPointError(f"Non-finite validation loss at index {candidate}: {scalar_losses}")
            for key, value in scalar_losses.items():
                totals[key] += value
            evaluated_indices.append(candidate)
            candidate += 1
    if len(evaluated_indices) != args.samples:
        raise RuntimeError(
            f"Only evaluated {len(evaluated_indices)}/{args.samples} CALVIN validation samples "
            f"after scanning {candidate} candidates"
        )

    averages = {key: value / len(evaluated_indices) for key, value in sorted(totals.items())}
    output = repo_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    evidence = {
        "status": "PASS",
        "test": "released ALAM checkpoint on real CALVIN validation samples",
        "checkpoint_argument": args.checkpoint,
        "checkpoint": str(checkpoint),
        "checkpoint_tensors": 458,
        "calvin_argument": calvin_value,
        "calvin_root": str(calvin_root),
        "validation_length": len(validation),
        "evaluated_indices": evaluated_indices,
        "samples": len(evaluated_indices),
        "mean_losses": averages,
    }
    output.write_text(json.dumps(evidence, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(evidence, indent=2, ensure_ascii=False))
    print(f"ALAM real-validation evaluation PASS: {output}")


if __name__ == "__main__":
    main()
