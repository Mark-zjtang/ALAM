#!/usr/bin/env python3
"""Run one real ALAM optimization step from repository-relative assets.

This acceptance test intentionally calls the preserved dataset classes, model
forward/loss implementation, and optimizer.  It does not call ``trainer.train``
because that production loop traverses a full epoch and writes a checkpoint at
step zero; those behaviours are inappropriate for a bounded smoke test.
"""

from __future__ import annotations

import argparse
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
    parser.add_argument("--calvin-root", default=None)
    parser.add_argument("--oxe-root", default=None)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--output", default="outputs/tests/alam_training_smoke/summary.json")
    return parser.parse_args()


def assert_sample(name: str, sample: dict) -> dict[str, object]:
    initial_shape = tuple(sample["rgb_initial"].shape)
    future_shape = tuple(sample["rgb_future"].shape)
    if initial_shape != (1, 3, 200, 200):
        raise AssertionError(f"{name}: unexpected rgb_initial shape {initial_shape}")
    if future_shape != (2, 3, 200, 200):
        raise AssertionError(f"{name}: unexpected rgb_future shape {future_shape}")
    if int(sample["latent_mask"].sum()) != 2:
        raise AssertionError(f"{name}: both future frames must be valid")
    return {
        "rgb_initial": list(initial_shape),
        "rgb_future": list(future_shape),
        "delta_t": int(sample["delta_t"]),
    }


def load_real_samples(calvin_root: Path, oxe_root: Path):
    from Algebraic_latent_action_model.data.data_utils import load_dataset
    from torchvision.transforms.v2 import InterpolationMode, Resize

    extra = {
        "sequence_length": 1,
        "do_extract_future_frames": True,
        "do_extract_action": False,
    }
    resize = Resize([200, 200], interpolation=InterpolationMode.BICUBIC, antialias=True)
    calvin_config = {
        "data_type": "video_npz_3seq",
        "npz_dir": str(calvin_root),
        "skip_frame": 10,
        "rgb_shape": [200, 200],
        "rgb_preprocessor": resize,
    }
    calvin_train, calvin_val = load_dataset(calvin_config, extra)
    calvin_sample = calvin_train.obtain_item(0, delta_t=10)

    oxe_display = require_dir(
        oxe_root / "fractal20220817_data" / "image",
        "OXE fractal20220817_data/image directory",
    )
    oxe_config = {
        "data_type": "video_json",
        "video_dir": str(oxe_display),
        "skip_frame": 5,
        "rgb_shape": [200, 200],
        "rgb_preprocessor": resize,
    }
    oxe_train, oxe_val = load_dataset(oxe_config, extra)
    oxe_sample = None
    last_error: Exception | None = None
    for index in range(min(50, len(oxe_train.videos))):
        _, frame_count, *_ = oxe_train.videos[index]
        if int(frame_count) <= 10:
            continue
        try:
            oxe_sample = oxe_train.obtain_item(index, start_local_step=0, delta_t=5)
            break
        except Exception as error:  # keep searching for a decodable public sample
            last_error = error
    if oxe_sample is None:
        raise RuntimeError("Could not decode an OXE smoke sample") from last_error

    lengths = {
        "calvin_train": len(calvin_train),
        "calvin_validation": len(calvin_val),
        "oxe_train": len(oxe_train),
        "oxe_validation": len(oxe_val),
    }
    return calvin_sample, oxe_sample, lengths


def main() -> None:
    args = parse_args()
    os.chdir(REPO_ROOT)
    configure_runtime_environment()
    sys.path.insert(0, str(REPO_ROOT))

    # Registers the compatibility alias required by byte-preserved imports.
    import Algebraic_latent_action_model  # noqa: F401

    calvin_value = args.calvin_root or os.environ.get("ALAM_CALVIN_ROOT", "data/alam/calvin")
    oxe_value = args.oxe_root or os.environ.get("ALAM_OXE_VIDEO_ROOT", "data/alam/oxe_videos")
    calvin_root = require_dir(repo_path(calvin_value), "CALVIN pretraining dataset")
    oxe_root = require_dir(repo_path(oxe_value), "OXE video dataset")
    checkpoint = require_dir(repo_path(args.checkpoint), "ALAM checkpoint")
    config_path = require_file(checkpoint / "config.yaml", "ALAM checkpoint config")
    weights_path = require_file(checkpoint / "pytorch_model.bin", "ALAM checkpoint weights")

    calvin_sample, oxe_sample, lengths = load_real_samples(calvin_root, oxe_root)
    sample_evidence = {
        "calvin": assert_sample("CALVIN", calvin_sample),
        "oxe": assert_sample("OXE", oxe_sample),
    }

    import torch
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the ALAM optimization-step smoke test")
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    device = torch.device(args.device)

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
    model = model.train().to(device)

    pretrain_config = OmegaConf.load(REPO_ROOT / "configs" / "lam" / "alam_pretrain.yaml")
    from Algebraic_latent_action_model.data.img_utils import get_rgb_preprocessor
    from Algebraic_latent_action_model.latent_action_model.trainers.ctl_latent_action_tokenizer_trainer import (
        LatentActionTokenizerTrainer,
    )
    from Algebraic_latent_action_model.latent_action_model.trainers.optimizer import get_optimizer

    rgb_preprocessor = get_rgb_preprocessor(**pretrain_config.rgb_preprocessor_config).to(device)
    optimizer = get_optimizer(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(pretrain_config.training_config.lr_max),
        wd=float(pretrain_config.training_config.weight_decay),
    )
    batch = {
        key: value.unsqueeze(0).to(device) if isinstance(value, torch.Tensor) else value
        for key, value in calvin_sample.items()
    }
    trainer_view = SimpleNamespace(
        latent_action_tokenizer=model,
        rgb_preprocessor=rgb_preprocessor,
    )

    torch.cuda.reset_peak_memory_stats(device)
    optimizer.zero_grad(set_to_none=True)
    before = model.action_latent.detach().clone()
    losses = LatentActionTokenizerTrainer.calculate_loss(trainer_view, batch, train=True)
    total_loss = losses["loss"]
    if not bool(torch.isfinite(total_loss)):
        raise FloatingPointError(f"Non-finite ALAM loss: {total_loss.item()}")
    total_loss.backward()
    grad_sq = torch.zeros((), device=device)
    for parameter in model.parameters():
        if parameter.grad is not None:
            grad_sq += parameter.grad.detach().float().square().sum()
    grad_norm = float(torch.sqrt(grad_sq).item())
    if not math.isfinite(grad_norm) or grad_norm <= 0:
        raise FloatingPointError(f"Invalid gradient norm: {grad_norm}")
    optimizer.step()
    parameter_delta = float((model.action_latent.detach() - before).abs().max().item())
    if not math.isfinite(parameter_delta) or parameter_delta <= 0:
        raise AssertionError(f"Optimizer did not update action_latent: delta={parameter_delta}")
    torch.cuda.synchronize(device)

    output = repo_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    evidence = {
        "status": "PASS",
        "test": "one real ALAM forward, backward, and optimizer step",
        "checkpoint_argument": args.checkpoint,
        "checkpoint": str(checkpoint),
        "checkpoint_tensors": 458,
        "calvin_argument": calvin_value,
        "calvin_root": str(calvin_root),
        "oxe_argument": oxe_value,
        "oxe_root": str(oxe_root),
        "dataset_lengths": lengths,
        "samples": sample_evidence,
        "losses": {
            key: float(value.detach().float().item())
            for key, value in losses.items()
        },
        "gradient_norm": grad_norm,
        "action_latent_max_update": parameter_delta,
        "cuda_peak_allocated_gib": torch.cuda.max_memory_allocated(device) / (1024**3),
    }
    output.write_text(json.dumps(evidence, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(evidence, indent=2, ensure_ascii=False))
    print(f"ALAM relative-path training smoke PASS: {output}")


if __name__ == "__main__":
    main()
