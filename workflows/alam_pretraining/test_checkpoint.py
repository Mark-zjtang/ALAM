#!/usr/bin/env python3
"""Strictly restore an ALAM tokenizer checkpoint; execute it when CUDA is selected."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

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
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--input-size",
        type=int,
        help="Square CUDA smoke-test image size; defaults to 224",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.chdir(REPO_ROOT)
    configure_runtime_environment()
    sys.path.insert(0, str(REPO_ROOT))
    checkpoint = require_dir(repo_path(args.checkpoint), "ALAM checkpoint")
    config_path = require_file(checkpoint / "config.yaml", "ALAM config")
    weights_path = require_file(checkpoint / "pytorch_model.bin", "ALAM weights")

    import torch

    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or len(state) != 458:
        raise AssertionError(f"Expected 458 state tensors, found {type(state).__name__}/{len(state)}")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")

    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    config = OmegaConf.load(config_path)
    config._target_ = PORTABLE_TARGET
    model = instantiate(config)
    missing, unexpected = model.load_state_dict(state, strict=True)
    if missing or unexpected:
        raise AssertionError(f"State mismatch: missing={missing}, unexpected={unexpected}")
    del state

    restored = model.state_dict()
    if len(restored) != 458:
        raise AssertionError(f"Restored model state contains {len(restored)} tensors, expected 458")
    if args.device == "cpu":
        parameter_tensors = sum(1 for _ in model.parameters())
        buffer_tensors = sum(1 for _ in model.buffers())
        print(
            f"ALAM strict CPU restore PASS: {checkpoint} tensors=458 "
            f"parameters={parameter_tensors} buffers={buffer_tensors}; "
            "encoder execution is CUDA-only in the preserved implementation"
        )
        return

    input_size = args.input_size or 224
    patch_size = int(config.patch_size)
    if input_size <= 0 or input_size % patch_size:
        raise ValueError(f"--input-size must be a positive multiple of patch_size={patch_size}")
    device = torch.device(args.device)
    model = model.eval().to(device)
    frames = torch.rand(1, 2, 3, input_size, input_size, device=device)
    with torch.inference_mode():
        result = model.vq_encode(frames)
    shape = tuple(result["physical_latent_action"].shape)
    if shape != (1, 1, 7, 128):
        raise AssertionError(f"Unexpected physical latent shape: {shape}")
    print(
        f"ALAM strict restore and encoder PASS: {checkpoint} device=cuda "
        f"input={input_size}x{input_size} tensors=458 physical_latent_action={shape}"
    )


if __name__ == "__main__":
    main()
