#!/usr/bin/env python3
"""Prepare a runtime-only ALAM checkpoint config with a portable class target."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import sys
import tempfile
import uuid

WORKFLOWS_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKFLOWS_ROOT))

from common.paths import REPO_ROOT, configure_runtime_environment, env_path, repo_path, require_dir, require_file


PRESERVED_TARGET = (
    "openpi.models.physics_lam_model.uni_world_model.latent_action_model.models."
    "ctl_latent_action_tokenizer_v3.LatentActionTokenizer"
)
PORTABLE_TARGET = "openpi.models.portable_lam_tokenizer.LatentActionTokenizer"
PRETRAIN_TARGET = (
    "Algebraic_latent_action_model.latent_action_model.models."
    "ctl_latent_action_tokenizer_v3_portable.LatentActionTokenizer"
)
SUPPORTED_TARGETS = (PRESERVED_TARGET, PRETRAIN_TARGET, PORTABLE_TARGET)


def prepare(source: Path) -> Path:
    source = require_dir(source.resolve(), "ALAM checkpoint directory")
    source_config = require_file(source / "config.yaml", "ALAM config")
    source_weights = require_file(source / "pytorch_model.bin", "ALAM weights")

    runtime_root = env_path("ALAM_RUNTIME_ROOT", ".runtime") / "alam_checkpoints"
    # Different experiments often save the same step/epoch directory name.
    source_id = hashlib.sha256(os.fsencode(source)).hexdigest()[:16]
    destination = runtime_root / f"{source.name}-{source_id}"
    destination.mkdir(parents=True, exist_ok=True)

    config_text = source_config.read_text(encoding="utf-8")
    source_target = next((target for target in SUPPORTED_TARGETS if target in config_text), None)
    if source_target is None:
        raise ValueError(f"Unexpected ALAM target in {source_config}")
    portable_config = config_text.replace(source_target, PORTABLE_TARGET, 1)
    config_path = destination / "config.yaml"
    temporary_config: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination,
            prefix=".config.yaml.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(portable_config)
            temporary_config = Path(handle.name)
        os.replace(temporary_config, config_path)
        temporary_config = None
    finally:
        if temporary_config is not None:
            temporary_config.unlink(missing_ok=True)

    weight_link = destination / "pytorch_model.bin"
    desired_target = os.path.relpath(source_weights, destination)
    if os.path.lexists(weight_link) and not weight_link.is_symlink():
        raise FileExistsError(f"Refusing to replace non-symlink runtime file: {weight_link}")
    if not weight_link.is_symlink() or os.readlink(weight_link) != desired_target:
        temporary_link = destination / (
            f".pytorch_model.bin.{os.getpid()}.{uuid.uuid4().hex}.tmp"
        )
        try:
            temporary_link.symlink_to(desired_target)
            if os.path.lexists(weight_link) and not weight_link.is_symlink():
                raise FileExistsError(f"Refusing to replace non-symlink runtime file: {weight_link}")
            os.replace(temporary_link, weight_link)
        finally:
            temporary_link.unlink(missing_ok=True)
    if not weight_link.is_symlink() or os.readlink(weight_link) != desired_target:
        raise RuntimeError(f"Runtime ALAM weight link was not prepared atomically: {weight_link}")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", help="ALAM checkpoint path, relative to repository root by default")
    args = parser.parse_args()
    os.chdir(REPO_ROOT)
    configure_runtime_environment()
    print(prepare(repo_path(args.checkpoint)))


if __name__ == "__main__":
    main()
