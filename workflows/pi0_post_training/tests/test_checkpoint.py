#!/usr/bin/env python3
"""Restore a released pi0 Orbax parameter tree and optionally test JAX CUDA."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import sys

WORKFLOWS_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKFLOWS_ROOT))

from common.paths import REPO_ROOT, configure_runtime_environment, repo_path, require_dir, require_file


EXPECTED_RELEASE_METADATA_SHA256 = "22bd0d80da1aa710a1ee5bb60fd5fe30528446fb7c09b37d14e9cf6a364ac910"
EXPECTED_PARAMETER_LEAVES = 68


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", choices=("metaworld", "libero"), default="libero")
    parser.add_argument("--checkpoint", help="Checkpoint directory; relative paths use the repository root")
    parser.add_argument(
        "--allow-train-state",
        action="store_true",
        help="Accept a training checkpoint that also contains train_state/",
    )
    parser.add_argument("--structure-only", action="store_true")
    parser.add_argument("--gpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.chdir(REPO_ROOT)
    configure_runtime_environment()
    relative = {
        "metaworld": "evaluation/checkpoints/metaworld/alam_plus_pi_metaworld_mt50_step30000",
        "libero": "evaluation/checkpoints/libero/alam_plus_pi_libero_step30000",
    }[args.benchmark]
    checkpoint = require_dir(repo_path(args.checkpoint or relative), "pi0 checkpoint")
    require_file(checkpoint / "_CHECKPOINT_METADATA", "Orbax checkpoint metadata")
    parameter_metadata = require_file(checkpoint / "params" / "_METADATA", "Orbax parameter metadata")
    if (checkpoint / "train_state").exists() and not args.allow_train_state:
        raise AssertionError("Inference checkpoint must not include train_state")
    if not args.allow_train_state:
        metadata_sha256 = hashlib.sha256(parameter_metadata.read_bytes()).hexdigest()
        if metadata_sha256 != EXPECTED_RELEASE_METADATA_SHA256:
            raise AssertionError(
                "Unexpected released pi0 parameter metadata: "
                f"expected={EXPECTED_RELEASE_METADATA_SHA256} actual={metadata_sha256}"
            )
    if args.structure_only:
        print(f"pi0 checkpoint structure PASS: {checkpoint}")
        return

    sys.path.insert(0, str(REPO_ROOT / "evaluation" / "src"))
    import numpy as np
    from flax import traverse_util
    from openpi.models import model as model_lib

    params = model_lib.restore_params(checkpoint / "params", restore_type=np.ndarray)
    flat = traverse_util.flatten_dict(params)
    if len(flat) != EXPECTED_PARAMETER_LEAVES:
        raise AssertionError(
            f"Unexpected pi0 parameter tree: expected={EXPECTED_PARAMETER_LEAVES} actual={len(flat)} leaves"
        )
    empty = ["/".join(map(str, key)) for key, value in flat.items() if np.asarray(value).size == 0]
    if empty:
        raise AssertionError(f"Empty pi0 parameter arrays: {empty}")
    logical_bytes = sum(np.asarray(value).nbytes for value in flat.values())
    keys = ["/".join(map(str, key)) for key in sorted(flat)]
    required_fragments = ("PaliGemma", "lam_proj", "action_out_proj")
    missing_fragments = [fragment for fragment in required_fragments if not any(fragment in key for key in keys)]
    if missing_fragments:
        raise AssertionError(f"Missing expected pi0 modules: {missing_fragments}")
    structure = hashlib.sha256()
    for key, value in sorted(flat.items()):
        array = np.asarray(value)
        structure.update("/".join(map(str, key)).encode("utf-8"))
        structure.update(b"\0")
        structure.update(str(tuple(array.shape)).encode("ascii"))
        structure.update(b"\0")
        structure.update(str(array.dtype).encode("ascii"))
        structure.update(b"\n")
    print(
        f"pi0 Orbax restore PASS: benchmark={args.benchmark} checkpoint={checkpoint} "
        f"leaves={len(flat)} logical_bytes={logical_bytes:,} "
        f"structure_sha256={structure.hexdigest()}"
    )

    if not args.gpu:
        return

    import jax
    import jax.numpy as jnp
    devices = jax.devices("gpu")
    if not devices:
        raise RuntimeError("JAX did not discover a GPU")
    value = jax.jit(lambda x: x @ x)(jnp.ones((32, 32), dtype=jnp.float32))
    value.block_until_ready()
    print(f"JAX CUDA backend PASS: devices={devices} sum={float(value.sum())}")


if __name__ == "__main__":
    main()
