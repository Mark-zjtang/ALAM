#!/usr/bin/env python3
"""Record the exact runtime and protocol used by a MetaWorld evaluation."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGES = (
    "metaworld",
    "mujoco",
    "gymnasium",
    "numpy",
    "torch",
    "torchvision",
    "jax",
    "jaxlib",
    "flax",
    "orbax-checkpoint",
    "websockets",
    "openpi-client",
)
SOURCE_FILES = (
    "evaluation/examples/metaworld/eval_metaworld_policy_client_0409.py",
    "evaluation/examples/metaworld/mt50_eval_instruction.py",
    "evaluation/examples/metaworld/requirements-eval.txt",
    "evaluation/src/openpi/models/physics_va_flow.py",
    "evaluation/src/openpi/policies/policy.py",
    "workflows/pi0_post_training/evaluate_metaworld_mt50.sh",
    "workflows/pi0_post_training/run_evaluation_client.py",
    "workflows/pi0_post_training/serve_policy.py",
    "workflows/metaworld_evaluation/release_profile_x071.json",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", choices=("client", "server"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--camera-x", type=float, required=True)
    parser.add_argument("--camera-y", type=float, required=True)
    parser.add_argument("--camera-z", type=float, required=True)
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()

    sources = {}
    for relative in SOURCE_FILES:
        path = REPO_ROOT / relative
        sources[relative] = sha256(path) if path.is_file() else None

    record = {
        "schema_version": 1,
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "role": args.role,
        "profile": args.profile,
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
            "executable": sys.executable,
        },
        "platform": platform.platform(),
        "packages": package_versions(),
        "protocol": {
            "camera_name": "corner2",
            "camera_position": [args.camera_x, args.camera_y, args.camera_z],
            "tasks": 50,
            "episodes_per_task": args.episodes,
            "raw_action_horizon": 6,
            "effective_inference_horizon": 5,
            "chunk_and_replan": 5,
            "max_steps": 200,
            "evaluator_seed_argument": 10,
            "port": args.port,
            "server_mode": "one_dedicated_server_one_client",
        },
        "reproducibility_boundary": {
            "mt50_constructed_before_evaluator_seed": True,
            "task_variants_frozen": False,
            "metaworld_2_reset_seed_controls_task_rand_vec": False,
            "policy_rng_initial_key": 0,
        },
        "source_sha256": sources,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)


if __name__ == "__main__":
    main()
