#!/usr/bin/env python3
"""Order created Hugging Face artifacts by exact local upload bytes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


WORKFLOWS_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKFLOWS_ROOT))

from common.paths import repo_path


MANIFEST = Path(__file__).with_name("huggingface_manifest.json")
DEFAULT_IDS = (
    "oxe_mix10_datasets",
    "alam_plus_pi_libero",
    "alam_plus_pi_metaworld_mt50",
    "alam_pretrain",
)


def ignored(path: Path) -> bool:
    return (
        any(part in {".git", ".cache", ".build", "__pycache__"} for part in path.parts)
        or path.name == ".gitattributes"
        or path.name.endswith(".partial")
    )


def source_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_path(path)


def upload_bytes(artifact: dict) -> int:
    total = repo_path(artifact["card"]).stat().st_size
    for source in artifact["upload_sources"]:
        root = source_path(source["path"])
        if not root.is_dir():
            raise FileNotFoundError(f"Missing staged upload source for {artifact['id']}: {root}")
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            relative = path.relative_to(root)
            if ignored(relative) or (source["path_in_repo"] == "." and relative.as_posix() == "README.md"):
                continue
            total += path.stat().st_size
    return total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", action="append", default=[])
    parser.add_argument("--format", choices=("json", "shell"), default="json")
    args = parser.parse_args()
    artifacts = json.loads(MANIFEST.read_text(encoding="utf-8"))["artifacts"]
    by_id = {artifact["id"]: artifact for artifact in artifacts}
    selected_ids = args.artifact or list(DEFAULT_IDS)
    unknown = sorted(set(selected_ids) - set(by_id))
    if unknown:
        raise ValueError(f"Unknown artifact IDs: {', '.join(unknown)}")
    records = sorted(
        ({"id": artifact_id, "bytes": upload_bytes(by_id[artifact_id])} for artifact_id in selected_ids),
        key=lambda item: (-item["bytes"], item["id"]),
    )
    if args.format == "shell":
        for record in records:
            print(f"{record['id']}\t{record['bytes']}")
    else:
        print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
