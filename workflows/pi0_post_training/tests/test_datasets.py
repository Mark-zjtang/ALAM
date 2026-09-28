#!/usr/bin/env python3
"""Validate both repository-relative LeRobot v2 fine-tuning datasets."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys


WORKFLOWS_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKFLOWS_ROOT))

from common.paths import configure_runtime_environment, repo_path, require_dir, require_file


SPECS = {
    "metaworld_mt50": {
        "episodes": 2500,
        "frames": 204806,
        "tasks": 49,
        "robot_type": "metaworld",
        "features": {"observation.state", "observation.image", "action"},
    },
    "libero_real": {
        "episodes": 1693,
        "frames": 273465,
        "tasks": 40,
        "robot_type": "panda",
        "features": {"state", "image", "wrist_image", "actions"},
    },
}


def read_jsonl(path: Path) -> list[dict]:
    records = []
    with require_file(path, path.name).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise AssertionError(f"Invalid JSON at {path}:{line_number}") from error
    return records


def check_dataset(parent: Path, repo_id: str, spec: dict[str, object]) -> dict[str, object]:
    root = require_dir(parent / repo_id, f"LeRobot dataset {repo_id}")
    metadata = require_dir(root / "meta", f"{repo_id} metadata")
    info = json.loads(require_file(metadata / "info.json", f"{repo_id} info").read_text(encoding="utf-8"))
    require_file(metadata / "stats.json", f"{repo_id} normalization statistics")
    expected_scalars = {
        "codebase_version": "v2.0",
        "total_episodes": spec["episodes"],
        "total_frames": spec["frames"],
        "total_tasks": spec["tasks"],
        "robot_type": spec["robot_type"],
    }
    found_scalars = {key: info.get(key) for key in expected_scalars}
    if found_scalars != expected_scalars:
        raise AssertionError(f"{repo_id} info mismatch: expected={expected_scalars}, found={found_scalars}")
    missing_features = set(spec["features"]) - set(info.get("features", {}))
    if missing_features:
        raise AssertionError(f"{repo_id} missing training features: {sorted(missing_features)}")

    episodes = read_jsonl(metadata / "episodes.jsonl")
    tasks = read_jsonl(metadata / "tasks.jsonl")
    if len(episodes) != spec["episodes"] or len(tasks) != spec["tasks"]:
        raise AssertionError(f"{repo_id} JSONL count mismatch: episodes={len(episodes)} tasks={len(tasks)}")
    if [record.get("episode_index") for record in episodes] != list(range(int(spec["episodes"]))):
        raise AssertionError(f"{repo_id} episode indices are not contiguous")
    if [record.get("task_index") for record in tasks] != list(range(int(spec["tasks"]))):
        raise AssertionError(f"{repo_id} task indices are not contiguous")
    if sum(int(record["length"]) for record in episodes) != spec["frames"]:
        raise AssertionError(f"{repo_id} episode lengths do not sum to total_frames")

    chunks_size = int(info["chunks_size"])
    checked_files = []
    for episode_index in (0, int(spec["episodes"]) // 2, int(spec["episodes"]) - 1):
        relative = info["data_path"].format(
            episode_chunk=episode_index // chunks_size,
            episode_index=episode_index,
        )
        path = require_file(root / relative, f"{repo_id} episode {episode_index}")
        if path.stat().st_size <= 0:
            raise AssertionError(f"Empty LeRobot parquet: {path}")
        checked_files.append(relative)
    return {
        "root": str(root),
        "episodes": len(episodes),
        "frames": sum(int(record["length"]) for record in episodes),
        "tasks": len(tasks),
        "representative_parquet": checked_files,
    }


def main() -> None:
    configure_runtime_environment()
    parent = repo_path(os.environ.get("HF_LEROBOT_HOME", "data/lerobot"))
    result = {repo_id: check_dataset(parent, repo_id, spec) for repo_id, spec in SPECS.items()}
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print("Relative LeRobot dataset layouts PASS")


if __name__ == "__main__":
    main()
