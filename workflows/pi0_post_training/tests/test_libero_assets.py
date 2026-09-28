#!/usr/bin/env python3
"""Check bundled LIBERO assets or perform a one-environment EGL render smoke test."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

WORKFLOWS_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKFLOWS_ROOT))

from common.paths import REPO_ROOT, repo_path, require_dir


SUITES = {
    "spatial": "libero_spatial",
    "object": "libero_object",
    "goal": "libero_goal",
    "long": "libero_10",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=tuple(SUITES), default="spatial")
    parser.add_argument("--render", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.chdir(REPO_ROOT)
    base = require_dir(repo_path("evaluation/third_party/libero/libero/libero"), "LIBERO benchmark root")
    bddl_count = sum(path.is_file() for path in (base / "bddl_files").rglob("*"))
    init_count = sum(path.is_file() for path in (base / "init_files").rglob("*"))
    asset_count = sum(path.is_file() for path in (base / "assets").rglob("*"))
    if (bddl_count, init_count, asset_count) != (135, 250, 585):
        raise AssertionError(
            f"Unexpected LIBERO bundle counts: bddl={bddl_count}, init={init_count}, assets={asset_count}"
        )
    if not args.render:
        print(f"LIBERO asset structure PASS: bddl={bddl_count}, init={init_count}, assets={asset_count}")
        return

    os.environ["MUJOCO_GL"] = "egl"
    os.environ["LIBERO_CONFIG_PATH"] = str(repo_path("configs/libero"))
    sys.path.insert(0, str(repo_path("evaluation/third_party/libero")))
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv

    suite = benchmark.get_benchmark_dict()[SUITES[args.suite]]()
    task = suite.get_task(0)
    bddl_file = base / "bddl_files" / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(bddl_file_name=str(bddl_file), camera_heights=256, camera_widths=256)
    try:
        env.seed(7)
        env.reset()
        observation = env.set_init_state(suite.get_task_init_states(0)[0])
        observation, _, _, _ = env.step([0.0] * 6 + [-1.0])
        image = observation["agentview_image"]
        if image.shape != (256, 256, 3):
            raise AssertionError(f"Unexpected LIBERO image shape: {image.shape}")
        print(f"LIBERO EGL PASS: suite={args.suite} image={image.shape}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
