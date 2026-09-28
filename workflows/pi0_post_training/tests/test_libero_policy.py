#!/usr/bin/env python3
"""Run one real LIBERO environment action through a live policy server."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import numpy as np

WORKFLOWS_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKFLOWS_ROOT))

from common.paths import REPO_ROOT, repo_path


SUITES = {
    "spatial": "libero_spatial",
    "object": "libero_object",
    "goal": "libero_goal",
    "long": "libero_10",
}


def quat2axisangle(quat: np.ndarray) -> np.ndarray:
    scalar = float(np.clip(quat[3], -1.0, 1.0))
    denominator = np.sqrt(1.0 - scalar * scalar)
    if np.isclose(denominator, 0.0):
        return np.zeros(3)
    return quat[:3] * 2.0 * np.arccos(scalar) / denominator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=tuple(SUITES), default="spatial")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.chdir(REPO_ROOT)
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["LIBERO_CONFIG_PATH"] = str(repo_path("configs/libero"))
    sys.path.insert(0, str(repo_path("evaluation/third_party/libero")))

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from openpi_client import image_tools
    from openpi_client.websocket_client_policy import WebsocketClientPolicy

    suite = benchmark.get_benchmark_dict()[SUITES[args.suite]]()
    task = suite.get_task(0)
    bddl_file = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(bddl_file_name=str(bddl_file), camera_heights=256, camera_widths=256)
    try:
        env.seed(args.seed)
        env.reset()
        observation = env.set_init_state(suite.get_task_init_states(0)[0])
        for _ in range(50):
            observation, _, _, _ = env.step([0.0] * 6 + [-1.0])

        image = np.ascontiguousarray(observation["agentview_image"][::-1, ::-1])
        wrist = np.ascontiguousarray(observation["robot0_eye_in_hand_image"][::-1, ::-1])
        image = image_tools.convert_to_uint8(image_tools.resize_with_pad(image, 224, 224))
        wrist = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist, 224, 224))
        image = np.expand_dims(image, axis=0)
        wrist = np.expand_dims(wrist, axis=0)
        state = np.concatenate(
            (
                observation["robot0_eef_pos"],
                quat2axisangle(observation["robot0_eef_quat"]),
                observation["robot0_gripper_qpos"],
            )
        )
        result = WebsocketClientPolicy(args.host, args.port).infer(
            {
                "observation/image": image,
                "observation/wrist_image": wrist,
                "observation/state": state,
                "prompt": str(task.language),
            }
        )
        actions = np.asarray(result["actions"])
        if actions.ndim != 2 or actions.shape[1] < 7 or not np.isfinite(actions).all():
            raise AssertionError(f"Unexpected policy actions: shape={actions.shape}")
        observation, _, _, _ = env.step(actions[0, :7].tolist())
        if "agentview_image" not in observation:
            raise AssertionError("LIBERO environment did not return the next observation")
        print(f"ALAM+pi0+LIBERO E2E PASS: suite={args.suite} actions={actions.shape}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
