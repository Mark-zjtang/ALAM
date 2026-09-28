#!/usr/bin/env python3
"""Strictly summarize one user-directed parallel LIBERO four-suite run."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re


SUITES = {
    "spatial": {
        "cli_suite": "libero_spatial",
        "infer_h": 14,
        "replan": 5,
        "released_successes": 496,
    },
    "object": {
        "cli_suite": "libero_object",
        "infer_h": 14,
        "replan": 10,
        "released_successes": 498,
    },
    "goal": {
        "cli_suite": "libero_goal",
        "infer_h": 18,
        "replan": 7,
        "released_successes": 495,
    },
    "long": {
        "cli_suite": "libero_10",
        "infer_h": 18,
        "replan": 12,
        "released_successes": 472,
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_suite(output_root: Path, suite: str, spec: dict[str, object]) -> dict[str, object]:
    cli_suite = str(spec["cli_suite"])
    infer_h = int(spec["infer_h"])
    replan = int(spec["replan"])
    model_name = f"alam_plus_pi_libero_trainH20_inferH{infer_h}_replan{replan}"
    log = (
        output_root
        / "evaluation"
        / "libero"
        / suite
        / "logs"
        / "alam_plus_pi_libero"
        / f"{cli_suite}_{model_name}.log"
    )
    if not log.is_file():
        raise AssertionError(f"missing suite log: {log}")
    lines = log.read_text(encoding="utf-8", errors="replace").splitlines()
    if sum(f"Task suite: {cli_suite}" in line for line in lines) != 1:
        raise AssertionError(f"{suite}: expected exactly one task-suite marker")
    if any(
        marker in line
        for line in lines
        for marker in (
            "Traceback (most recent call last)",
            "CUDA out of memory",
            "ConnectionClosed",
            "Connection refused",
            "Fatal Python error",
        )
    ):
        raise AssertionError(f"{suite}: fatal/error marker found in client log")

    task = None
    task_episodes: Counter[str] = Counter()
    task_successes: Counter[str] = Counter()
    success_values: list[bool] = []
    replan_values: set[int] = set()
    final_rates: list[float] = []
    final_episode_counts: list[int] = []
    for line in lines:
        match = re.search(r"Task:\s*(.+?)\s*$", line)
        if match:
            task = match.group(1)
        match = re.search(r"Success:\s*(True|False)\s*$", line)
        if match:
            if task is None:
                raise AssertionError(f"{suite}: success line before task marker")
            value = match.group(1) == "True"
            success_values.append(value)
            task_episodes[task] += 1
            task_successes[task] += int(value)
        match = re.search(r"port:\s*\d+, replan_steps:\s*(\d+)", line)
        if match:
            replan_values.add(int(match.group(1)))
        match = re.search(r"Total success rate:\s*([0-9.]+)", line)
        if match:
            final_rates.append(float(match.group(1)))
        match = re.search(r"Total episodes:\s*(\d+)", line)
        if match:
            final_episode_counts.append(int(match.group(1)))

    if len(success_values) != 500:
        raise AssertionError(f"{suite}: expected 500 outcomes, found {len(success_values)}")
    if len(task_episodes) != 10 or set(task_episodes.values()) != {50}:
        raise AssertionError(f"{suite}: invalid task/episode structure {dict(task_episodes)}")
    if replan_values != {replan}:
        raise AssertionError(f"{suite}: expected replan {replan}, found {sorted(replan_values)}")
    if len(final_rates) != 1 or len(final_episode_counts) != 1:
        raise AssertionError(
            f"{suite}: expected one final result pair, found rates={final_rates}, episodes={final_episode_counts}"
        )
    if final_episode_counts[0] != 500:
        raise AssertionError(f"{suite}: final episode count is {final_episode_counts[0]}")
    successes = sum(success_values)
    rate = successes / 500
    if not math.isclose(final_rates[0], rate, abs_tol=1e-12):
        raise AssertionError(f"{suite}: computed rate {rate} != logged {final_rates[0]}")

    released_successes = int(spec["released_successes"])
    return {
        "suite": suite,
        "cli_suite": cli_suite,
        "training_horizon": 20,
        "inference_horizon": infer_h,
        "replan_steps": replan,
        "tasks": 10,
        "episodes": 500,
        "successes": successes,
        "success_percent": rate * 100,
        "released_evidence_successes": released_successes,
        "delta_successes_vs_released": successes - released_successes,
        "task_results": [
            {
                "task": name,
                "successes": task_successes[name],
                "episodes": task_episodes[name],
                "success_percent": task_successes[name] / task_episodes[name] * 100,
            }
            for name in task_episodes
        ],
        "log": str(log),
        "log_sha256": sha256(log),
        "validation_status": "STRUCTURALLY_VALID",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")

    results = [parse_suite(args.output_root, suite, spec) for suite, spec in SUITES.items()]
    successes = sum(int(item["successes"]) for item in results)
    episodes = sum(int(item["episodes"]) for item in results)
    released_successes = sum(int(item["released_evidence_successes"]) for item in results)
    payload = {
        "schema_version": 1,
        "execution_status": "COMPLETE",
        "validation_status": "STRUCTURALLY_VALID",
        "client_topology": "one_dedicated_server_and_client_per_suite_parallel_resource_aware",
        "suite_order": list(SUITES),
        "suites": results,
        "successes": successes,
        "episodes": episodes,
        "episode_weighted_percent": successes / episodes * 100,
        "suite_macro_percent": sum(float(item["success_percent"]) for item in results) / len(results),
        "released_evidence_successes": released_successes,
        "released_evidence_episodes": 2000,
        "released_evidence_percent": released_successes / 2000 * 100,
        "delta_successes_vs_released": successes - released_successes,
        "acceptance_threshold": None,
        "ranking_or_selection_performed": False,
        "resampling_performed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp")
    if temporary.exists():
        raise SystemExit(f"refusing to overwrite temporary file {temporary}")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
