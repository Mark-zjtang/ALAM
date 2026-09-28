#!/usr/bin/env python3
"""Recompute the released MetaWorld and LIBERO table values from logs."""

from __future__ import annotations

from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import re


REPO_ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = REPO_ROOT / "evaluation" / "evaluation_evidence"

LIBERO = {
    "spatial": (
        "replan5_libero_spatial_phylam_train_20_infer__eval_30k_0506_last.log",
        "libero_spatial",
        5,
        0.992,
    ),
    "object": (
        "replan10_libero_object_phylam_train_20_infer_14_eval_30k_0504.log",
        "libero_object",
        10,
        0.996,
    ),
    "goal": ("replan7_libero_goal.log", "libero_goal", 7, 0.990),
    "long": ("replan12_libero_10.log", "libero_10", 12, 0.944),
}


def last_complete_libero_result(lines: list[str]) -> tuple[float, int]:
    pending_rate: float | None = None
    complete: list[tuple[float, int]] = []
    for line in lines:
        rate_match = re.search(r"Total success rate: ([0-9.]+)", line)
        if rate_match:
            pending_rate = float(rate_match.group(1))
            continue
        episode_match = re.search(r"Total episodes: (\d+)", line)
        if episode_match and pending_rate is not None:
            complete.append((pending_rate, int(episode_match.group(1))))
            pending_rate = None
    if not complete:
        raise AssertionError("No complete LIBERO result pair")
    return complete[-1]


def verify_libero() -> dict:
    results = {}
    for suite, (filename, suite_name, replan, expected_rate) in LIBERO.items():
        lines = (EVIDENCE / "libero" / filename).read_text(encoding="utf-8", errors="replace").splitlines()
        if not any(f"Task suite: {suite_name}" in line for line in lines):
            raise AssertionError(f"LIBERO {suite}: suite marker missing")
        replans = {
            int(match.group(1))
            for line in lines
            if (match := re.search(r"replan_steps: (\d+)", line))
        }
        if replan not in replans:
            raise AssertionError(f"LIBERO {suite}: replan {replan} missing; found {sorted(replans)}")
        rate, episodes = last_complete_libero_result(lines)
        if episodes != 500 or not math.isclose(rate, expected_rate, abs_tol=1e-12):
            raise AssertionError(
                f"LIBERO {suite}: expected rate/episodes {expected_rate}/500, found {rate}/{episodes}"
            )
        results[suite] = {
            "replan": replan,
            "successes": round(rate * episodes),
            "episodes": episodes,
            "success_percent": round(rate * 100, 10),
        }
    average = sum(item["success_percent"] for item in results.values()) / len(results)
    if not math.isclose(average, 98.05, abs_tol=1e-12):
        raise AssertionError(f"Unexpected LIBERO four-suite average: {average}")
    return {"suites": results, "average_percent": average, "paper_rounded_percent": 98.1}


def verify_metaworld() -> dict:
    lines = (EVIDENCE / "metaworld" / "mt50_cam_x_0.70.log").read_text(
        encoding="utf-8", errors="replace"
    ).splitlines()
    episode_outcomes = [
        match.group(1) == "True"
        for line in lines
        if (match := re.search(r"Episode \d+: success=(True|False)", line))
    ]
    if len(episode_outcomes) != 500 or sum(episode_outcomes) != 434:
        raise AssertionError(
            f"MetaWorld episode totals mismatch: successes={sum(episode_outcomes)} episodes={len(episode_outcomes)}"
        )
    replan_values = [
        int(match.group(1))
        for line in lines
        if (match := re.search(r"replan_steps: (\d+)", line))
    ]
    if len(replan_values) != 500 or set(replan_values) != {5}:
        raise AssertionError("MetaWorld replan evidence is not 500 entries of step 5")

    current_difficulty: str | None = None
    task_names: set[str] = set()
    difficulty_tasks: Counter[str] = Counter()
    difficulty_successes: defaultdict[str, int] = defaultdict(int)
    for line in lines:
        difficulty_match = re.search(r"Task difficulty: (easy|medium|hard|very_hard)", line)
        if difficulty_match:
            current_difficulty = difficulty_match.group(1)
            continue
        task_match = re.search(r" - ([a-z0-9-]+-v3) Success rate: ([0-9.]+)%", line)
        if task_match:
            if current_difficulty is None:
                raise AssertionError("MetaWorld task result without difficulty")
            task_name, rate_value = task_match.groups()
            if task_name in task_names:
                raise AssertionError(f"Duplicate MetaWorld task result: {task_name}")
            task_names.add(task_name)
            difficulty_tasks[current_difficulty] += 1
            difficulty_successes[current_difficulty] += round(float(rate_value) / 10.0)
            current_difficulty = None

    expected_tasks = {"easy": 28, "medium": 11, "hard": 6, "very_hard": 5}
    expected_successes = {"easy": 250, "medium": 92, "hard": 51, "very_hard": 41}
    if dict(difficulty_tasks) != expected_tasks or dict(difficulty_successes) != expected_successes:
        raise AssertionError(
            f"MetaWorld difficulty totals mismatch: tasks={dict(difficulty_tasks)} "
            f"successes={dict(difficulty_successes)}"
        )
    rates = {
        difficulty: expected_successes[difficulty] / (count * 10) * 100
        for difficulty, count in expected_tasks.items()
    }
    exact_macro = sum(rates.values()) / len(rates)
    return {
        "tasks": len(task_names),
        "successes": sum(episode_outcomes),
        "episodes": len(episode_outcomes),
        "difficulty_percent": rates,
        "exact_difficulty_macro_percent": exact_macro,
        "episode_weighted_percent": sum(episode_outcomes) / len(episode_outcomes) * 100,
        "paper_rounded_percent": 85.0,
    }


def main() -> None:
    result = {"metaworld_mt50": verify_metaworld(), "libero_table9": verify_libero(), "status": "PASS"}
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
