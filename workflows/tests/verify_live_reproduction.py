#!/usr/bin/env python3
"""Verify completed live MetaWorld/LIBERO runs against released targets."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import re
import sys


WORKFLOWS_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKFLOWS_ROOT))

from common.paths import REPO_ROOT, repo_path, require_dir, require_file


LIBERO = {
    "spatial": ("libero_spatial", 14, 5, 496),
    "object": ("libero_object", 14, 10, 498),
    "goal": ("libero_goal", 18, 7, 495),
    "long": ("libero_10", 18, 12, 472),
}


def last_complete_libero_run(lines: list[str]) -> tuple[list[bool], list[int], float, int]:
    """Return the final complete 500-episode run from a possibly appended log."""
    summary_indices = [
        index
        for index, line in enumerate(lines)
        if re.search(r"Total episodes: 500$", line)
    ]
    if not summary_indices:
        raise AssertionError("No complete 500-episode LIBERO result in evaluator log")

    summary_index = summary_indices[-1]
    prior_lines = lines[:summary_index]
    outcomes = [
        match.group(1) == "True"
        for line in prior_lines
        if (match := re.search(r"Success: (True|False)$", line))
    ][-500:]
    replans = [
        int(match.group(1))
        for line in prior_lines
        if (match := re.search(r"replan_steps: (\d+)", line))
    ][-500:]
    rates = [
        float(match.group(1))
        for line in prior_lines
        if (match := re.search(r"Total success rate: ([0-9.]+)$", line))
    ]
    if len(outcomes) != 500 or len(replans) != 500 or not rates:
        raise AssertionError(
            "Final LIBERO result is incomplete: "
            f"outcomes={len(outcomes)} replans={len(replans)} rates={len(rates)}"
        )
    return outcomes, replans, rates[-1], 500


def server_spec(path: Path, checkpoint_name: str, policy_checkpoint: Path) -> dict:
    lines = require_file(path, "policy server log").read_text(encoding="utf-8", errors="replace").splitlines()
    joined = "\n".join(lines)
    if "Traceback (most recent call last)" in joined:
        raise AssertionError(f"Policy server traceback found in {path}")
    if not any("pytorch_model.bin" in line and line.lstrip().startswith("load ") for line in lines):
        raise AssertionError(f"No ALAM weight-load evidence in {path}")
    if not any("missing:" in line and "set()" in line for line in lines):
        raise AssertionError(f"ALAM missing-key proof is absent or non-empty in {path}")
    if not any("unexpected:" in line and "[]" in line for line in lines):
        raise AssertionError(f"ALAM unexpected-key proof is absent or non-empty in {path}")
    prefix = "Release policy specification: "
    specs = [json.loads(line.split(prefix, 1)[1]) for line in lines if prefix in line]
    if not specs:
        raise AssertionError(f"No release policy specification in {path}")
    spec = specs[-1]
    if Path(spec["policy_checkpoint"]).resolve() != policy_checkpoint.resolve():
        raise AssertionError(f"Unexpected policy checkpoint in {path}: {spec['policy_checkpoint']}")
    if Path(spec["alam_checkpoint"]).name != checkpoint_name:
        raise AssertionError(f"Unexpected ALAM checkpoint in {path}: {spec['alam_checkpoint']}")
    return spec


def verify_metaworld(output_root: Path) -> dict[str, object]:
    run_root = require_dir(output_root / "evaluation" / "metaworld_mt50", "MetaWorld live output")
    policy = repo_path("evaluation/checkpoints/metaworld/alam_plus_pi_metaworld_mt50_step30000")
    spec = server_spec(run_root / "server" / "server.log", "metaworld_epoch19_step58216", policy)
    if (int(spec["effective_horizon"]), int(spec["replan"])) != (5, 5):
        raise AssertionError(f"MetaWorld inference configuration mismatch: {spec}")

    logs = sorted(run_root.glob("*/pi0_wflow_policy_49_task_evaluate_logging_*/*.log"), key=lambda p: p.stat().st_mtime)
    if not logs:
        raise FileNotFoundError(f"No MetaWorld evaluator log below {run_root}")
    log_path = logs[-1]
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    outcomes = [
        match.group(1) == "True"
        for line in lines
        if (match := re.search(r"Episode \d+: success=(True|False)", line))
    ]
    replans = [
        int(match.group(1))
        for line in lines
        if (match := re.search(r"replan_steps: (\d+)", line))
    ]
    if len(outcomes) != 500 or sum(outcomes) != 434:
        raise AssertionError(f"MetaWorld target mismatch: {sum(outcomes)}/{len(outcomes)}, expected 434/500")
    if len(replans) != 500 or set(replans) != {5}:
        raise AssertionError(f"MetaWorld replan evidence mismatch: entries={len(replans)} values={set(replans)}")

    current_difficulty: str | None = None
    task_names: set[str] = set()
    task_counts: Counter[str] = Counter()
    successes: defaultdict[str, int] = defaultdict(int)
    for line in lines:
        difficulty = re.search(r"Task difficulty: (easy|medium|hard|very_hard)", line)
        if difficulty:
            current_difficulty = difficulty.group(1)
            continue
        task = re.search(r" - ([a-z0-9-]+-v3) Success rate: ([0-9.]+)%", line)
        if task:
            if current_difficulty is None or task.group(1) in task_names:
                raise AssertionError(f"Invalid or duplicate MetaWorld task record: {line}")
            task_names.add(task.group(1))
            task_counts[current_difficulty] += 1
            successes[current_difficulty] += round(float(task.group(2)) / 10.0)
            current_difficulty = None
    expected_counts = {"easy": 28, "medium": 11, "hard": 6, "very_hard": 5}
    expected_successes = {"easy": 250, "medium": 92, "hard": 51, "very_hard": 41}
    if dict(task_counts) != expected_counts or dict(successes) != expected_successes:
        raise AssertionError(
            f"MetaWorld difficulty mismatch: tasks={dict(task_counts)} successes={dict(successes)}"
        )
    difficulty_percent = {
        name: expected_successes[name] / (expected_counts[name] * 10) * 100
        for name in expected_counts
    }
    return {
        "status": "PASS",
        "log": str(log_path),
        "policy_checkpoint": str(policy),
        "alam_checkpoint": "evaluation/checkpoints/alam/metaworld_epoch19_step58216",
        "tasks": len(task_names),
        "successes": sum(outcomes),
        "episodes": len(outcomes),
        "episode_weighted_percent": sum(outcomes) / len(outcomes) * 100,
        "difficulty_percent": difficulty_percent,
        "difficulty_macro_percent": sum(difficulty_percent.values()) / len(difficulty_percent),
        "paper_rounded_percent": 85.0,
    }


def verify_libero(
    output_root: Path, checkpoint_name: str = "metaworld_epoch19_step58216"
) -> dict[str, object]:
    policy = repo_path("evaluation/checkpoints/libero/alam_plus_pi_libero_step30000")
    results: dict[str, dict[str, object]] = {}
    for suite, (cli_suite, horizon, replan, expected_successes) in LIBERO.items():
        run_root = require_dir(output_root / "evaluation" / "libero" / suite, f"LIBERO {suite} live output")
        spec = server_spec(run_root / "server" / "server.log", checkpoint_name, policy)
        if (int(spec["effective_horizon"]), int(spec["replan"])) != (horizon, replan):
            raise AssertionError(f"LIBERO {suite} inference configuration mismatch: {spec}")
        model_name = f"alam_plus_pi_libero_trainH20_inferH{horizon}_replan{replan}"
        log_path = require_file(
            run_root / "logs" / "alam_plus_pi_libero" / f"{cli_suite}_{model_name}.log",
            f"LIBERO {suite} evaluator log",
        )
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        outcomes, replan_entries, final_rate, final_episodes = last_complete_libero_run(lines)
        if len(outcomes) != 500 or sum(outcomes) != expected_successes:
            raise AssertionError(
                f"LIBERO {suite} target mismatch: {sum(outcomes)}/{len(outcomes)}, "
                f"expected {expected_successes}/500"
            )
        replan_values = set(replan_entries)
        if replan_values != {replan}:
            raise AssertionError(f"LIBERO {suite} replan mismatch: {replan_values}")
        expected_rate = expected_successes / 500
        if not math.isclose(final_rate, expected_rate, abs_tol=1e-12):
            raise AssertionError(f"LIBERO {suite} final rate mismatch: {final_rate}")
        if final_episodes != 500:
            raise AssertionError(f"LIBERO {suite} final episode count mismatch: {final_episodes}")
        results[suite] = {
            "status": "PASS",
            "log": str(log_path),
            "inference_horizon": horizon,
            "replan": replan,
            "successes": sum(outcomes),
            "episodes": len(outcomes),
            "success_percent": expected_rate * 100,
        }
    average = sum(float(result["success_percent"]) for result in results.values()) / len(results)
    if not math.isclose(average, 98.05, abs_tol=1e-12):
        raise AssertionError(f"LIBERO average mismatch: {average}")
    return {
        "status": "PASS",
        "policy_checkpoint": str(policy),
        "alam_checkpoint": f"evaluation/checkpoints/alam/{checkpoint_name}",
        "suites": results,
        "average_percent": average,
        "paper_rounded_percent": 98.1,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("benchmark", choices=("metaworld", "libero", "all"))
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--summary")
    args = parser.parse_args()
    output_root = require_dir(repo_path(args.output_root), "live reproduction output root")
    result: dict[str, object] = {"status": "PASS", "output_root": str(output_root)}
    if args.benchmark in ("metaworld", "all"):
        result["metaworld_mt50"] = verify_metaworld(output_root)
    if args.benchmark in ("libero", "all"):
        result["libero"] = verify_libero(output_root)
    if args.summary:
        summary = repo_path(args.summary)
        summary.parent.mkdir(parents=True, exist_ok=True)
        summary.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
