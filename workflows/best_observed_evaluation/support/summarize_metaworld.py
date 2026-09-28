#!/usr/bin/env python3
"""Strictly validate and summarize one original MetaWorld 10x50 log.

This parser intentionally has no metric threshold, acceptance gate, ranking, or
best-of-N behavior.  A zero exit status means only that the preserved evaluator
log is structurally complete and internally consistent.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
from typing import NoReturn


EPISODES_PER_TASK = 10
EXPECTED_REPLAN_STEPS = 5
CAMERA_Y = 0.075
CAMERA_Z = 0.70
DIFFICULTIES = ("easy", "medium", "hard", "very_hard")

EXPECTED_TASKS = (
    "assembly-v3",
    "basketball-v3",
    "bin-picking-v3",
    "box-close-v3",
    "button-press-topdown-v3",
    "button-press-topdown-wall-v3",
    "button-press-v3",
    "button-press-wall-v3",
    "coffee-button-v3",
    "coffee-pull-v3",
    "coffee-push-v3",
    "dial-turn-v3",
    "disassemble-v3",
    "door-close-v3",
    "door-lock-v3",
    "door-open-v3",
    "door-unlock-v3",
    "hand-insert-v3",
    "drawer-close-v3",
    "drawer-open-v3",
    "faucet-open-v3",
    "faucet-close-v3",
    "hammer-v3",
    "handle-press-side-v3",
    "handle-press-v3",
    "handle-pull-side-v3",
    "handle-pull-v3",
    "lever-pull-v3",
    "pick-place-wall-v3",
    "pick-out-of-hole-v3",
    "pick-place-v3",
    "plate-slide-v3",
    "plate-slide-side-v3",
    "plate-slide-back-v3",
    "plate-slide-back-side-v3",
    "peg-insert-side-v3",
    "peg-unplug-side-v3",
    "soccer-v3",
    "stick-push-v3",
    "stick-pull-v3",
    "push-v3",
    "push-wall-v3",
    "push-back-v3",
    "reach-v3",
    "reach-wall-v3",
    "shelf-place-v3",
    "sweep-into-v3",
    "sweep-v3",
    "window-open-v3",
    "window-close-v3",
)

TASK_DIFFICULTY = {
    "easy": (
        "button-press-v3",
        "button-press-topdown-v3",
        "button-press-topdown-wall-v3",
        "button-press-wall-v3",
        "coffee-button-v3",
        "dial-turn-v3",
        "door-close-v3",
        "door-lock-v3",
        "door-open-v3",
        "door-unlock-v3",
        "drawer-close-v3",
        "drawer-open-v3",
        "faucet-close-v3",
        "faucet-open-v3",
        "handle-press-v3",
        "handle-press-side-v3",
        "handle-pull-v3",
        "handle-pull-side-v3",
        "lever-pull-v3",
        "plate-slide-v3",
        "plate-slide-back-v3",
        "plate-slide-back-side-v3",
        "plate-slide-side-v3",
        "reach-v3",
        "reach-wall-v3",
        "window-close-v3",
        "window-open-v3",
        "peg-unplug-side-v3",
    ),
    "medium": (
        "basketball-v3",
        "bin-picking-v3",
        "box-close-v3",
        "coffee-pull-v3",
        "coffee-push-v3",
        "hammer-v3",
        "peg-insert-side-v3",
        "push-wall-v3",
        "soccer-v3",
        "sweep-v3",
        "sweep-into-v3",
    ),
    "hard": (
        "assembly-v3",
        "hand-insert-v3",
        "pick-out-of-hole-v3",
        "pick-place-v3",
        "push-v3",
        "push-back-v3",
    ),
    "very_hard": (
        "shelf-place-v3",
        "disassemble-v3",
        "stick-pull-v3",
        "stick-push-v3",
        "pick-place-wall-v3",
    ),
}

HEADER_RE = re.compile(r" - __main__ - INFO - Evaluating policy on (.+)$")
WAIT_RE = re.compile(r" - root - INFO - Waiting for server at ws://0\.0\.0\.0:(\d+)\.\.\.$")
EPISODE_RE = re.compile(
    r" - __main__ - INFO - Episode (\d+): success=(True|False), "
    r"reward=([-+0-9.eE]+)$"
)
REPLAN_RE = re.compile(
    r" - __main__ - INFO - port: (\d+), replan_steps: (\d+)$"
)
MODEL_RE = re.compile(r" - __main__ - INFO - evaluate_model: (.+)$")
DIFFICULTY_RE = re.compile(
    r" - __main__ - INFO - Task difficulty: (easy|medium|hard|very_hard)$"
)
TASK_RE = re.compile(
    r" - __main__ - INFO - ([a-z0-9-]+-v3) Success rate: ([0-9]+\.[0-9]+)%$"
)
AGGREGATE_RE = re.compile(
    r" - __main__ - INFO - (EASY|MEDIUM|HARD|VERY_HARD)\s+\| "
    r"Success Rate: ([0-9]+\.[0-9]+)% \| Tasks: (\d+) \| Episodes: (\d+)$"
)


def fail(message: str, line_number: int | None = None) -> NoReturn:
    location = f"line {line_number}: " if line_number is not None else ""
    raise ValueError(f"{location}{message}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_percent_text(successes: int, episodes: int) -> str:
    return f"{successes / episodes * 100.0:.2f}"


def validate_constants() -> dict[str, str]:
    if len(EXPECTED_TASKS) != 50 or len(set(EXPECTED_TASKS)) != 50:
        fail("internal expected task order is not exactly 50 unique tasks")
    reverse: dict[str, str] = {}
    for difficulty, task_names in TASK_DIFFICULTY.items():
        for task_name in task_names:
            if task_name in reverse:
                fail(f"internal duplicate difficulty assignment for {task_name}")
            reverse[task_name] = difficulty
    if set(reverse) != set(EXPECTED_TASKS):
        missing = sorted(set(EXPECTED_TASKS) - set(reverse))
        extra = sorted(set(reverse) - set(EXPECTED_TASKS))
        fail(f"internal difficulty coverage mismatch: missing={missing}, extra={extra}")
    return reverse


def summarize(
    log_path: Path,
    *,
    expected_port: int,
    camera_x: float,
    expected_model_name: str | None,
) -> dict[str, object]:
    task_to_difficulty = validate_constants()
    lines = log_path.read_text(encoding="utf-8", errors="strict").splitlines()

    headers = [
        (line_number, match.group(1))
        for line_number, line in enumerate(lines, 1)
        if (match := HEADER_RE.search(line))
    ]
    if len(headers) != 1:
        fail(f"expected exactly one evaluation header, found {len(headers)}")
    model_name = headers[0][1]
    if expected_model_name is not None and model_name != expected_model_name:
        fail(
            f"model name mismatch: expected {expected_model_name!r}, found {model_name!r}",
            headers[0][0],
        )

    waits = [
        (line_number, int(match.group(1)))
        for line_number, line in enumerate(lines, 1)
        if (match := WAIT_RE.search(line))
    ]
    if len(waits) != 1:
        fail(f"expected exactly one websocket wait record, found {len(waits)}")
    if waits[0][1] != expected_port:
        fail(
            f"websocket wait port mismatch: expected {expected_port}, found {waits[0][1]}",
            waits[0][0],
        )

    tasks: list[dict[str, object]] = []
    episodes: list[dict[str, object]] = []
    replan_records: list[tuple[int, int, int]] = []
    evaluated_model: str | None = None
    difficulty: str | None = None
    aggregate_records: list[dict[str, object]] = []
    total_episode_records = 0
    total_replan_records = 0

    for line_number, line in enumerate(lines, 1):
        if match := EPISODE_RE.search(line):
            if evaluated_model is not None or difficulty is not None:
                fail("episode appeared after the task footer had started", line_number)
            if len(replan_records) != len(episodes):
                fail("episode appeared before the preceding episode's replan record", line_number)
            episode_id = int(match.group(1))
            expected_episode_id = len(episodes)
            if episode_id != expected_episode_id:
                fail(
                    f"episode id mismatch: expected {expected_episode_id}, found {episode_id}",
                    line_number,
                )
            reward = float(match.group(3))
            if not math.isfinite(reward):
                fail(f"non-finite reward for episode {episode_id}: {reward}", line_number)
            episodes.append(
                {
                    "episode_id": episode_id,
                    "success": match.group(2) == "True",
                    "reward": reward,
                    "line": line_number,
                }
            )
            total_episode_records += 1
            continue

        if match := REPLAN_RE.search(line):
            if evaluated_model is not None or difficulty is not None:
                fail("replan record appeared after the task footer had started", line_number)
            if len(episodes) != len(replan_records) + 1:
                fail("replan record did not immediately pair with one episode", line_number)
            port = int(match.group(1))
            replan_steps = int(match.group(2))
            if port != expected_port:
                fail(f"port mismatch: expected {expected_port}, found {port}", line_number)
            if replan_steps != EXPECTED_REPLAN_STEPS:
                fail(
                    f"replan mismatch: expected {EXPECTED_REPLAN_STEPS}, found {replan_steps}",
                    line_number,
                )
            replan_records.append((port, replan_steps, line_number))
            total_replan_records += 1
            continue

        if match := MODEL_RE.search(line):
            if evaluated_model is not None:
                fail("duplicate evaluate_model footer in one task block", line_number)
            if difficulty is not None:
                fail("evaluate_model footer appeared after task difficulty", line_number)
            if len(episodes) != EPISODES_PER_TASK or len(replan_records) != EPISODES_PER_TASK:
                fail(
                    "task footer began without exactly 10 paired episode/replan records: "
                    f"episodes={len(episodes)}, replans={len(replan_records)}",
                    line_number,
                )
            evaluated_model = match.group(1)
            if evaluated_model != model_name:
                fail(
                    f"task footer model mismatch: expected {model_name!r}, found {evaluated_model!r}",
                    line_number,
                )
            continue

        if match := DIFFICULTY_RE.search(line):
            if evaluated_model is None:
                fail("task difficulty appeared before evaluate_model footer", line_number)
            if difficulty is not None:
                fail("duplicate task difficulty in one task block", line_number)
            difficulty = match.group(1)
            continue

        if match := TASK_RE.search(line):
            if evaluated_model is None or difficulty is None:
                fail("task summary appeared before a complete task footer", line_number)
            task_id = len(tasks)
            if task_id >= len(EXPECTED_TASKS):
                fail(f"unexpected extra task summary {match.group(1)!r}", line_number)
            task_name = match.group(1)
            expected_task_name = EXPECTED_TASKS[task_id]
            if task_name != expected_task_name:
                fail(
                    f"task order mismatch at task_id {task_id}: "
                    f"expected {expected_task_name!r}, found {task_name!r}",
                    line_number,
                )
            expected_difficulty = task_to_difficulty[task_name]
            if difficulty != expected_difficulty:
                fail(
                    f"difficulty mismatch for {task_name}: "
                    f"expected {expected_difficulty}, found {difficulty}",
                    line_number,
                )
            successes = sum(bool(record["success"]) for record in episodes)
            reported_percent = match.group(2)
            expected_percent = expected_percent_text(successes, EPISODES_PER_TASK)
            if reported_percent != expected_percent:
                fail(
                    f"task summary mismatch for {task_name}: "
                    f"raw={successes}/{EPISODES_PER_TASK}, "
                    f"reported={reported_percent}%",
                    line_number,
                )
            tasks.append(
                {
                    "task_id": task_id,
                    "task": task_name,
                    "difficulty": difficulty,
                    "successes": successes,
                    "episodes": EPISODES_PER_TASK,
                    "success_percent": successes / EPISODES_PER_TASK * 100.0,
                    "episode_ids": [int(record["episode_id"]) for record in episodes],
                    "outcomes": [bool(record["success"]) for record in episodes],
                    "rewards": [float(record["reward"]) for record in episodes],
                }
            )
            episodes = []
            replan_records = []
            evaluated_model = None
            difficulty = None
            continue

        if match := AGGREGATE_RE.search(line):
            aggregate_records.append(
                {
                    "difficulty": match.group(1).lower(),
                    "reported_percent": match.group(2),
                    "tasks": int(match.group(3)),
                    "episodes": int(match.group(4)),
                    "line": line_number,
                }
            )

    if episodes or replan_records or evaluated_model is not None or difficulty is not None:
        fail(
            "log ended inside an incomplete task block: "
            f"episodes={len(episodes)}, replans={len(replan_records)}, "
            f"model_footer={evaluated_model is not None}, difficulty={difficulty!r}"
        )
    if tuple(str(task["task"]) for task in tasks) != EXPECTED_TASKS:
        fail(f"expected exactly 50 ordered task blocks, found {len(tasks)}")
    if total_episode_records != 500:
        fail(f"expected 500 episode records, found {total_episode_records}")
    if total_replan_records != 500:
        fail(f"expected 500 replan records, found {total_replan_records}")

    difficulty_summary: dict[str, dict[str, object]] = {}
    for name in DIFFICULTIES:
        grouped = [task for task in tasks if task["difficulty"] == name]
        successes = sum(int(task["successes"]) for task in grouped)
        episode_count = len(grouped) * EPISODES_PER_TASK
        difficulty_summary[name] = {
            "tasks": len(grouped),
            "episodes": episode_count,
            "successes": successes,
            "success_percent": successes / episode_count * 100.0,
        }

    if len(aggregate_records) != len(DIFFICULTIES):
        fail(
            f"expected {len(DIFFICULTIES)} final difficulty summaries, "
            f"found {len(aggregate_records)}"
        )
    if tuple(str(record["difficulty"]) for record in aggregate_records) != DIFFICULTIES:
        fail(
            "final difficulty summary order mismatch: "
            f"found={[record['difficulty'] for record in aggregate_records]}"
        )
    for aggregate in aggregate_records:
        name = str(aggregate["difficulty"])
        computed = difficulty_summary[name]
        if int(aggregate["tasks"]) != int(computed["tasks"]):
            fail(
                f"final {name} task count mismatch: "
                f"reported={aggregate['tasks']}, computed={computed['tasks']}",
                int(aggregate["line"]),
            )
        if int(aggregate["episodes"]) != int(computed["episodes"]):
            fail(
                f"final {name} episode count mismatch: "
                f"reported={aggregate['episodes']}, computed={computed['episodes']}",
                int(aggregate["line"]),
            )
        expected_percent = expected_percent_text(
            int(computed["successes"]), int(computed["episodes"])
        )
        if str(aggregate["reported_percent"]) != expected_percent:
            fail(
                f"final {name} percentage mismatch: "
                f"reported={aggregate['reported_percent']}%, computed={expected_percent}%",
                int(aggregate["line"]),
            )

    total_successes = sum(int(task["successes"]) for task in tasks)
    difficulty_macro = sum(
        float(difficulty_summary[name]["success_percent"]) for name in DIFFICULTIES
    ) / len(DIFFICULTIES)
    return {
        "schema_version": 1,
        "execution_status": "COMPLETE",
        "validation_status": "STRUCTURALLY_VALID",
        "acceptance_threshold": None,
        "ranking_or_selection_performed": False,
        "log": str(log_path.resolve()),
        "log_sha256": sha256_file(log_path),
        "model_name": model_name,
        "camera": {"x": camera_x, "y": CAMERA_Y, "z": CAMERA_Z, "name": "corner2"},
        "protocol_context": {
            "intended_scanned_factor": "cam_pos_x",
            "known_nuisance_variables": ["fresh_unseeded_MT50_variants"],
            "camera_x_source": "runner_argument_not_present_in_evaluator_log",
            "explicit_seed_cli": None,
            "evaluator_seed_cli_default": 10,
            "env_reset_seed_argument": "10 + task_id * 1000 + episode",
            "metaworld_2_0_reset_seed_argument_effect": "ignored_use_env.seed_instead",
        },
        "port": expected_port,
        "episodes_per_task": EPISODES_PER_TASK,
        "replan_steps": EXPECTED_REPLAN_STEPS,
        "tasks_count": len(tasks),
        "episodes_count": total_episode_records,
        "successes": total_successes,
        "episode_weighted_percent": total_successes / total_episode_records * 100.0,
        "difficulty": difficulty_summary,
        "difficulty_macro_percent": difficulty_macro,
        "task_order_sha256": hashlib.sha256(
            ("\n".join(EXPECTED_TASKS) + "\n").encode("utf-8")
        ).hexdigest(),
        "tasks": tasks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Strictly validate one original 10-episode x 50-task MetaWorld log."
    )
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-port", type=int, required=True)
    parser.add_argument("--camera-x", type=float, required=True)
    parser.add_argument("--expected-model-name")
    args = parser.parse_args()

    if not args.log.is_file():
        parser.error(f"log is not a regular file: {args.log}")
    if not 1 <= args.expected_port <= 65535:
        parser.error(f"expected port is outside 1..65535: {args.expected_port}")
    if not math.isfinite(args.camera_x):
        parser.error(f"camera x must be finite: {args.camera_x}")
    if args.output.exists():
        parser.error(f"refusing to overwrite output: {args.output}")

    result = summarize(
        args.log,
        expected_port=args.expected_port,
        camera_x=args.camera_x,
        expected_model_name=args.expected_model_name,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    temporary = args.output.with_name(f".{args.output.name}.tmp")
    if temporary.exists():
        parser.error(f"refusing to overwrite temporary output: {temporary}")
    temporary.write_text(serialized, encoding="utf-8")
    temporary.replace(args.output)
    print(serialized, end="")


if __name__ == "__main__":
    main()
