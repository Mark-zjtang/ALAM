#!/usr/bin/env python3
"""Report local dataset-shard packaging progress without touching the uploader."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time


REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = Path(__file__).with_name("huggingface_manifest.json")


def format_gib(value: int) -> str:
    return f"{value / 1024**3:.2f} GiB"


def load_archives() -> list[tuple[str, Path, int]]:
    artifacts = json.loads(MANIFEST.read_text(encoding="utf-8"))["artifacts"]
    result = []
    for artifact in artifacts:
        if artifact.get("archive_format") != "tar_shards_v1":
            continue
        source = artifact["upload_sources"][0]["path"]
        staging = Path(source)
        if not staging.is_absolute():
            staging = REPO_ROOT / staging
        result.append((artifact["id"], staging, int(artifact["archive_source"]["shard_size_bytes"])))
    return result


def process_stage() -> str:
    """Best-effort read-only detection of the local publication stage."""

    proc = Path("/proc")
    if not proc.is_dir():
        return "unknown"
    found: set[str] = set()
    for entry in proc.iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            command = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(
                "utf-8", errors="replace"
            )
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if "publish_huggingface.py" in command:
            found.add("huggingface-upload")
        elif "prepare_dataset_archives.py" in command:
            found.add("local-packaging")
        elif "upload_created_repositories.sh" in command:
            found.add("orchestrator")
    for stage in ("huggingface-upload", "local-packaging", "orchestrator"):
        if stage in found:
            return stage
    return "idle"


def snapshot(archives: list[tuple[str, Path, int]], stage: str) -> tuple[int, list[str]]:
    total = 0
    lines = []
    for artifact_id, staging, target in archives:
        completed = sorted(staging.glob("data-*.tar")) if staging.is_dir() else []
        partials = sorted(staging.glob("data-*.tar.partial")) if staging.is_dir() else []
        artifact_bytes = sum(path.stat().st_size for path in completed + partials)
        total += artifact_bytes
        detail = f"{artifact_id}: completed_shards={len(completed)} staged={format_gib(artifact_bytes)}"
        if partials:
            partial = partials[-1]
            size = partial.stat().st_size
            detail += (
                f" active={partial.name} {format_gib(size)}/{format_gib(target)}"
                f" ({100 * size / target:.1f}%)"
            )
        elif (staging / "ARCHIVE_MANIFEST.sha256").is_file():
            detail += " archive_status=ready"
        elif completed:
            status = "hashing-or-between-shards" if stage == "local-packaging" else "incomplete-idle"
            detail += f" archive_status={status}"
        else:
            detail += " archive_status=not-started"
        lines.append(detail)
    return total, lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=float, default=5.0, help="Sampling interval in seconds")
    parser.add_argument("--once", action="store_true", help="Print one snapshot without a speed estimate")
    args = parser.parse_args()
    if args.interval <= 0:
        raise ValueError("--interval must be positive")

    archives = load_archives()
    previous_bytes: int | None = None
    previous_time: float | None = None
    while True:
        now = time.monotonic()
        stage = process_stage()
        total, lines = snapshot(archives, stage)
        rate = None
        if previous_bytes is not None and previous_time is not None:
            elapsed = max(now - previous_time, 1e-9)
            rate = (total - previous_bytes) / elapsed / 1024**2
        stamp = time.strftime("%Y-%m-%d %H:%M:%S %Z")
        rate_text = "warming-up" if rate is None else f"{rate:.2f} MiB/s"
        print(
            f"[{stamp}] process_stage={stage} archive_bytes={format_gib(total)} rate={rate_text}",
            flush=True,
        )
        for line in lines:
            print(f"  {line}", flush=True)
        if args.once:
            return
        previous_bytes = total
        previous_time = now
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
