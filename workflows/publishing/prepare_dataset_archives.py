#!/usr/bin/env python3
"""Build deterministic tar shards for the two large ALAM pretraining datasets.

The source datasets contain far more files per repository/folder than the
Hugging Face Hub recommendations allow.  This tool packages only the exact
loader-visible files into uncompressed, resumable tar shards.  Download tooling
restores the original directory layout before ALAM training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tarfile
import time
from typing import Iterator


WORKFLOWS_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKFLOWS_ROOT))

from common.paths import REPO_ROOT, repo_path


MANIFEST = Path(__file__).with_name("huggingface_manifest.json")
ARCHIVE_ARTIFACTS = ("calvin_task_abc_d", "oxe_mix10_datasets")


def format_duration(seconds: float) -> str:
    if not math.isfinite(seconds) or seconds < 0:
        return "unknown"
    total = int(seconds)
    hours, remainder = divmod(total, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def sha256_file(path: Path, *, progress_label: str | None = None) -> str:
    digest = hashlib.sha256()
    size = path.stat().st_size
    completed = 0
    started = time.monotonic()
    last_report = started
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
            completed += len(chunk)
            now = time.monotonic()
            if progress_label and now - last_report >= 30:
                elapsed = max(now - started, 1e-9)
                rate = completed / elapsed
                remaining = (size - completed) / rate if rate else math.inf
                print(
                    f"{progress_label}: {completed / 1024**3:.2f}/{size / 1024**3:.2f} GiB "
                    f"({100 * completed / max(size, 1):.1f}%) "
                    f"rate={rate / 1024**2:.1f} MiB/s eta={format_duration(remaining)}",
                    flush=True,
                )
                last_report = now
    return digest.hexdigest()


def safe_leaf(value: str, suffix: str) -> str:
    path = Path(value)
    if path.name != value or path.suffix != suffix or value in {".", ".."}:
        raise ValueError(f"Unsafe dataset metadata filename: {value!r}")
    return value


def calvin_entries(artifact: dict) -> Iterator[tuple[Path, str]]:
    archive = artifact["archive_source"]
    root = Path(archive["root"])
    for split in archive["splits"]:
        split_root = root / split
        metadata_path = split_root / "npz_metadata.json"
        names = json.loads(metadata_path.read_text(encoding="utf-8"))
        yield metadata_path, f"{split}/npz_metadata.json"
        for name in names:
            filename = safe_leaf(name, ".npz")
            yield split_root / filename, f"{split}/{filename}"


def oxe_entries(artifact: dict) -> Iterator[tuple[Path, str]]:
    archive = artifact["archive_source"]
    root = Path(archive["root"])
    for relative_value in archive["dataset_display_paths"]:
        relative = Path(relative_value)
        display_root = root / relative
        metadata_path = display_root / "video_metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        yield metadata_path, f"{relative.as_posix()}/video_metadata.json"
        for split in sorted(metadata):
            for record in metadata[split]["videos"]:
                filename = safe_leaf(record[0], ".mp4")
                yield display_root / filename, f"{relative.as_posix()}/{filename}"


def artifact_entries(artifact: dict) -> Iterator[tuple[Path, str]]:
    if artifact["id"] == "calvin_task_abc_d":
        yield from calvin_entries(artifact)
    elif artifact["id"] == "oxe_mix10_datasets":
        yield from oxe_entries(artifact)
    else:
        raise ValueError(f"No archive enumerator for {artifact['id']}")


def tar_member_bytes(file_bytes: int) -> int:
    return 512 + math.ceil(file_bytes / 512) * 512


def group_signature(entries: list[tuple[Path, str, int, int]]) -> str:
    digest = hashlib.sha256()
    for _, archive_name, size, mtime_ns in entries:
        digest.update(archive_name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(size).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(mtime_ns).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def validate_recoverable_shard(
    target: Path,
    entries: list[tuple[Path, str, int, int]],
) -> None:
    """Validate a tar that was atomically renamed before its state was saved.

    ``build_shard`` renames ``*.tar.partial`` only after ``tarfile`` has closed
    the archive.  A process interruption can therefore leave a complete tar
    without the small JSON state file.  Validate its complete member sequence
    before trusting and hashing it; a partial archive is never recovered.
    """

    source_bytes = sum(size for _, _, size, _ in entries)
    completed_bytes = 0
    started = time.monotonic()
    last_report = started
    with tarfile.open(target, mode="r:") as archive:
        members = iter(archive)
        for index, (source, archive_name, size, _) in enumerate(entries, 1):
            member = next(members, None)
            if member is None:
                raise RuntimeError(
                    f"Interrupted shard is truncated at member {index}/{len(entries)}: {target}"
                )
            if not member.isfile() or member.name != archive_name or member.size != size:
                raise RuntimeError(
                    "Interrupted shard does not match its source group at "
                    f"member {index}/{len(entries)}: {target}"
                )
            archived_stream = archive.extractfile(member)
            if archived_stream is None:
                raise RuntimeError(f"Cannot read archived member during recovery: {archive_name}")
            with archived_stream, source.open("rb") as source_stream:
                while True:
                    source_chunk = source_stream.read(8 * 1024 * 1024)
                    archived_chunk = archived_stream.read(len(source_chunk) or 1)
                    if not source_chunk:
                        if archived_chunk:
                            raise RuntimeError(
                                f"Interrupted shard member has extra payload: {archive_name}"
                            )
                        break
                    if archived_chunk != source_chunk:
                        raise RuntimeError(
                            f"Interrupted shard payload differs from source: {archive_name}"
                        )
                    completed_bytes += len(source_chunk)
                    now = time.monotonic()
                    if now - last_report >= 30:
                        elapsed = max(now - started, 1e-9)
                        rate = completed_bytes / elapsed
                        remaining = (source_bytes - completed_bytes) / rate if rate else math.inf
                        print(
                            f"Validate recovered {target.name}: members={index:,}/{len(entries):,} "
                            f"source={completed_bytes / 1024**3:.2f}/{source_bytes / 1024**3:.2f} GiB "
                            f"({100 * completed_bytes / max(source_bytes, 1):.1f}%) "
                            f"rate={rate / 1024**2:.1f} MiB/s eta={format_duration(remaining)}",
                            flush=True,
                        )
                        last_report = now
        if next(members, None) is not None:
            raise RuntimeError(f"Interrupted shard contains unexpected extra members: {target}")


def build_shard(
    staging: Path,
    build_state: Path,
    index: int,
    entries: list[tuple[Path, str, int, int]],
) -> dict:
    name = f"data-{index:05d}.tar"
    target = staging / name
    state_path = build_state / f"{name}.json"
    signature = group_signature(entries)
    if target.is_file() and state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if (
            state.get("group_signature") == signature
            and state.get("members") == len(entries)
            and state.get("archive_bytes") == target.stat().st_size
        ):
            print(f"Reuse completed shard: {target}", flush=True)
            return state
        raise RuntimeError(f"Existing shard does not match its source group; move the staging directory: {target}")
    if target.is_file() and not state_path.exists():
        partial = target.with_suffix(".tar.partial")
        if partial.exists():
            raise RuntimeError(f"Both completed and partial shard files exist; inspect them manually: {target}")
        print(f"Recover completed shard state: {target}", flush=True)
        validate_recoverable_shard(target, entries)
        source_bytes = sum(size for _, _, size, _ in entries)
        state = {
            "file": name,
            "sha256": sha256_file(target, progress_label=f"Hash {name}"),
            "archive_bytes": target.stat().st_size,
            "source_bytes": source_bytes,
            "members": len(entries),
            "group_signature": signature,
        }
        atomic_json(state_path, state)
        print(f"Recovered completed shard state: {target}", flush=True)
        return state
    if target.exists() or state_path.exists():
        raise RuntimeError(f"Incomplete shard state; move the staging directory before rebuilding: {target}")

    partial = target.with_suffix(".tar.partial")
    if partial.exists():
        partial.unlink()
    estimated_bytes = 1024 + sum(tar_member_bytes(size) for _, _, size, _ in entries)
    free_bytes = shutil.disk_usage(staging).free
    required_bytes = estimated_bytes + 1024**3
    if free_bytes < required_bytes:
        raise OSError(
            f"Insufficient staging space for {name}: free={free_bytes:,}, required={required_bytes:,}"
        )
    source_bytes = sum(size for _, _, size, _ in entries)
    print(
        f"Build {name}: members={len(entries):,} source={source_bytes / 1024**3:.2f} GiB",
        flush=True,
    )
    started = time.monotonic()
    last_report = started
    completed_bytes = 0
    with tarfile.open(partial, mode="w", format=tarfile.PAX_FORMAT) as archive:
        for member_index, (source, archive_name, size, _) in enumerate(entries, 1):
            tarinfo = archive.gettarinfo(str(source), arcname=archive_name)
            if not tarinfo.isfile():
                raise ValueError(f"Only regular dataset files can be archived: {source}")
            tarinfo.uid = 0
            tarinfo.gid = 0
            tarinfo.uname = ""
            tarinfo.gname = ""
            tarinfo.mtime = 0
            tarinfo.mode = 0o644
            with source.open("rb") as stream:
                archive.addfile(tarinfo, stream)
            completed_bytes += size
            now = time.monotonic()
            if now - last_report >= 30:
                elapsed = max(now - started, 1e-9)
                rate = completed_bytes / elapsed
                remaining = (source_bytes - completed_bytes) / rate if rate else math.inf
                print(
                    f"Build {name}: members={member_index:,}/{len(entries):,} "
                    f"source={completed_bytes / 1024**3:.2f}/{source_bytes / 1024**3:.2f} GiB "
                    f"({100 * completed_bytes / max(source_bytes, 1):.1f}%) "
                    f"rate={rate / 1024**2:.1f} MiB/s eta={format_duration(remaining)}",
                    flush=True,
                )
                last_report = now
    os.replace(partial, target)
    elapsed = max(time.monotonic() - started, 1e-9)
    print(
        f"Built {name}: {target.stat().st_size / 1024**3:.2f} GiB in "
        f"{format_duration(elapsed)} ({source_bytes / elapsed / 1024**2:.1f} MiB/s); hashing",
        flush=True,
    )
    state = {
        "file": name,
        "sha256": sha256_file(target, progress_label=f"Hash {name}"),
        "archive_bytes": target.stat().st_size,
        "source_bytes": source_bytes,
        "members": len(entries),
        "group_signature": signature,
    }
    atomic_json(state_path, state)
    return state


def verify_complete(staging: Path) -> None:
    layout = json.loads((staging / "DATASET_LAYOUT.json").read_text(encoding="utf-8"))
    lines = (staging / "ARCHIVE_MANIFEST.sha256").read_text(encoding="utf-8").splitlines()
    if len(lines) != int(layout["archive_shards"]):
        raise AssertionError("Archive manifest/layout shard-count mismatch")
    expected_names: set[str] = set()
    for index, line in enumerate(lines, 1):
        expected, name = line.split("  ", 1)
        if Path(name).name != name or not name.startswith("data-") or not name.endswith(".tar"):
            raise ValueError(f"Unsafe archive manifest entry: {name}")
        expected_names.add(name)
        print(f"Verify shard {index}/{len(lines)}: {name}", flush=True)
        found = sha256_file(staging / name, progress_label=f"Verify {name}")
        if found != expected:
            raise AssertionError(f"Archive hash mismatch: {name}")
    actual_names = {path.name for path in staging.glob("data-*.tar") if path.is_file()}
    if expected_names != actual_names:
        raise AssertionError(
            f"Archive file-set mismatch: missing={sorted(expected_names - actual_names)} "
            f"extra={sorted(actual_names - expected_names)}"
        )
    actual_bytes = sum((staging / name).stat().st_size for name in expected_names)
    if actual_bytes != int(layout["archive_bytes"]):
        raise AssertionError("Archive byte-count/layout mismatch")
    print(f"Archive verification PASS: {staging} ({len(lines)} shards)", flush=True)


def verify_staged_metadata(staging: Path) -> None:
    """Fast resume gate using the manifest, build receipts, and file metadata.

    A completed first run already hashed every shard.  Resume mode avoids
    reading hundreds of GiB again, but it must still reject missing, renamed,
    truncated, or receipt-mismatched archives before the network upload starts.
    """

    layout = json.loads((staging / "DATASET_LAYOUT.json").read_text(encoding="utf-8"))
    lines = (staging / "ARCHIVE_MANIFEST.sha256").read_text(encoding="utf-8").splitlines()
    expected_names: set[str] = set()
    archive_bytes = 0
    source_bytes = 0
    source_members = 0
    for line in lines:
        expected_hash, name = line.split("  ", 1)
        if Path(name).name != name or not name.startswith("data-") or not name.endswith(".tar"):
            raise ValueError(f"Unsafe archive manifest entry: {name}")
        if name in expected_names:
            raise ValueError(f"Duplicate archive manifest entry: {name}")
        expected_names.add(name)
        receipt_path = staging / ".build" / f"{name}.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt.get("file") != name or receipt.get("sha256") != expected_hash:
            raise AssertionError(f"Archive build receipt mismatch: {name}")
        archive = staging / name
        found_bytes = archive.stat().st_size
        if found_bytes != int(receipt["archive_bytes"]):
            raise AssertionError(f"Archive byte-count/build-receipt mismatch: {name}")
        archive_bytes += found_bytes
        source_bytes += int(receipt["source_bytes"])
        source_members += int(receipt["members"])
    actual_names = {path.name for path in staging.glob("data-*.tar") if path.is_file()}
    if expected_names != actual_names:
        raise AssertionError(
            f"Archive file-set mismatch: missing={sorted(expected_names - actual_names)} "
            f"extra={sorted(actual_names - expected_names)}"
        )
    if list(staging.glob("data-*.tar.partial")):
        raise AssertionError(f"Partial archives remain beside a completed dataset layout: {staging}")
    expected_totals = {
        "archive_shards": len(lines),
        "archive_bytes": archive_bytes,
        "source_bytes": source_bytes,
        "source_members": source_members,
    }
    for key, found in expected_totals.items():
        if int(layout[key]) != found:
            raise AssertionError(f"Archive {key}/layout mismatch: expected {layout[key]}, found {found}")
    print(f"Archive resume metadata PASS: {staging} ({len(lines)} shards)", flush=True)


def build_artifact(artifact: dict, verify_existing: bool) -> None:
    staging = repo_path(artifact["upload_sources"][0]["path"])
    complete_manifest = staging / "ARCHIVE_MANIFEST.sha256"
    layout_path = staging / "DATASET_LAYOUT.json"
    if complete_manifest.is_file() and layout_path.is_file():
        verify_staged_metadata(staging)
        if verify_existing:
            verify_complete(staging)
        else:
            print(f"Reuse completed dataset staging: {staging}", flush=True)
        return

    staging.mkdir(parents=True, exist_ok=True)
    build_state = staging / ".build"
    build_state.mkdir(exist_ok=True)
    target_bytes = int(artifact["archive_source"]["shard_size_bytes"])
    shard_records: list[dict] = []
    group: list[tuple[Path, str, int, int]] = []
    group_bytes = 1024
    total_source_bytes = 0
    total_members = 0
    enumeration_started = time.monotonic()
    enumeration_last_report = enumeration_started

    def flush() -> None:
        nonlocal group, group_bytes
        if not group:
            return
        shard_records.append(build_shard(staging, build_state, len(shard_records), group))
        group = []
        group_bytes = 1024

    for source, archive_name in artifact_entries(artifact):
        if not source.is_file():
            raise FileNotFoundError(f"Dataset metadata references a missing file: {source}")
        stat = source.stat()
        size = stat.st_size
        contribution = tar_member_bytes(size)
        if group and group_bytes + contribution > target_bytes:
            flush()
        group.append((source, archive_name, size, stat.st_mtime_ns))
        group_bytes += contribution
        total_source_bytes += size
        total_members += 1
        now = time.monotonic()
        if now - enumeration_last_report >= 30:
            print(
                f"Enumerate {artifact['id']}: members={total_members:,} "
                f"source={total_source_bytes / 1024**3:.2f} GiB "
                f"elapsed={format_duration(now - enumeration_started)}",
                flush=True,
            )
            enumeration_last_report = now
    flush()

    layout = {
        "schema_version": 1,
        "archive_format": "tar_shards_v1",
        "artifact_id": artifact["id"],
        "repo_id": artifact["repo_id"],
        "source_members": total_members,
        "source_bytes": total_source_bytes,
        "archive_shards": len(shard_records),
        "archive_bytes": sum(item["archive_bytes"] for item in shard_records),
        "statistics": artifact["statistics"],
    }
    atomic_json(layout_path, layout)
    manifest_text = "".join(f"{item['sha256']}  {item['file']}\n" for item in shard_records)
    temporary_manifest = complete_manifest.with_suffix(".sha256.partial")
    temporary_manifest.write_text(manifest_text, encoding="utf-8")
    os.replace(temporary_manifest, complete_manifest)
    print(json.dumps(layout, indent=2, ensure_ascii=False))
    verify_complete(staging)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", action="append", choices=ARCHIVE_ARTIFACTS, default=[])
    parser.add_argument("--execute", action="store_true", help="Create tar shards; otherwise print the plan")
    parser.add_argument("--verify-existing", action="store_true", help="Re-hash an already completed staging tree")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifacts = json.loads(MANIFEST.read_text(encoding="utf-8"))["artifacts"]
    selected_ids = args.artifact or list(ARCHIVE_ARTIFACTS)
    by_id = {artifact["id"]: artifact for artifact in artifacts}
    for artifact_id in selected_ids:
        artifact = by_id[artifact_id]
        plan = {
            "artifact": artifact_id,
            "repo_id": artifact["repo_id"],
            "source": artifact["archive_source"],
            "staging": str(repo_path(artifact["upload_sources"][0]["path"])),
            "format": artifact["archive_format"],
        }
        print(json.dumps(plan, indent=2, ensure_ascii=False))
        if args.execute:
            build_artifact(artifact, args.verify_existing)
    if not args.execute:
        print("Dry run only. Add --execute to create the upload tar shards.")


if __name__ == "__main__":
    main()
