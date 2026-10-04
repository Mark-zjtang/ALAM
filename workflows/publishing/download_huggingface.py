#!/usr/bin/env python3
"""Download ALAM models or datasets into the documented layout."""

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

WORKFLOWS_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKFLOWS_ROOT))

from common.paths import REPO_ROOT, repo_path

MANIFEST = Path(__file__).with_name("huggingface_manifest.json")


def format_duration(seconds: float) -> str:
    if not math.isfinite(seconds) or seconds < 0:
        return "unknown"
    total = int(seconds)
    hours, remainder = divmod(total, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def ensure_download_directory(path: Path) -> None:
    if path.is_symlink():
        raise FileExistsError(f"Refusing to download into a symlink target: {path}")
    path.mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path, *, label: str | None = None) -> str:
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
            if label and now - last_report >= 30:
                elapsed = max(now - started, 1e-9)
                rate = completed / elapsed
                remaining = (size - completed) / rate if rate else math.inf
                print(
                    f"[restore] hash {label}: {completed / 1024**3:.2f}/{size / 1024**3:.2f} GiB "
                    f"({100 * completed / max(size, 1):.1f}%) "
                    f"rate={rate / 1024**2:.1f} MiB/s eta={format_duration(remaining)}",
                    flush=True,
                )
                last_report = now
    return digest.hexdigest()


def verify_archive_manifest(staging: Path) -> list[tuple[str, Path]]:
    manifest = staging / "ARCHIVE_MANIFEST.sha256"
    if not manifest.is_file():
        raise FileNotFoundError(f"Missing dataset archive manifest: {manifest}")
    verified = []
    lines = manifest.read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(lines, 1):
        expected, filename = line.split("  ", 1)
        archive = staging / filename
        if archive.parent != staging or archive.suffix != ".tar" or not archive.is_file():
            raise ValueError(f"Unsafe or missing archive path: {filename}")
        print(f"[restore] verify shard {index}/{len(lines)}: {filename}", flush=True)
        found = sha256_file(archive, label=filename)
        if found != expected:
            raise AssertionError(f"Dataset archive hash mismatch: {archive}")
        verified.append((expected, archive))
    if not verified:
        raise AssertionError(f"Dataset archive manifest is empty: {manifest}")
    return verified


def extract_archive_dataset(staging: Path, target: Path, repo_id: str) -> None:
    layout_path = staging / "DATASET_LAYOUT.json"
    if not layout_path.is_file():
        raise FileNotFoundError(f"Missing dataset layout: {layout_path}")
    layout = json.loads(layout_path.read_text(encoding="utf-8"))
    if layout.get("repo_id") != repo_id:
        raise AssertionError(f"Dataset layout repository mismatch: {layout.get('repo_id')} != {repo_id}")
    archives = verify_archive_manifest(staging)
    if len(archives) != int(layout["archive_shards"]):
        raise AssertionError("Dataset archive-count/layout mismatch")
    layout_digest = sha256_file(layout_path)
    marker_name = ".alam_hf_dataset.json"
    marker = target / marker_name
    if marker.is_file():
        current = json.loads(marker.read_text(encoding="utf-8"))
        if current.get("layout_sha256") == layout_digest and current.get("repo_id") == repo_id:
            print(f"Dataset already restored for this archive manifest: {target}")
            return
        raise RuntimeError(f"Existing dataset marker does not match the downloaded revision: {marker}")
    if target.is_symlink():
        raise FileExistsError(f"Refusing to replace a dataset symlink: {target}")
    if target.exists():
        raise FileExistsError(f"Refusing to merge a downloaded dataset into an unmarked directory: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    extracting = target.parent / f".{target.name}.extracting"
    extracting.mkdir(parents=True, exist_ok=True)
    state_path = extracting / ".extraction_state.json"
    if state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("repo_id") != repo_id or state.get("layout_sha256") != layout_digest:
            raise RuntimeError(
                f"Extraction state belongs to another repository revision; inspect or move: {extracting}"
            )
    else:
        state = {
            "schema_version": 1,
            "repo_id": repo_id,
            "layout_sha256": layout_digest,
            "shards": {},
        }
        atomic_json(state_path, state)
    archive_names = {archive.name for _, archive in archives}
    stale_state = sorted(set(state["shards"]) - archive_names)
    if stale_state:
        raise RuntimeError(f"Extraction state contains stale shards: {stale_state}")
    for expected_hash, archive_path in archives:
        completed = state["shards"].get(archive_path.name)
        if completed and completed.get("sha256") == expected_hash:
            print(f"Reuse extracted shard: {archive_path.name}")
            continue
        members_extracted = 0
        extracted_bytes = 0
        started = time.monotonic()
        last_report = started
        print(f"[restore] extract {archive_path.name}", flush=True)
        with tarfile.open(archive_path, mode="r:") as archive:
            for member in archive:
                if not member.isfile():
                    raise ValueError(f"Only regular files are allowed in dataset shards: {member.name}")
                destination = (extracting / member.name).resolve()
                if not destination.is_relative_to(extracting.resolve()):
                    raise ValueError(f"Archive member escapes the dataset root: {member.name}")
                archive.extract(member, path=extracting)
                members_extracted += 1
                extracted_bytes += member.size
                now = time.monotonic()
                if now - last_report >= 30:
                    elapsed = max(now - started, 1e-9)
                    print(
                        f"[restore] extract {archive_path.name}: members={members_extracted:,} "
                        f"data={extracted_bytes / 1024**3:.2f} GiB "
                        f"rate={extracted_bytes / elapsed / 1024**2:.1f} MiB/s",
                        flush=True,
                    )
                    last_report = now
        state["shards"][archive_path.name] = {
            "sha256": expected_hash,
            "members": members_extracted,
        }
        atomic_json(state_path, state)
        print(
            f"Extracted {archive_path.name}: members={members_extracted:,} "
            f"data={extracted_bytes / 1024**3:.2f} GiB",
            flush=True,
        )
    total_members = sum(int(value["members"]) for value in state["shards"].values())
    if total_members != int(layout["source_members"]):
        raise AssertionError(
            f"Dataset member count mismatch: expected {layout['source_members']}, extracted {total_members}"
        )
    state_path.unlink(missing_ok=True)
    atomic_json(
        extracting / marker_name,
        {
            "repo_id": repo_id,
            "layout_sha256": layout_digest,
            "source_members": total_members,
            "archive_shards": len(archives),
        },
    )
    os.replace(extracting, target)
    print(f"Dataset restore PASS: {repo_id} -> {target}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--org", help="Override the owner recorded in the download manifest")
    parser.add_argument("--artifact", action="append", default=[], help="Artifact ID; defaults to all three model repos")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifacts = json.loads(MANIFEST.read_text(encoding="utf-8"))["artifacts"]
    by_id = {artifact["id"]: artifact for artifact in artifacts}
    ids = args.artifact or ["alam_pretrain", "alam_plus_pi_metaworld_mt50", "alam_plus_pi_libero"]
    unknown = sorted(set(ids) - set(by_id))
    if unknown:
        raise ValueError(f"Unknown artifact IDs: {', '.join(unknown)}")

    from huggingface_hub import snapshot_download

    for artifact_id in ids:
        artifact = by_id[artifact_id]
        repo_id = artifact.get("repo_id")
        if repo_id is None:
            raise ValueError(f"Hugging Face repository ID is still pending for: {artifact_id}")
        if args.org:
            _, name = repo_id.split("/", 1)
            repo_id = f"{args.org}/{name}"
        target = repo_path(artifact["download_path"])
        archive_format = artifact.get("archive_format")
        download_target = repo_path(artifact["download_staging_path"]) if archive_format else target
        ensure_download_directory(download_target)
        print(f"Downloading {repo_id} -> {download_target}")
        snapshot_download(
            repo_id=repo_id,
            repo_type=artifact["repo_type"],
            local_dir=str(download_target),
            allow_patterns=artifact.get("allow_patterns"),
            ignore_patterns=["README.md", ".gitattributes"],
        )
        local_cache = download_target / ".cache" / "huggingface"
        if local_cache.is_dir():
            shutil.rmtree(local_cache)
            try:
                local_cache.parent.rmdir()
            except OSError:
                pass
        if archive_format:
            if archive_format != "tar_shards_v1":
                raise ValueError(f"Unsupported dataset archive format: {archive_format}")
            extract_archive_dataset(download_target, target, repo_id)
            continue
        if alias_value := artifact.get("alias_path"):
            alias = repo_path(alias_value)
            alias.parent.mkdir(parents=True, exist_ok=True)
            desired = os.path.relpath(target, alias.parent)
            if alias.is_symlink():
                if os.readlink(alias) != desired:
                    alias.unlink()
            elif alias.exists():
                raise FileExistsError(f"Refusing to replace non-symlink alias: {alias}")
            if not alias.exists():
                alias.symlink_to(desired)
            print(f"Alias: {alias} -> {desired}")


if __name__ == "__main__":
    main()
