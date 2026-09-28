#!/usr/bin/env python3
"""Verify remote Hugging Face files against the exact local upload sources."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time


WORKFLOWS_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKFLOWS_ROOT))

from common.paths import REPO_ROOT, repo_path


MANIFEST = Path(__file__).with_name("huggingface_manifest.json")


def ignored(path: Path) -> bool:
    return (
        any(part in {".git", ".cache", ".build", "__pycache__"} for part in path.parts)
        or path.name == ".gitattributes"
        or path.name.endswith(".partial")
    )


def expected_files(artifact: dict) -> dict[str, Path]:
    expected = {"README.md": repo_path(artifact["card"])}
    for source in artifact["upload_sources"]:
        root_value = Path(source["path"])
        root = root_value if root_value.is_absolute() else repo_path(root_value)
        if not root.is_dir():
            raise FileNotFoundError(f"Missing local upload source: {root}")
        prefix = "" if source["path_in_repo"] == "." else source["path_in_repo"].strip("/") + "/"
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            relative = path.relative_to(root)
            if ignored(relative) or (not prefix and relative.as_posix() == "README.md"):
                continue
            remote_path = prefix + relative.as_posix()
            if remote_path in expected:
                raise ValueError(f"Duplicate remote upload path: {remote_path}")
            expected[remote_path] = path
    return expected


def format_duration(seconds: float) -> str:
    if not math.isfinite(seconds) or seconds < 0:
        return "unknown"
    total = int(seconds)
    hours, remainder = divmod(total, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def progress_message(label: str, completed: int, size: int, started: float) -> str:
    elapsed = max(time.monotonic() - started, 1e-9)
    rate = completed / elapsed
    remaining = (size - completed) / rate if rate else math.inf
    return (
        f"[verify] {label}: {completed / 1024**3:.2f}/{size / 1024**3:.2f} GiB "
        f"({100 * completed / max(size, 1):.1f}%) "
        f"rate={rate / 1024**2:.1f} MiB/s eta={format_duration(remaining)}"
    )


def sha256_file(path: Path, *, label: str) -> str:
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
            if now - last_report >= 30:
                print(progress_message(label, completed, size, started), flush=True)
                last_report = now
    return digest.hexdigest()


def git_blob_id(path: Path, size: int, *, label: str) -> str:
    digest = hashlib.sha1()
    digest.update(f"blob {size}\0".encode("ascii"))
    completed = 0
    started = time.monotonic()
    last_report = started
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
            completed += len(chunk)
            now = time.monotonic()
            if now - last_report >= 30:
                print(progress_message(label, completed, size, started), flush=True)
                last_report = now
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--org", help="Override the owner recorded in the manifest")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifacts = json.loads(MANIFEST.read_text(encoding="utf-8"))["artifacts"]
    by_id = {artifact["id"]: artifact for artifact in artifacts}
    if args.artifact not in by_id:
        raise ValueError(f"Unknown artifact ID: {args.artifact}")
    artifact = by_id[args.artifact]
    repo_id = artifact.get("repo_id")
    if repo_id is None:
        raise ValueError(f"Repository ID is pending for: {args.artifact}")
    if args.org:
        _, name = repo_id.split("/", 1)
        repo_id = f"{args.org}/{name}"

    from huggingface_hub import HfApi
    from huggingface_hub.hf_api import RepoFile

    api = HfApi()
    print(f"[verify] repository={repo_id}: enumerate local and remote files", flush=True)
    expected = expected_files(artifact)
    remote = {
        item.path: item
        for item in api.list_repo_tree(repo_id, repo_type=artifact["repo_type"], recursive=True, expand=True)
        if isinstance(item, RepoFile) and item.path != ".gitattributes"
    }
    missing = sorted(set(expected) - set(remote))
    extra = sorted(set(remote) - set(expected))
    if missing or extra:
        raise AssertionError(f"Remote file-set mismatch for {repo_id}: missing={missing} extra={extra}")

    total_files = len(expected)
    for index, (remote_path, local_path) in enumerate(expected.items(), 1):
        print(f"[verify] {repo_id}: file {index}/{total_files} {remote_path}", flush=True)
        info = remote[remote_path]
        size = local_path.stat().st_size
        if info.size != size:
            raise AssertionError(f"Remote size mismatch for {remote_path}: local={size} remote={info.size}")
        if info.lfs is not None:
            found = sha256_file(local_path, label=f"{repo_id}/{remote_path}")
            if found != info.lfs.sha256:
                raise AssertionError(f"Remote LFS hash mismatch for {remote_path}")
        elif info.xet_hash is not None:
            raise AssertionError(
                f"Hub returned an Xet content hash without the Git-LFS SHA-256 for {remote_path}; "
                "exact local-to-remote content verification is unavailable"
            )
        else:
            found = git_blob_id(local_path, size, label=f"{repo_id}/{remote_path}")
            if found != info.blob_id:
                raise AssertionError(f"Remote Git blob mismatch for {remote_path}")
    revision = api.repo_info(repo_id, repo_type=artifact["repo_type"]).sha
    print(
        json.dumps(
            {
                "artifact": args.artifact,
                "repo_id": repo_id,
                "revision": revision,
                "files": len(expected),
                "bytes": sum(path.stat().st_size for path in expected.values()),
                "status": "PASS",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
