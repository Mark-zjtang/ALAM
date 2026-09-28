#!/usr/bin/env python3
"""Validate and optionally publish release artifacts to Hugging Face."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

WORKFLOWS_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKFLOWS_ROOT))

from common.paths import REPO_ROOT, repo_path

MANIFEST = Path(__file__).with_name("huggingface_manifest.json")


def load_artifacts() -> list[dict]:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))["artifacts"]


def select_artifacts(ids: list[str]) -> list[dict]:
    artifacts = load_artifacts()
    by_id = {artifact["id"]: artifact for artifact in artifacts}
    unknown = sorted(set(ids) - set(by_id))
    if unknown:
        raise ValueError(f"Unknown artifact IDs: {', '.join(unknown)}")
    return [by_id[item] for item in ids] if ids else artifacts


def source_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else repo_path(path)


def resolved_repo_id(artifact: dict, owner_override: str | None) -> str | None:
    repo_id = artifact.get("repo_id")
    if repo_id is None:
        return None
    if owner_override:
        _, name = repo_id.split("/", 1)
        return f"{owner_override}/{name}"
    return repo_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--org", help="Override the owner recorded in the publication manifest")
    parser.add_argument("--artifact", action="append", default=[], help="Artifact ID; repeat to select several")
    parser.add_argument("--private", action="store_true", help="Create new repositories as private")
    parser.add_argument("--acknowledge-rights", action="store_true", help="Confirm required licenses/rights were reviewed")
    parser.add_argument("--execute", action="store_true", help="Perform uploads; otherwise only print and validate the plan")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected = select_artifacts(args.artifact)
    plan = []
    for artifact in selected:
        repo_id = resolved_repo_id(artifact, args.org)
        sources = []
        for item in artifact["upload_sources"]:
            path = source_path(item["path"])
            if not path.is_dir():
                if args.execute or not artifact.get("archive_format"):
                    raise FileNotFoundError(f"Missing upload source for {artifact['id']}: {path}")
            sources.append({
                "path": str(path),
                "path_in_repo": item["path_in_repo"],
                "upload_method": item.get("upload_method", "folder"),
                "exists": path.is_dir(),
            })
        if repo_id is None:
            rights = "BLOCKED until the downstream dataset repository is created and repo_id is recorded"
        elif artifact.get("redistribution_review_required") and not args.acknowledge_rights:
            rights = "BLOCKED until --acknowledge-rights is supplied"
        else:
            rights = "acknowledged or not flagged"
        plan.append({
            "id": artifact["id"],
            "repo_id": repo_id or "<PENDING_REPOSITORY>",
            "repo_type": artifact["repo_type"],
            "sources": sources,
            "rights": rights,
        })
    print(json.dumps(plan, indent=2, ensure_ascii=False))
    if not args.execute:
        print("Dry run only. Add --execute after reviewing repository IDs, cards, and rights.")
        return
    pending = [entry["id"] for entry, artifact in zip(plan, selected, strict=True) if artifact.get("repo_id") is None]
    if pending:
        raise ValueError(f"Hugging Face repository ID is still pending for: {', '.join(pending)}")
    blocked = [entry["id"] for entry, artifact in zip(plan, selected, strict=True) if artifact.get("redistribution_review_required") and not args.acknowledge_rights]
    if blocked:
        raise PermissionError(f"Redistribution review not acknowledged for: {', '.join(blocked)}")
    from huggingface_hub import HfApi, get_token

    token = os.environ.get("HF_TOKEN") or get_token()
    if not token:
        raise EnvironmentError("No Hugging Face token found; run 'hf auth login' or export HF_TOKEN")
    api = HfApi(token=token)
    for entry, artifact in zip(plan, selected, strict=True):
        print(
            f"[publish] repository={entry['repo_id']} type={entry['repo_type']}: create/check",
            flush=True,
        )
        api.create_repo(
            repo_id=entry["repo_id"],
            repo_type=entry["repo_type"],
            private=args.private,
            exist_ok=True,
        )
        card = repo_path(artifact["card"])
        print(f"[publish] repository={entry['repo_id']}: upload README.md", flush=True)
        api.upload_file(
            repo_id=entry["repo_id"],
            repo_type=entry["repo_type"],
            path_or_fileobj=str(card),
            path_in_repo="README.md",
        )
        for source in entry["sources"]:
            ignore_patterns = [
                ".git/**",
                ".cache/**",
                ".build/**",
                "__pycache__/**",
                "*.partial",
                ".gitattributes",
            ]
            # The curated card above is authoritative for root-folder uploads.
            # Do not let a source dataset's historical README replace it.
            if source["path_in_repo"] == ".":
                ignore_patterns.append("README.md")
            if source["upload_method"] != "folder":
                raise ValueError(f"Unsupported upload method: {source['upload_method']}")
            print(
                f"[publish] repository={entry['repo_id']}: upload {source['path']} "
                f"to {source['path_in_repo']}",
                flush=True,
            )
            source_root = Path(source["path"]).resolve()
            cache_writes_are_scoped = source_root.is_relative_to(REPO_ROOT)
            if source["path_in_repo"] == "." and cache_writes_are_scoped:
                # The large-folder uploader is resumable and emits a transfer
                # report periodically, which is important for 10 GiB shards.
                # It writes resume metadata into the source folder, so it is
                # restricted to release-owned staging/checkpoint directories.
                api.upload_large_folder(
                    repo_id=entry["repo_id"],
                    repo_type=entry["repo_type"],
                    folder_path=source["path"],
                    ignore_patterns=ignore_patterns,
                    print_report=True,
                    print_report_every=30,
                )
            else:
                # External source datasets are a strict read-only boundary.
                # upload_folder does not create resume metadata inside them.
                api.upload_folder(
                    repo_id=entry["repo_id"],
                    repo_type=entry["repo_type"],
                    folder_path=source["path"],
                    path_in_repo=None if source["path_in_repo"] == "." else source["path_in_repo"],
                    ignore_patterns=ignore_patterns,
                )
        print(f"[publish] repository={entry['repo_id']}: upload complete", flush=True)


if __name__ == "__main__":
    main()
