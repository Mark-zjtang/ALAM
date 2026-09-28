#!/usr/bin/env python3
"""Exercise the dataset shard verifier and atomic restore on a tiny fixture."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from workflows.publishing.download_huggingface import ensure_download_directory, extract_archive_dataset


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="alam-archive-roundtrip-") as temporary:
        root = Path(temporary)
        staging = root / "staging"
        target = root / "restored"
        staging.mkdir()
        archive_path = staging / "data-00000.tar"
        members = {
            "training/npz_metadata.json": b'{"data": ["episode_0000000.npz"]}\n',
            "training/episode_0000000.npz": b"fixture-npz-payload",
        }
        with tarfile.open(archive_path, mode="w:") as archive:
            for name, payload in members.items():
                info = tarfile.TarInfo(name)
                info.size = len(payload)
                info.mode = 0o644
                archive.addfile(info, io.BytesIO(payload))

        digest = hashlib.sha256(archive_path.read_bytes()).hexdigest()
        (staging / "ARCHIVE_MANIFEST.sha256").write_text(
            f"{digest}  {archive_path.name}\n",
            encoding="utf-8",
        )
        (staging / "DATASET_LAYOUT.json").write_text(
            json.dumps(
                {
                    "repo_id": "fixture/alam-dataset",
                    "archive_shards": 1,
                    "source_members": len(members),
                }
            )
            + "\n",
            encoding="utf-8",
        )

        extract_archive_dataset(staging, target, "fixture/alam-dataset")
        for name, payload in members.items():
            if (target / name).read_bytes() != payload:
                raise AssertionError(f"Archive roundtrip mismatch: {name}")
        if not (target / ".alam_hf_dataset.json").is_file():
            raise AssertionError("Dataset restore marker was not created")

        # A second invocation must recognize the verified marker and not merge.
        extract_archive_dataset(staging, target, "fixture/alam-dataset")

        mismatched_target = root / "mismatched-restored"
        mismatched_extracting = root / ".mismatched-restored.extracting"
        mismatched_extracting.mkdir()
        (mismatched_extracting / ".extraction_state.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "repo_id": "fixture/alam-dataset",
                    "layout_sha256": "0" * 64,
                    "shards": {},
                }
            )
            + "\n",
            encoding="utf-8",
        )
        try:
            extract_archive_dataset(staging, mismatched_target, "fixture/alam-dataset")
        except RuntimeError as error:
            if "another repository revision" not in str(error):
                raise
        else:
            raise AssertionError("Mismatched extraction revision was not rejected")

        external = root / "external-read-only-root"
        external.mkdir()
        download_link = root / "download-link"
        download_link.symlink_to(external, target_is_directory=True)
        try:
            ensure_download_directory(download_link)
        except FileExistsError as error:
            if "symlink target" not in str(error):
                raise
        else:
            raise AssertionError("Downloader followed a symlink into an external root")

    print("Dataset archive download verification and atomic restore: PASS")


if __name__ == "__main__":
    main()
