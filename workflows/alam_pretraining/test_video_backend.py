#!/usr/bin/env python3
"""Decode one real relative-path OXE video with the installed decord backend."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys


WORKFLOWS_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKFLOWS_ROOT))

from common.paths import configure_runtime_environment, repo_path, require_dir, require_file


def main() -> None:
    configure_runtime_environment()
    oxe_value = os.environ.get("ALAM_OXE_VIDEO_ROOT", "data/alam/oxe_videos")
    display = require_dir(
        repo_path(oxe_value) / "fractal20220817_data" / "image",
        "OXE fractal video directory",
    )
    metadata = json.loads(
        require_file(display / "video_metadata.json", "OXE video metadata").read_text(encoding="utf-8")
    )
    filename = metadata["train"]["videos"][0][0]
    video_path = require_file(display / filename, "OXE decode probe video")

    from decord import VideoReader, cpu

    reader = VideoReader(str(video_path), ctx=cpu(0))
    frame = reader[0].asnumpy()
    if len(reader) <= 0 or frame.ndim != 3 or frame.shape[-1] != 3:
        raise AssertionError(f"Unexpected decoded OXE video: frames={len(reader)} shape={frame.shape}")
    print(
        f"decord real-video decode PASS: path={video_path} "
        f"frames={len(reader)} first_frame={frame.shape} dtype={frame.dtype}"
    )


if __name__ == "__main__":
    main()
