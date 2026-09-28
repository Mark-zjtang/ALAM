#!/usr/bin/env python3
"""Validate the CALVIN/OXE layouts consumed by ALAM pretraining."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


WORKFLOWS_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKFLOWS_ROOT))

from common.paths import configure_runtime_environment, repo_path


EXPECTED_OXE = {
    "fractal20220817_data/image": {"train": (87_212, 3_786_400)},
    "bridge/image": {"train": (25_460, 864_292), "test": (3_475, 118_603)},
    "taco_play/rgb_static": {"train": (3_242, 213_972), "test": (361, 23_826)},
    "jaco_play/image": {"train": (976, 70_127), "test": (109, 7_838)},
    "berkeley_cable_routing/image": {"train": (1_482, 38_240), "test": (165, 4_088)},
    "roboturk/front_rgb": {"train": (1_790, 168_417), "test": (199, 19_084)},
    "nyu_door_opening_surprising_effectiveness/image": {
        "train": (435, 18_196),
        "test": (49, 2_209),
    },
    "viola/agentview_rgb": {"train": (135, 68_913), "test": (15, 7_411)},
    "berkeley_autolab_ur5/image": {"train": (896, 87_783), "test": (104, 10_156)},
    "toto/image": {"train": (902, 294_139), "test": (101, 31_560)},
}


def selected_records(records: list, full_files: bool) -> list:
    if full_files or len(records) <= 6:
        return records
    return records[:3] + records[-3:]


def check_calvin(root: Path, full_files: bool) -> None:
    expected = {"training": 1_795_045, "validation": 99_022}
    for split, expected_count in expected.items():
        split_root = root / split
        metadata_path = split_root / "npz_metadata.json"
        names = json.loads(metadata_path.read_text(encoding="utf-8"))
        if len(names) != expected_count:
            raise AssertionError(f"CALVIN {split}: expected {expected_count}, found {len(names)}")
        for filename in selected_records(names, full_files):
            if not (split_root / filename).is_file():
                raise FileNotFoundError(split_root / filename)
    print("CALVIN dataset PASS: training=1,795,045 validation=99,022")


def check_oxe(root: Path, full_files: bool) -> None:
    for relative, expected_splits in EXPECTED_OXE.items():
        display_root = root / relative
        metadata = json.loads((display_root / "video_metadata.json").read_text(encoding="utf-8"))
        for split, expected in expected_splits.items():
            records = metadata[split]["videos"]
            found = (len(records), metadata[split]["total_frames"])
            if found != expected:
                raise AssertionError(f"OXE {relative}/{split}: expected {expected}, found {found}")
            for filename, _ in selected_records(records, full_files):
                if not (display_root / filename).is_file():
                    raise FileNotFoundError(display_root / filename)
    print("OXE dataset PASS: 10 active dataset/display directories")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calvin-root")
    parser.add_argument("--oxe-root")
    parser.add_argument("--full-files", action="store_true", help="Stat every metadata-referenced file")
    args = parser.parse_args()
    configure_runtime_environment()
    calvin = repo_path(args.calvin_root or os.environ.get("ALAM_CALVIN_ROOT", "data/alam/calvin"))
    oxe = repo_path(args.oxe_root or os.environ.get("ALAM_OXE_VIDEO_ROOT", "data/alam/oxe_videos"))
    check_calvin(calvin, args.full_files)
    check_oxe(oxe, args.full_files)


if __name__ == "__main__":
    main()
