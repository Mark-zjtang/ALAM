#!/usr/bin/env python3
"""Verify that installed wheel metadata does not point to missing files."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import subprocess


def site_packages(python: str) -> Path:
    result = subprocess.run(
        [python, "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    return Path(result.stdout.strip()).resolve()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", required=True)
    args = parser.parse_args()

    root = site_packages(args.python)
    records = sorted(root.glob("*.dist-info/RECORD"))
    if not records:
        raise FileNotFoundError(f"No wheel RECORD files found under {root}")

    incomplete: list[tuple[str, list[str]]] = []
    for record in records:
        missing: list[str] = []
        with record.open(newline="", encoding="utf-8", errors="replace") as handle:
            for row in csv.reader(handle):
                if not row or not row[0]:
                    continue
                installed_path = (root / row[0]).resolve()
                if not installed_path.exists():
                    missing.append(row[0])
        if missing:
            incomplete.append((record.parent.name, missing))

    if incomplete:
        lines = [f"Incomplete wheel installations under {root}:"]
        for distribution, paths in incomplete:
            lines.append(f"- {distribution}: {len(paths)} missing RECORD path(s)")
            lines.extend(f"    {path}" for path in paths[:20])
            if len(paths) > 20:
                lines.append(f"    ... {len(paths) - 20} more")
        raise AssertionError("\n".join(lines))

    print(f"Installed wheel file audit: PASS ({len(records)} RECORD files under {root})")


if __name__ == "__main__":
    main()
