#!/usr/bin/env python3
"""Run uv's dependency audit with one narrowly verified legacy-wheel waiver."""

from __future__ import annotations

import argparse
import subprocess
import sys


DECORD_PLATFORM_LINE = "The package `decord` was built for a different platform"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--uv", required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--allow-decord-wheel-tag", action="store_true")
    args = parser.parse_args()

    result = subprocess.run(
        [args.uv, "pip", "check", "--python", args.python],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    output = result.stdout.rstrip()
    if output:
        print(output, flush=True)
    if result.returncode == 0:
        return

    issue_lines = [line.strip() for line in output.splitlines() if line.startswith("The package `")]
    exact_known_issue = (
        args.allow_decord_wheel_tag
        and issue_lines == [DECORD_PLATFORM_LINE]
        and "Found 1 incompatibility" in output
    )
    if not exact_known_issue:
        raise SystemExit(result.returncode)

    # decord 0.6.0 is distributed under a py3 wheel filename but its embedded
    # WHEEL metadata incorrectly retains a cp36 tag. The package is ctypes-based;
    # accept only this exact uv diagnostic after its shared library imports.
    probe = subprocess.run(
        [
            args.python,
            "-c",
            "import decord; from decord import VideoReader; "
            "assert decord.__version__ == '0.6.0'; print('decord runtime import: PASS')",
        ],
        check=False,
    )
    if probe.returncode:
        raise SystemExit(probe.returncode)
    print("Dependency audit: PASS with verified decord 0.6.0 wheel-tag waiver")


if __name__ == "__main__":
    main()
