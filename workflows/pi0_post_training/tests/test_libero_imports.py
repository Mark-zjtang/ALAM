#!/usr/bin/env python3
"""Import the pinned LIBERO stack and bypass its faulty native teardown."""

from __future__ import annotations

import os
import sys


def main() -> None:
    import openpi_client  # noqa: F401
    import robosuite  # noqa: F401
    from libero.libero import benchmark

    if "libero_spatial" not in benchmark.get_benchmark_dict():
        raise AssertionError("libero_spatial benchmark is unavailable")
    print("LIBERO evaluation environment import: PASS")
    sys.stdout.flush()
    sys.stderr.flush()
    # Imports succeeded. The pinned native stack can abort only while Python
    # destroys extension globals, so skip that teardown without masking import
    # or assertion exceptions above.
    os._exit(0)


if __name__ == "__main__":
    main()
