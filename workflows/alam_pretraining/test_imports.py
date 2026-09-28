#!/usr/bin/env python3
"""Verify the canonical ALAM name and preserved training import are compatible."""

from __future__ import annotations

import importlib
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


def main() -> None:
    canonical = importlib.import_module("Algebraic_latent_action_model")
    legacy = importlib.import_module("uni_world_model")
    if canonical is not legacy:
        raise AssertionError("Canonical and historical ALAM package names are not the same module")
    train_module = importlib.import_module("train_lam")
    if not callable(getattr(train_module, "main", None)):
        raise AssertionError("Preserved train_lam.main is unavailable")
    print("ALAM import compatibility: PASS (Algebraic_latent_action_model == uni_world_model)")


if __name__ == "__main__":
    main()
