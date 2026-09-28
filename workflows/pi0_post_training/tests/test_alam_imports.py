#!/usr/bin/env python3
"""Verify ALAM can be imported through the frozen pi0/OpenPI environment."""

from __future__ import annotations

import importlib
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "evaluation" / "src"))


def main() -> None:
    canonical = importlib.import_module(
        "openpi.models.physics_lam_model.Algebraic_latent_action_model"
    )
    legacy = importlib.import_module("openpi.models.physics_lam_model.uni_world_model")
    if canonical is not legacy:
        raise AssertionError("Embedded canonical and historical ALAM names differ")
    importlib.import_module("openpi.models.portable_lam_tokenizer")
    importlib.import_module("openpi.models.pytorch_lam_wrapper")
    print("pi0 + ALAM import compatibility: PASS (hydra-core, lpips, and IPython available)")


if __name__ == "__main__":
    main()
