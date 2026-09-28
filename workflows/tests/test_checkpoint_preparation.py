#!/usr/bin/env python3
"""Exercise concurrent runtime preparation of one released ALAM checkpoint."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import sys
import tempfile

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from workflows.common.prepare_lam_checkpoint import PORTABLE_TARGET, PRESERVED_TARGET, PRETRAIN_TARGET, prepare


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="alam-checkpoint-prepare-") as temporary:
        root = Path(temporary)
        source = root / "source" / "shared_checkpoint"
        source.mkdir(parents=True)
        (source / "config.yaml").write_text(f"_target_: {PRESERVED_TARGET}\n", encoding="utf-8")
        weights = source / "pytorch_model.bin"
        weights.write_bytes(b"checkpoint-fixture")
        os.environ["ALAM_RUNTIME_ROOT"] = str(root / "runtime")

        with ThreadPoolExecutor(max_workers=8) as executor:
            destinations = list(executor.map(lambda _: prepare(source), range(32)))

        destination = destinations[0]
        if any(path != destination for path in destinations):
            raise AssertionError("Concurrent checkpoint preparation returned different destinations")
        config = (destination / "config.yaml").read_text(encoding="utf-8")
        if config != f"_target_: {PORTABLE_TARGET}\n":
            raise AssertionError(f"Unexpected portable config: {config!r}")
        weight_link = destination / "pytorch_model.bin"
        if not weight_link.is_symlink() or weight_link.resolve() != weights.resolve():
            raise AssertionError(f"Incorrect runtime weight link: {weight_link}")
        temporary_files = list(destination.glob(".*.tmp"))
        if temporary_files:
            raise AssertionError(f"Checkpoint preparation left temporary files: {temporary_files}")

        # User-trained tokenizers can have the same basename as release weights.
        for index, target in enumerate((PRETRAIN_TARGET, PORTABLE_TARGET)):
            custom = root / f"experiment-{index}" / source.name
            custom.mkdir(parents=True)
            (custom / "config.yaml").write_text(f"_target_: {target}\n", encoding="utf-8")
            (custom / "pytorch_model.bin").write_bytes(b"custom-checkpoint-fixture")
            prepared = prepare(custom)
            if prepared == destination:
                raise AssertionError("Same-named checkpoints shared their runtime directory")
            if (prepared / "config.yaml").read_text() != f"_target_: {PORTABLE_TARGET}\n":
                raise AssertionError("User-trained tokenizer target was not adapted")
            if (prepared / "pytorch_model.bin").resolve() != custom / "pytorch_model.bin":
                raise AssertionError("Prepared tokenizer selected the wrong weights")
            if weight_link.resolve() != weights.resolve():
                raise AssertionError("Preparing a custom tokenizer changed the release weight link")

    print("ALAM checkpoint preparation: PASS (concurrent calls, custom targets, isolated weights)")


if __name__ == "__main__":
    main()
