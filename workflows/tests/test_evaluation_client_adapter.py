#!/usr/bin/env python3
"""Verify that the LIBERO success-only hard exit cannot mask failures."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import tempfile


REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    env_root = Path(os.environ.get("ALAM_ENV_ROOT", REPO_ROOT / ".venvs"))
    if not env_root.is_absolute():
        env_root = REPO_ROOT / env_root
    python = env_root / "libero" / "bin" / "python"
    adapter = REPO_ROOT / "workflows" / "pi0_post_training" / "run_evaluation_client.py"
    if not python.is_file():
        raise FileNotFoundError(f"Missing LIBERO Python: {python}")

    with tempfile.TemporaryDirectory(prefix="alam-evaluation-adapter-") as temporary:
        evaluator = Path(temporary) / "evaluator.py"
        evaluator.write_text(
            "import atexit\n"
            "import sys\n"
            "atexit.register(lambda: print('ATEXIT_RAN', flush=True))\n"
            "if '--fail' in sys.argv:\n"
            "    raise RuntimeError('fixture failure')\n"
            "print('EVALUATOR_RETURNED', flush=True)\n",
            encoding="utf-8",
        )
        base = [str(python), str(adapter), "--hard-exit-on-success", str(evaluator)]
        environment = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
        success = subprocess.run(base, cwd=REPO_ROOT, env=environment, text=True, capture_output=True)
        if success.returncode != 0:
            raise AssertionError(f"Success adapter returned {success.returncode}: {success.stderr}")
        if "EVALUATOR_RETURNED" not in success.stdout or "ATEXIT_RAN" in success.stdout:
            raise AssertionError(f"Success-only hard exit contract failed: {success.stdout!r}")

        failure = subprocess.run(
            [*base, "--fail"], cwd=REPO_ROOT, env=environment, text=True, capture_output=True
        )
        if failure.returncode == 0 or "RuntimeError: fixture failure" not in failure.stderr:
            raise AssertionError(
                f"Evaluator failure was masked: returncode={failure.returncode} stderr={failure.stderr!r}"
            )

    print("Evaluation-client success-only hard exit: PASS")


if __name__ == "__main__":
    main()
