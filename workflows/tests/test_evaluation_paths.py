#!/usr/bin/env python3
"""Check custom checkpoint routing without loading models or starting simulators."""

import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]


def commands(script, args, cwd):
    result = subprocess.run(
        ["bash", str(ROOT / script), *args, "--dry-run"],
        cwd=cwd, text=True, capture_output=True, check=True,
    )
    return [shlex.split(line[2:]) for line in result.stdout.splitlines() if line.startswith("+ ")]


def option(command, name):
    return command[command.index(name) + 1]


def main():
    with tempfile.TemporaryDirectory(prefix="alam-eval-paths-") as temporary:
        base = Path(temporary)
        policy = base / "my policy" / "30000"
        tokenizer = base / "my tokenizer"
        policy.mkdir(parents=True)
        tokenizer.mkdir()
        overrides = ["--policy-checkpoint", str(policy), "--alam-checkpoint", str(tokenizer)]
        output = "outputs/custom evaluation"
        entrypoints = [
            ("workflows/metaworld_evaluation/evaluate_mt50.sh", []),
            ("workflows/libero_evaluation/evaluate_libero.sh", ["goal"]),
            ("workflows/libero_evaluation/evaluate_all_suites.sh", []),
        ]
        for script, prefix in entrypoints:
            default = commands(script, prefix, base)
            custom = commands(script, [*prefix, *overrides, "--output-dir", output], base)
            assert len(custom) == len(default) == (8 if "all_suites" in script else 2)
            for index in range(0, len(custom), 2):
                server, client = custom[index:index + 2]
                assert option(server, "--policy-checkpoint") == str(policy)
                assert option(server, "--alam-checkpoint") == str(tokenizer)
                assert server[:-len(overrides)] == default[index]
                suite_suffix = ("/" + option(server, "--suite")) if "all_suites" in script else ""
                run_root = str(ROOT / output) + suite_suffix
                if "metaworld" in script:
                    assert option(client, "--output_dir") == run_root
                    output_flags = ("--output_dir",)
                else:
                    assert option(client, "--args.checkpoint_dir") == run_root + "/logs"
                    assert option(client, "--args.video_out_path") == run_root + "/videos"
                    output_flags = ("--args.checkpoint_dir", "--args.video_out_path")
                for flag in output_flags:
                    client[client.index(flag) + 1] = option(default[index + 1], flag)
                assert client == default[index + 1], "Custom paths changed evaluation settings"
            for invalid in (["--policy-checkpoint"], ["--alam-checkpoint", "--dry-run"], ["--output-dir", ""]):
                result = subprocess.run(["bash", str(ROOT / script), *prefix, *invalid], capture_output=True)
                assert result.returncode == 2
            if "all_suites" not in script:
                result = subprocess.run(
                    ["bash", str(ROOT / script), *prefix, *overrides, "--client-only", "--dry-run"],
                    capture_output=True,
                )
                assert result.returncode == 2

        # The server must choose CLI checkpoints even when env defaults disagree.
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        for benchmark in ("metaworld", "libero"):
            env[f"ALAM_{benchmark.upper()}_POLICY"] = str(base / "wrong-policy")
            env[f"ALAM_{benchmark.upper()}_TOKENIZER"] = str(base / "wrong-tokenizer")
            result = subprocess.run(
                [sys.executable, "-B", str(ROOT / "workflows/pi0_post_training/serve_policy.py"),
                 benchmark, *overrides, "--dry-run"],
                cwd=base, env=env, text=True, capture_output=True, check=True,
            )
            spec = json.loads(result.stdout)
            assert spec["policy_checkpoint"] == str(policy)
            assert spec["alam_checkpoint"] == str(tokenizer)
        assert not (ROOT / output).exists(), "Dry run created an output directory"
    print("Custom evaluation paths: PASS (MetaWorld, LIBERO, all suites, CLI precedence)")


if __name__ == "__main__":
    main()
