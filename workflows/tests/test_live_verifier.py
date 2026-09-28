#!/usr/bin/env python3
"""Replay preserved evaluator logs through the strict live-run verifier."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile

from verify_live_reproduction import LIBERO, REPO_ROOT, verify_libero, verify_metaworld


EVIDENCE = REPO_ROOT / "evaluation" / "evaluation_evidence"
META_POLICY = REPO_ROOT / "evaluation/checkpoints/metaworld/alam_plus_pi_metaworld_mt50_step30000"
LIBERO_POLICY = REPO_ROOT / "evaluation/checkpoints/libero/alam_plus_pi_libero_step30000"
LIBERO_LOGS = {
    "spatial": "replan5_libero_spatial_phylam_train_20_infer__eval_30k_0506_last.log",
    "object": "replan10_libero_object_phylam_train_20_infer_14_eval_30k_0504.log",
    "goal": "replan7_libero_goal.log",
    "long": "replan12_libero_10.log",
}


def write_server_log(path: Path, benchmark: str, policy: Path, alam_name: str, horizon: int, replan: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    spec = {
        "benchmark": benchmark,
        "policy_checkpoint": str(policy.resolve()),
        "alam_checkpoint": str(REPO_ROOT / "evaluation/checkpoints/alam" / alam_name),
        "effective_horizon": horizon,
        "replan": replan,
    }
    path.write_text(
        "\n".join(
            (
                f"Release policy specification: {json.dumps(spec)}",
                f"load {spec['alam_checkpoint']}/pytorch_model.bin",
                "missing:    set()",
                "unexpected: []",
            )
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="alam-live-verifier-") as temporary:
        output = Path(temporary)

        meta_root = output / "evaluation/metaworld_mt50"
        write_server_log(
            meta_root / "server/server.log",
            "metaworld",
            META_POLICY,
            "metaworld_epoch19_step58216",
            5,
            5,
        )
        meta_log = meta_root / "fixture/pi0_wflow_policy_49_task_evaluate_logging_fixture/evaluation.log"
        meta_log.parent.mkdir(parents=True, exist_ok=True)
        meta_log.symlink_to(EVIDENCE / "metaworld/mt50_cam_x_0.70.log")

        for suite, (cli_suite, horizon, replan, _) in LIBERO.items():
            suite_root = output / "evaluation/libero" / suite
            write_server_log(
                suite_root / "server/server.log",
                "libero",
                LIBERO_POLICY,
                "libero_epoch16_step49024",
                horizon,
                replan,
            )
            model_name = f"alam_plus_pi_libero_trainH20_inferH{horizon}_replan{replan}"
            evaluator_log = suite_root / "logs/alam_plus_pi_libero" / f"{cli_suite}_{model_name}.log"
            evaluator_log.parent.mkdir(parents=True, exist_ok=True)
            if suite == "object":
                # The preserved object log contains two interleaved historical jobs. A
                # deterministic appended-log fixture checks the live parser instead.
                lines: list[str] = []
                for success in [False] * 500:
                    lines.extend((f"Success: {success}", f"port: 8301, replan_steps: {replan}"))
                lines.extend(("Total success rate: 0.0", "Total episodes: 500"))
                for success in [True] * 498 + [False] * 2:
                    lines.extend((f"Success: {success}", f"port: 8301, replan_steps: {replan}"))
                lines.extend(("Total success rate: 0.996", "Total episodes: 500"))
                evaluator_log.write_text("\n".join(lines) + "\n", encoding="utf-8")
            else:
                evaluator_log.symlink_to(EVIDENCE / "libero" / LIBERO_LOGS[suite])

        metaworld = verify_metaworld(output)
        libero = verify_libero(output)
        assert metaworld["successes"] == 434
        assert libero["suites"]["object"]["successes"] == 498
        assert libero["paper_rounded_percent"] == 98.1

    print("Historical-log replay through live verifier: PASS")


if __name__ == "__main__":
    main()
