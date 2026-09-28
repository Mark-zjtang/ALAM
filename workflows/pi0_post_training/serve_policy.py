#!/usr/bin/env python3
"""Portable ALAM + pi0 policy server with release-correct checkpoints/horizons."""

from __future__ import annotations

import argparse
import dataclasses
import functools
import json
import logging
import os
import socket
import sys
from pathlib import Path

WORKFLOWS_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKFLOWS_ROOT))

from common.paths import REPO_ROOT, configure_runtime_environment, repo_path, require_dir
from common.prepare_lam_checkpoint import prepare


LIBERO_SUITES = {
    "spatial": {"cli_suite": "libero_spatial", "effective_horizon": 14, "raw_horizon": 15, "replan": 5},
    "object": {"cli_suite": "libero_object", "effective_horizon": 14, "raw_horizon": 15, "replan": 10},
    "goal": {"cli_suite": "libero_goal", "effective_horizon": 18, "raw_horizon": 19, "replan": 7},
    "long": {"cli_suite": "libero_10", "effective_horizon": 18, "raw_horizon": 19, "replan": 12},
}


def make_single_client_server_class(base_server_class):
    """Wrap a policy server so exactly one WebSocket session can consume RNG."""

    class SingleClientWebsocketPolicyServer(base_server_class):
        def __init__(self, *server_args, **server_kwargs):
            super().__init__(*server_args, **server_kwargs)
            self._client_session_claimed = False

        async def _handler(self, websocket):
            if self._client_session_claimed:
                logging.error(
                    "Rejecting extra policy client in single-client serial mode: remote=%s",
                    websocket.remote_address,
                )
                await websocket.close(
                    code=1008,
                    reason="This evaluation server accepts exactly one client session",
                )
                return
            self._client_session_claimed = True
            logging.info(
                "Accepted the only policy client in single-client serial mode: remote=%s",
                websocket.remote_address,
            )
            await super()._handler(websocket)

    return SingleClientWebsocketPolicyServer


def build_spec(args: argparse.Namespace) -> dict[str, object]:
    if args.benchmark == "metaworld":
        return {
            "benchmark": "metaworld",
            "config": "phy_metaworld_full_finetune",
            "policy_checkpoint": args.policy_checkpoint or os.environ.get(
                "ALAM_METAWORLD_POLICY", "evaluation/checkpoints/metaworld/alam_plus_pi_metaworld_mt50_step30000"
            ),
            "alam_checkpoint": args.alam_checkpoint or os.environ.get(
                "ALAM_METAWORLD_TOKENIZER", "evaluation/checkpoints/alam/metaworld_epoch19_step58216"
            ),
            "effective_horizon": 5,
            "raw_horizon": 6,
            "replan": 5,
        }
    suite = LIBERO_SUITES[args.suite]
    return {
        "benchmark": "libero",
        "suite": args.suite,
        "config": "phy_libero_full_finetune",
        "policy_checkpoint": args.policy_checkpoint or os.environ.get(
            "ALAM_LIBERO_POLICY", "evaluation/checkpoints/libero/alam_plus_pi_libero_step30000"
        ),
        "alam_checkpoint": args.alam_checkpoint or os.environ.get(
            "ALAM_LIBERO_TOKENIZER", "evaluation/checkpoints/alam/libero_epoch16_step49024"
        ),
        **suite,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("benchmark", choices=("metaworld", "libero"))
    parser.add_argument("--suite", choices=tuple(LIBERO_SUITES), default="spatial")
    parser.add_argument("--policy-checkpoint")
    parser.add_argument("--alam-checkpoint")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--default-prompt")
    parser.add_argument(
        "--single-client",
        action="store_true",
        help="Accept exactly one WebSocket client session for an isolated serial evaluation.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.chdir(REPO_ROOT)
    configure_runtime_environment()
    sys.path.insert(0, str(REPO_ROOT))
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.65")
    spec = build_spec(args)
    spec["client_mode"] = "single_client_serial" if args.single_client else "multi_client"
    spec["policy_checkpoint"] = str(require_dir(repo_path(str(spec["policy_checkpoint"])), "pi0 policy checkpoint"))
    source_lam = require_dir(repo_path(str(spec["alam_checkpoint"])), "ALAM tokenizer checkpoint")
    if args.dry_run:
        spec["alam_checkpoint"] = str(source_lam)
        print(json.dumps(spec, indent=2, ensure_ascii=False))
        return

    portable_lam = prepare(source_lam)
    spec["alam_checkpoint"] = str(portable_lam)
    sys.path.insert(0, str(REPO_ROOT / "evaluation" / "src"))

    from openpi.policies import policy_config
    from openpi.serving import websocket_policy_server
    from openpi.training import config as openpi_config

    train_config = openpi_config.get_config(str(spec["config"]))
    model_config = dataclasses.replace(
        train_config.model,
        action_horizon=int(spec["raw_horizon"]),
        phy_lam_ckpt=str(portable_lam),
    )
    train_config = dataclasses.replace(train_config, model=model_config)
    logging.info("Release policy specification: %s", json.dumps(spec, ensure_ascii=False))
    policy = policy_config.create_trained_policy(
        train_config,
        str(spec["policy_checkpoint"]),
        default_prompt=args.default_prompt,
    )
    hostname = socket.gethostname()
    logging.info("Serving on host=%s port=%d", hostname, args.port)
    # The preserved policy handler performs accelerator inference synchronously
    # on the asyncio thread. First-call JIT compilation can legitimately exceed
    # websockets' 20-second default ping timeout, so retain keepalive pings but
    # do not close a healthy connection while compilation is in progress.
    websocket_policy_server._server.serve = functools.partial(  # type: ignore[attr-defined]
        websocket_policy_server._server.serve,  # type: ignore[attr-defined]
        ping_timeout=None,
    )
    logging.info("WebSocket inference keepalive timeout disabled")
    server_class = websocket_policy_server.WebsocketPolicyServer
    if args.single_client:
        server_class = make_single_client_server_class(server_class)

    server = server_class(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy.metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
