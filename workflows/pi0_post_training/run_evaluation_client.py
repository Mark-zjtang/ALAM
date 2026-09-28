#!/usr/bin/env python3
"""Run a preserved evaluator with an accelerator-safe WebSocket transport."""

from __future__ import annotations

import functools
import inspect
import os
from pathlib import Path
import runpy
import sys

import websockets.sync.client


def main() -> None:
    arguments = sys.argv[1:]
    hard_exit_on_success = False
    if arguments and arguments[0] == "--hard-exit-on-success":
        hard_exit_on_success = True
        arguments = arguments[1:]
    if not arguments:
        raise SystemExit(
            f"Usage: {Path(sys.argv[0]).name} [--hard-exit-on-success] "
            "EVALUATOR.py [EVALUATOR_ARGS ...]"
        )
    evaluator = Path(arguments[0]).resolve()
    if not evaluator.is_file():
        raise FileNotFoundError(f"Evaluator does not exist: {evaluator}")

    # The preserved client calls the module attribute at connection time, so a
    # process-local default is sufficient and doesn't edit the client package.
    # websockets 15 starts a keepalive thread in its sync client. A policy
    # server performing first-call JAX compilation cannot answer that ping on
    # time, so match the validated release run and disable the client watchdog.
    connect_parameters = inspect.signature(websockets.sync.client.connect).parameters
    if "ping_interval" in connect_parameters:
        websockets.sync.client.connect = functools.partial(
            websockets.sync.client.connect,
            ping_interval=None,
        )
        print("WebSocket evaluation-client keepalive ping disabled", flush=True)
    elif "ping_timeout" in connect_parameters:
        websockets.sync.client.connect = functools.partial(
            websockets.sync.client.connect,
            ping_timeout=None,
        )
        print("WebSocket evaluation-client keepalive timeout disabled", flush=True)
    else:
        # websockets 13.x sync clients expose neither option. Passing either via
        # **kwargs reaches socket.create_connection() and raises TypeError.
        print("WebSocket client has no sync keepalive option; using native behavior", flush=True)
    # Match `python path/to/evaluator.py`: sibling modules must resolve from
    # the evaluator's own directory, not from this adapter's directory.
    sys.path.insert(0, str(evaluator.parent))
    sys.argv = [str(evaluator), *arguments[1:]]
    try:
        runpy.run_path(str(evaluator), run_name="__main__")
    except SystemExit as error:
        if not hard_exit_on_success or error.code not in (None, 0):
            raise
    if hard_exit_on_success:
        # The preserved Python 3.8 LIBERO stack can abort in third-party native
        # destructors after a fully successful evaluator return.  Bypass only
        # that interpreter-teardown path; evaluator exceptions still propagate.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
