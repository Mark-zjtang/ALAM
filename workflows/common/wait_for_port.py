#!/usr/bin/env python3
"""Wait until a downstream policy server passes its HTTP health check."""

from __future__ import annotations

import argparse
import http.client
import os
from pathlib import Path
import time


def process_alive(pid: int) -> bool:
    """Return false for a missing or zombie local server process."""
    stat_path = Path(f"/proc/{pid}/stat")
    if stat_path.exists():
        try:
            return stat_path.read_text(encoding="utf-8").split()[2] != "Z"
        except (FileNotFoundError, IndexError, PermissionError):
            pass
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def health_ready(host: str, port: int, timeout: float = 2.0) -> None:
    """Require the policy server's native /healthz response."""
    connection = http.client.HTTPConnection(host, port, timeout=timeout)
    try:
        connection.request("GET", "/healthz", headers={"Connection": "close"})
        response = connection.getresponse()
        body = response.read()
    finally:
        connection.close()
    if response.status != 200 or body != b"OK\n":
        raise OSError(
            f"health check returned status={response.status} body={body!r}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--pid", type=int, help="fail immediately if this server process exits")
    args = parser.parse_args()

    deadline = time.monotonic() + args.timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        if args.pid is not None and not process_alive(args.pid):
            raise SystemExit(
                f"Policy server process {args.pid} exited before opening {args.host}:{args.port}"
            )
        try:
            health_ready(args.host, args.port)
            print(f"policy server ready: {args.host}:{args.port}")
            return
        except (OSError, http.client.HTTPException) as error:
            last_error = error
            time.sleep(1.0)
    raise SystemExit(f"Timed out waiting for {args.host}:{args.port}: {last_error}")


if __name__ == "__main__":
    main()
