#!/usr/bin/env python3
"""Verify the readiness probe uses the policy server's HTTP health endpoint."""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import sys
from threading import Thread


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from workflows.common.wait_for_port import health_ready


class Handler(BaseHTTPRequestHandler):
    requested_paths: list[str] = []

    def do_GET(self) -> None:  # noqa: N802 - standard-library callback name
        self.requested_paths.append(self.path)
        if self.path == "/healthz":
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"OK\n")
            return
        self.send_response(404)
        self.end_headers()

    def log_message(self, format: str, *args) -> None:
        return


def main() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        health_ready("127.0.0.1", server.server_port)
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    if Handler.requested_paths != ["/healthz"]:
        raise AssertionError(f"Unexpected readiness request: {Handler.requested_paths}")
    print("Policy-server HTTP readiness probe: PASS")


if __name__ == "__main__":
    main()
