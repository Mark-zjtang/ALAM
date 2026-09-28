#!/usr/bin/env python3
"""Fail before launch when a requested local policy-server port is occupied."""

from __future__ import annotations

import argparse
import socket


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", required=True, type=int)
    args = parser.parse_args()

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            listener.bind((args.host, args.port))
        except OSError as error:
            raise SystemExit(f"Policy-server port is not available: {args.host}:{args.port}: {error}")


if __name__ == "__main__":
    main()
