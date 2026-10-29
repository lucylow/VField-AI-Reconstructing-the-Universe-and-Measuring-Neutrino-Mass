#!/usr/bin/env python3
"""Serve the νField static frontend locally using Python's standard library."""
from __future__ import annotations

import argparse
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from functools import partial


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1)")
    parser.add_argument("--port", default=8000, type=int, help="HTTP port (default: 8000)")
    args = parser.parse_args()

    # Serve the repository root so /frontend/ and examples/sample_output/ use the same root.
    repo_root = Path(__file__).resolve().parents[1]
    handler = partial(SimpleHTTPRequestHandler, directory=str(repo_root))
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"νField frontend available at http://{args.host}:{args.port}/frontend/")
    print("Press Ctrl+C to stop. This server serves static files only.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping local frontend server.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
