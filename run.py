#!/usr/bin/env python3
"""
run.py — PINN Research Platform entry point
============================================
Usage:
    python run.py [--port PORT] [--no-browser]

Opens http://localhost:8765 automatically.
Press Ctrl+C to stop.
"""

import argparse, socket, sys, threading, time, webbrowser


def find_free_port(start: int = 8765) -> int:
    for p in range(start, start + 20):
        with socket.socket() as s:
            try:
                s.bind(("", p)); return p
            except OSError:
                continue
    return start


def open_browser(port: int, delay: float = 1.5):
    def _open():
        time.sleep(delay)
        webbrowser.open(f"http://localhost:{port}")
    threading.Thread(target=_open, daemon=True).start()


def check_deps():
    missing = []
    for pkg in ("torch", "fastapi", "uvicorn", "numpy"):
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"\n[PINN] Missing: {', '.join(missing)}")
        print(f"[PINN] Run:  pip install {' '.join(missing)}\n")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="PINN Research Platform")
    parser.add_argument("--port",       type=int, default=0)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    check_deps()

    import torch, uvicorn
    from src.server import app, DEVICE

    port = args.port if args.port else find_free_port(8765)

    print()
    print("=" * 56)
    print("  PINN Research Platform — Autonomous Self-Training")
    print(f"  Device  : {DEVICE}")
    print(f"  PyTorch : {torch.__version__}")
    print(f"  URL     : http://localhost:{port}")
    print("=" * 56)
    print()
    print("  Press Ctrl+C to stop.\n")

    if not args.no_browser:
        open_browser(port)

    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")


if __name__ == "__main__":
    main()
