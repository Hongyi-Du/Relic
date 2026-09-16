"""Run just the human workspace.

The experiment stack this lives in is heavy — LLM providers, faiss, datasets,
pandas — but the HCI server touches none of it: its import graph is standard
library plus this repository, and serving it needs only FastAPI and uvicorn.
So this entry point stands alone, and ``requirements-hci.txt`` is four lines.

    python tools/run_hci.py                      # localhost, warmed up, clock running
    python tools/run_hci.py --host 0.0.0.0       # let other people take seats
    python tools/run_hci.py --llm                # LLM-backed working agent

No WSL, no dataset, no API key unless you ask for one.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

MISSING_DEPS = """
The workspace needs two packages that are not installed:

    {missing}

Install just what this server needs (not the whole experiment stack):

    pip install -r requirements-hci.txt

Or run it in Docker instead, which needs nothing installed locally:

    docker compose -f docker-compose.hci.yml up
"""


def _check_dependencies() -> None:
    import importlib.util

    missing = [name for name, module in (("fastapi", "fastapi"), ("uvicorn", "uvicorn"))
               if importlib.util.find_spec(module) is None]
    if missing:
        print(MISSING_DEPS.format(missing="  ".join(missing)), file=sys.stderr)
        raise SystemExit(2)


def _lan_address() -> str:
    import socket

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))       # no packet leaves; this just picks a route
            return s.getsockname()[0]
    except OSError:
        return ""


def _free_port(start: int) -> int:
    import socket

    for port in range(start, start + 200):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    return start


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="run_hci.py", description="Run the Relic human member workspace.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--host", default="127.0.0.1",
                   help="0.0.0.0 lets other machines take seats. The only credential "
                        "is the per-seat token, so use it on a trusted network.")
    p.add_argument("--port", type=int, default=8100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--warmup", type=int, default=36, metavar="HOURS",
                   help="Organizational hours to run before anyone joins, so the "
                        "company has a history to walk into. 0 starts on day one.")
    p.add_argument("--tick-seconds", type=float, default=15.0, metavar="SECONDS",
                   help="Real seconds per organizational hour once the clock runs.")
    p.add_argument("--paused", action="store_true",
                   help="Do not start the clock; the company waits until someone "
                        "presses resume.")
    p.add_argument("--llm", action="store_true",
                   help="Use the configured LLM for the members and the working "
                        "agents. Off by default: everything runs on rules and "
                        "templates, which needs no API key.")
    p.add_argument("--open", action="store_true", help="Open a browser when ready.")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    _check_dependencies()

    # Line-buffer stdout. Piped or captured (a wrapper script, docker logs, an
    # IDE terminal) Python block-buffers instead, and since uvicorn stays quiet
    # at this log level the startup banner would sit unseen in the buffer while
    # the server was already up -- indistinguishable from a hang.
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except (AttributeError, OSError):
        pass

    os.environ.setdefault("ORG_LLM", "1" if args.llm else "0")
    # Setup is lazy: the world and its clock are created only after the browser
    # selects a pack.  Preserve this CLI promise across that later API call
    # instead of merely parsing --paused and then losing it.
    os.environ["ORG_HCI_START_PAUSED"] = "1" if args.paused else "0"
    import uvicorn

    from environments.org_env.backend import main as server

    port = args.port if _port_is_free(args.port) else _free_port(args.port)
    if port != args.port:
        print(f"Port {args.port} is busy; using {port} instead.")
    shown = _lan_address() or args.host if args.host == "0.0.0.0" else args.host

    print()
    print(f"  Project setup      http://{shown}:{port}/org/setup")
    print(f"  P2 transparent     http://{shown}:{port}/org/seat")
    print(f"  P3 integrated      http://{shown}:{port}/org/liaison")
    print(f"  Debug inspector    http://{shown}:{port}/org/inspector")
    print()
    # The world is intentionally created only after the browser submits its
    # pack and timing choices.  Touching HUMAN.runtime() before that point
    # would dereference the lazy SESSION and make the standalone launcher exit.
    print("  Choose a project and configuration before claiming a seat.")
    if args.host == "0.0.0.0":
        print("  Reachable from the network: anyone who can reach this port "
              "can claim a seat.")
    print()

    if args.open:
        import threading
        import webbrowser
        threading.Timer(1.0, lambda: webbrowser.open(
            f"http://127.0.0.1:{port}/org/setup")).start()

    try:
        uvicorn.run(server.build_app(), host=args.host, port=port, log_level="warning")
    finally:
        server.HUMAN.shutdown()


def _port_is_free(port: int) -> bool:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) != 0


if __name__ == "__main__":
    main()
