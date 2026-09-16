"""Windows desktop launcher for the P2 and P3 human interfaces.

The packaged executable owns both the local FastAPI process and one isolated
Edge app window. Closing that window stops the in-memory world and the server.
"""
from __future__ import annotations

import ctypes
import logging
import multiprocessing
import os
import runpy
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path


APP_NAME = "Relic Secretary"
APP_VERSION = "0.1.13"
PACK_ID = "traffic_watch_v1"
START_PORT = 8189


def bundle_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS"))
    return Path(__file__).resolve().parents[2]


def user_root() -> Path:
    base = Path(os.environ.get("LOCALAPPDATA") or Path.home())
    path = base / APP_NAME
    path.mkdir(parents=True, exist_ok=True)
    return path


def configure_runtime() -> None:
    root = bundle_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    os.chdir(root)
    os.environ["ORG_LLM"] = "1"
    os.environ["ORG_HCI_START_PAUSED"] = "0"
    os.environ["ORG_LLM_LOCAL_CONFIG"] = str(root / "config" / "llm.local.yaml")


def free_port(start: int = START_PORT) -> int:
    for port in range(start, start + 100):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            if probe.connect_ex(("127.0.0.1", port)) != 0:
                return port
    raise RuntimeError("No local port is available for the Secretary service.")


def wait_until_listening(port: int, timeout: float = 20.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            if probe.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(0.1)
    raise RuntimeError("The Secretary service did not start within 20 seconds.")


def edge_executable() -> Path:
    candidates = [
        Path(os.environ.get("ProgramFiles(x86)", "")) / "Microsoft/Edge/Application/msedge.exe",
        Path(os.environ.get("ProgramFiles", "")) / "Microsoft/Edge/Application/msedge.exe",
        Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft/Edge/Application/msedge.exe",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise RuntimeError("Microsoft Edge is required to open the Secretary desktop window.")


def show_error(message: str) -> None:
    ctypes.windll.user32.MessageBoxW(0, message, APP_NAME, 0x10)


def run_embedded_python() -> int | None:
    """Act as the packaged world's Python command for its declared checks.

    Product smoke checks are intentionally launched through ``sys.executable``.
    In a frozen application that path is Secretary.exe, so Python-style child
    invocations must be dispatched before the desktop launcher is initialized.
    """
    arguments = sys.argv[1:]
    if not arguments or arguments[0] not in {"-m", "-c"} and not arguments[0].endswith(".py"):
        return None

    working_directory = os.getcwd()
    if working_directory not in sys.path:
        sys.path.insert(0, working_directory)
    try:
        if arguments[0] == "-m":
            if len(arguments) < 2:
                raise ValueError("-m requires a module name")
            module = arguments[1]
            sys.argv = [module, *arguments[2:]]
            runpy.run_module(module, run_name="__main__", alter_sys=True)
        elif arguments[0] == "-c":
            if len(arguments) < 2:
                raise ValueError("-c requires program text")
            # Product checks arrive as tokenized command lines.  Windows can
            # therefore pass ``-c import package`` as three arguments even
            # though ``import package`` is one program.  This embedded runner
            # treats every token after -c as program text; it does not expose
            # CPython's optional ``-c ... arg`` convention.
            program = " ".join(arguments[1:])
            sys.argv = ["-c"]
            namespace = {"__name__": "__main__", "__package__": None}
            exec(compile(program, "<string>", "exec"), namespace, namespace)
        else:
            script = str(Path(arguments[0]).resolve())
            sys.argv = [script, *arguments[1:]]
            runpy.run_path(script, run_name="__main__")
    except SystemExit as exit_signal:
        return int(exit_signal.code or 0)
    return 0


def start_server(port: int, *, start_runtime: bool = True):
    import uvicorn
    from environments.org_env.backend import main as server

    initialized = server.api_init_world(
        PACK_ID,
        warmup=0,
        clock_speed=30.0,
        user_id="desktop",
        start_runtime=start_runtime,
    )
    if initialized.get("error"):
        raise RuntimeError(str(initialized["error"]))

    config = uvicorn.Config(
        server.build_app(),
        host="127.0.0.1",
        port=port,
        log_level="warning",
        log_config=None,
        loop="asyncio",
        http="h11",
    )
    instance = uvicorn.Server(config)
    thread = threading.Thread(target=instance.run, name="secretary-server", daemon=True)
    thread.start()
    wait_until_listening(port)
    return server, instance, thread


def self_test(url: str) -> None:
    with urllib.request.urlopen(url, timeout=10) as response:
        if response.status != 200 or b'<div id="root"></div>' not in response.read():
            raise RuntimeError("Packaged HCI page did not pass its startup check.")


def interface_mode(arguments: list[str] | None = None) -> str:
    """Return the requested study interface; P3 remains the default."""
    arguments = sys.argv[1:] if arguments is None else arguments
    return "p2" if "--p2" in arguments else "p3"


def interface_path(mode: str) -> str:
    return "/org/seat" if mode == "p2" else "/org/liaison"


def run() -> int:
    configure_runtime()
    mode = interface_mode()
    log_path = user_root() / "Secretary.log"
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        encoding="utf-8",
        force=True,
    )
    logging.info("Starting %s %s in %s mode", APP_NAME, APP_VERSION, mode.upper())
    port = free_port()
    server = instance = thread = None
    try:
        server, instance, thread = start_server(
            port,
            start_runtime="--self-test" not in sys.argv,
        )
        url = f"http://127.0.0.1:{port}{interface_path(mode)}"
        self_test(url)
        if "--self-test" in sys.argv:
            return 0

        # Edge keeps background processes for a user-data directory and may
        # hand a later launch to one of them.  Waiting on that short-lived
        # hand-off process used to shut the Secretary server down while its
        # window was still opening.  A versioned app profile gives this build
        # one browser owner; disabling background mode makes closing the app
        # window end that owner and, consequently, the local server.
        profile = user_root() / f"BrowserProfile-{APP_VERSION}-{mode}"
        profile.mkdir(parents=True, exist_ok=True)
        browser = subprocess.Popen([
            str(edge_executable()),
            f"--app={url}",
            f"--user-data-dir={profile}",
            "--no-first-run",
            "--no-default-browser-check",
            "--disable-sync",
            "--disable-background-mode",
        ])
        logging.info("Listening on http://127.0.0.1:%s; Edge pid=%s", port, browser.pid)
        browser.wait()
        logging.info("Desktop window closed; stopping local server")
        return 0
    except Exception as exc:
        logging.exception("Secretary failed to start")
        if "--self-test" in sys.argv:
            print(f"Secretary self-test failed: {exc}", file=sys.stderr, flush=True)
            return 1
        show_error(f"Secretary could not start.\n\n{exc}\n\nLog: {log_path}")
        return 1
    finally:
        if instance is not None:
            instance.should_exit = True
        if thread is not None:
            thread.join(timeout=10)
        if server is not None:
            server.HUMAN.shutdown()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    python_exit = run_embedded_python()
    raise SystemExit(run() if python_exit is None else python_exit)
