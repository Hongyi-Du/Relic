"""Launch the pair worker in an isolated subprocess and load its receipt."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

from environments.org_env.cooperbench.contract import PairOutcome, PairRequest


def wait_for_preserved_outcome(
    path: Path, *, worker_exited: Callable[[], bool], error_message: str
) -> None:
    """Wait for a retained worker, allowing publication during the exit check."""
    while not path.exists():
        if worker_exited():
            # The worker can publish its atomic outcome and exit while the
            # collector takes its process snapshot. Re-read the durable result.
            if path.exists():
                return
            raise RuntimeError(error_message)
        time.sleep(5)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def execute_pair_via_worker(
    request: PairRequest, timeout_seconds: int
) -> PairOutcome:
    """Execute one pair without leaking process-global OrgEnv configuration."""

    from .resume_identity import resolve_request
    request = resolve_request(request)
    root = Path(request.log_dir).resolve() / "orgenv_b3_two_agent"
    root.mkdir(parents=True, exist_ok=True)
    request_path = root / "pair_request.json"
    result_path = root / "pair_outcome.json"
    worker_log_path = root / "worker.log"
    _atomic_json(request_path, request.to_dict())
    if result_path.exists():
        result_path.unlink()

    command = [
        sys.executable,
        "-m",
        "environments.org_env.cooperbench.worker",
        "--request",
        str(request_path),
        "--result",
        str(result_path),
    ]
    resumed = any((root / "checkpoints").glob("t*.pkl"))
    with worker_log_path.open("a" if resumed else "w", encoding="utf-8", newline="\n") as log:
        if resumed:
            log.write("\nCHECKPOINT_RESUME_WORKER_RELAUNCH\n")
            log.flush()
        try:
            completed = subprocess.run(
                command,
                cwd=str(Path(__file__).resolve().parents[3]),
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=int(timeout_seconds),
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            raise RuntimeError(
                f"orgenv_cooper_worker_timeout:{int(timeout_seconds)}"
            ) from error
    if completed.returncode != 0:
        raise RuntimeError(
            f"orgenv_cooper_worker_failed:exit={completed.returncode}:"
            f"log={worker_log_path}"
        )
    if not result_path.is_file():
        raise RuntimeError("orgenv_cooper_worker_result_missing")
    try:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError("orgenv_cooper_worker_result_invalid") from error
    return PairOutcome.from_dict(payload)


__all__ = ["execute_pair_via_worker"]
