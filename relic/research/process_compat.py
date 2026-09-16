"""Process-group and advisory-lock primitives that work off POSIX too.

The organization runners need two things from the operating system: refuse to
start when another runner already owns the resumable execution state, and tear
a hung case down together with everything it spawned. Both are POSIX idioms
(``flock`` and ``killpg``), and writing them inline locked the runners to POSIX
at import time - the module-level ``import fcntl`` failed before argparse ever
ran. Keeping both implementations here lets the callers carry one code path.

One difference is real and is not smoothed over: the POSIX tree kill is a
kernel-enforced signal to a process group, while the Windows one walks the
parent/child table, so a grandchild that has already been reparented can
survive it. Long unattended sweeps still belong on Linux for that reason.
"""

from __future__ import annotations

import os
import signal
import subprocess
from typing import IO, Any

_IS_WINDOWS = os.name == "nt"


def _descriptor(target: IO[Any] | int) -> int:
    return target if isinstance(target, int) else target.fileno()


def acquire_exclusive_lock(target: IO[Any] | int) -> bool:
    """Take an exclusive advisory lock without waiting.

    Accepts an open file object or a raw descriptor. Returns False when another
    process already holds the lock, rather than raising, so the caller decides
    whether that is fatal. Closing releases it on both platforms.
    """

    descriptor = _descriptor(target)
    if _IS_WINDOWS:
        import msvcrt

        try:
            # A one-byte range at offset zero is enough to make the file
            # exclusive between cooperating runners.
            os.lseek(descriptor, 0, os.SEEK_SET)
            msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
        except OSError:
            return False
        return True

    import fcntl

    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (BlockingIOError, OSError):
        return False
    return True


def release_exclusive_lock(target: IO[Any] | int) -> None:
    """Drop a lock taken by ``acquire_exclusive_lock``, ignoring a stale one."""

    descriptor = _descriptor(target)
    try:
        if _IS_WINDOWS:
            import msvcrt

            os.lseek(descriptor, 0, os.SEEK_SET)
            msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
            return
        import fcntl

        fcntl.flock(descriptor, fcntl.LOCK_UN)
    except OSError:
        pass


def interruption_signals() -> tuple[signal.Signals, ...]:
    """The signals a long run should treat as an external stop request.

    ``SIGHUP`` does not exist on Windows, so asking for it by name there is an
    AttributeError rather than a no-op.
    """

    names = ("SIGTERM", "SIGHUP") if not _IS_WINDOWS else ("SIGTERM",)
    return tuple(getattr(signal, name) for name in names if hasattr(signal, name))


def new_process_group_kwargs() -> dict[str, Any]:
    """Popen kwargs that put the child in its own killable process group.

    ``start_new_session`` is accepted but silently ignored on Windows, which
    would leave nothing to kill; the equivalent there is an explicit creation
    flag.
    """

    if _IS_WINDOWS:
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


def terminate_process_tree(
    process: subprocess.Popen,
    *,
    grace_seconds: float = 15.0,
) -> None:
    """Ask the child's whole group to stop, then force it, then reap it."""

    if process.poll() is not None:
        return
    if _IS_WINDOWS:
        _terminate_windows_tree(process, grace_seconds=grace_seconds)
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=grace_seconds)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def _terminate_windows_tree(
    process: subprocess.Popen,
    *,
    grace_seconds: float,
) -> None:
    # taskkill walks the live parent/child table, so it has to run while that
    # table is still intact. Asking the direct child to exit first - the
    # CTRL_BREAK_EVENT analogue of SIGTERM - races it: the child can be gone
    # before the walk starts, leaving its own children reparented and immortal.
    # A tree teardown is not the place to be polite about it.
    killed = False
    try:
        killed = (
            subprocess.run(
                ["taskkill", "/T", "/F", "/PID", str(process.pid)],
                capture_output=True,
                check=False,
                timeout=grace_seconds,
            ).returncode
            == 0
        )
    except (OSError, subprocess.TimeoutExpired):
        killed = False
    if not killed:
        process.kill()
    try:
        process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


__all__ = [
    "acquire_exclusive_lock",
    "interruption_signals",
    "new_process_group_kwargs",
    "release_exclusive_lock",
    "terminate_process_tree",
]
