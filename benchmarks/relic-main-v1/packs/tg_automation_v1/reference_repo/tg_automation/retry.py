"""Retry helpers and the Telegram FloodWaitError abstraction."""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Optional, TypeVar

LOG = logging.getLogger(__name__)

T = TypeVar("T")


class FloodWaitError(Exception):
    """Raised when Telegram demands the caller back off for ``seconds`` seconds."""

    def __init__(self, seconds: int, message: str = "flood wait"):
        super().__init__(f"{message}: wait {seconds}s")
        self.seconds = int(seconds)


class NetworkError(Exception):
    """Transient network failure that warrants retry."""


def retry_with_backoff(
    func: Callable[..., T],
    *args: Any,
    max_attempts: int = 5,
    base_delay: float = 1.0,
    sleep: Optional[Callable[[float], None]] = None,
    **kwargs: Any,
) -> T:
    """Call ``func`` and retry on FloodWaitError / NetworkError.

    FloodWaitError sleeps for the requested duration and does not count
    against ``max_attempts``-style decay (caller asked for an exact wait).
    NetworkError uses exponential backoff up to ``max_attempts`` total tries.
    """
    _sleep = sleep if sleep is not None else time.sleep
    attempt = 0
    while True:
        try:
            return func(*args, **kwargs)
        except FloodWaitError as flood:
            LOG.warning("flood wait %ss, sleeping", flood.seconds)
            _sleep(float(flood.seconds))
            continue
        except NetworkError as net:
            attempt += 1
            if attempt >= max_attempts:
                LOG.error("network error after %d attempts: %s", attempt, net)
                raise
            delay = base_delay * (2 ** (attempt - 1))
            LOG.warning("network error (attempt %d): %s; backoff %.1fs", attempt, net, delay)
            _sleep(delay)
