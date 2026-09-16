"""Retry helpers and the Telegram FloodWaitError abstraction."""
from __future__ import annotations
import logging
import time
from typing import Any, Callable, Optional, TypeVar
LOG = logging.getLogger(__name__)
T = TypeVar('T')

class FloodWaitError(Exception):
    """Raised when Telegram demands the caller back off for ``seconds`` seconds."""

    def __init__(self, seconds: int, message: str='flood wait'):
        raise NotImplementedError('__init__ is not implemented yet')

class NetworkError(Exception):
    """Transient network failure that warrants retry."""

def retry_with_backoff(func: Callable[..., T], *args: Any, max_attempts: int=5, base_delay: float=1.0, sleep: Optional[Callable[[float], None]]=None, **kwargs: Any) -> T:
    """Call ``func`` and retry on FloodWaitError / NetworkError.

    FloodWaitError sleeps for the requested duration and does not count
    against ``max_attempts``-style decay (caller asked for an exact wait).
    NetworkError uses exponential backoff up to ``max_attempts`` total tries.
    """
    raise NotImplementedError('retry_with_backoff is not implemented yet')
