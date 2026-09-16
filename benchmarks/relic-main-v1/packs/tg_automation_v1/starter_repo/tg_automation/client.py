"""Client pool: rotates across accounts and tracks per-account rate budgets."""
from __future__ import annotations
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
from tg_automation.config import AccountConfig, AppConfig
LOG = logging.getLogger(__name__)

@dataclass
class _AccountState:
    config: AccountConfig
    used_in_window: int = 0
    window_started_at: float = 0.0

class ClientPool:
    """Round-robin pool over configured accounts.

    Each account has a per-minute rate budget. When the budget is exhausted
    for the current account, the pool advances to the next account. The
    underlying Telegram client is produced by ``client_factory`` and cached
    per-account so tests can inject fakes.
    """

    def __init__(self, config: AppConfig, client_factory: Optional[Callable[[AccountConfig], Any]]=None, clock: Callable[[], float]=time.monotonic):
        raise NotImplementedError('__init__ is not implemented yet')

    def _advance_window(self, state: _AccountState) -> None:
        raise NotImplementedError('_advance_window is not implemented yet')

    def acquire(self) -> tuple[AccountConfig, Any]:
        """Return ``(account, client)`` for the next available account.

        Raises ``RuntimeError`` if no account has remaining budget in the
        current minute.
        """
        raise NotImplementedError('acquire is not implemented yet')

    def usage(self) -> Dict[str, int]:
        raise NotImplementedError('usage is not implemented yet')

def _default_factory(account: AccountConfig) -> Any:
    """Default Telethon factory; deferred import so tests can run without Telethon installed."""
    raise NotImplementedError('_default_factory is not implemented yet')
_POOL: Optional[ClientPool] = None

def get_client_pool(config: Optional[AppConfig]=None, client_factory: Optional[Callable[[AccountConfig], Any]]=None) -> ClientPool:
    """Return a process-wide ClientPool. Initializes on first call."""
    raise NotImplementedError('get_client_pool is not implemented yet')

def reset_client_pool() -> None:
    """Test helper: drop the cached singleton."""
    raise NotImplementedError('reset_client_pool is not implemented yet')
