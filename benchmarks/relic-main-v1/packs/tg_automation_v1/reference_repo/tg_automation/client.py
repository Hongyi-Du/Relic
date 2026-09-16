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

    def __init__(
        self,
        config: AppConfig,
        client_factory: Optional[Callable[[AccountConfig], Any]] = None,
        clock: Callable[[], float] = time.monotonic,
    ):
        if not config.accounts:
            raise ValueError("ClientPool requires at least one configured account")
        self.config = config
        self._states: List[_AccountState] = [_AccountState(config=a) for a in config.accounts]
        self._clients: Dict[str, Any] = {}
        self._client_factory = client_factory or _default_factory
        self._clock = clock
        self._cursor = 0

    def _advance_window(self, state: _AccountState) -> None:
        now = self._clock()
        if now - state.window_started_at >= 60.0:
            state.window_started_at = now
            state.used_in_window = 0

    def acquire(self) -> tuple[AccountConfig, Any]:
        """Return ``(account, client)`` for the next available account.

        Raises ``RuntimeError`` if no account has remaining budget in the
        current minute.
        """
        n = len(self._states)
        for offset in range(n):
            idx = (self._cursor + offset) % n
            state = self._states[idx]
            self._advance_window(state)
            if state.used_in_window < state.config.rate_per_minute:
                state.used_in_window += 1
                self._cursor = (idx + 1) % n
                client = self._clients.get(state.config.name)
                if client is None:
                    client = self._client_factory(state.config)
                    self._clients[state.config.name] = client
                return state.config, client
        raise RuntimeError("all accounts exhausted their per-minute rate budget")

    def usage(self) -> Dict[str, int]:
        return {s.config.name: s.used_in_window for s in self._states}


def _default_factory(account: AccountConfig) -> Any:
    """Default Telethon factory; deferred import so tests can run without Telethon installed."""
    try:
        from telethon import TelegramClient  # type: ignore
    except Exception as exc:  # pragma: no cover - exercised only without telethon installed
        raise RuntimeError(
            "Telethon is not installed; install it or pass a custom client_factory"
        ) from exc
    proxy = None
    if account.proxy is not None:
        if account.proxy.kind == "socks5":
            proxy = ("socks5", account.proxy.host, account.proxy.port,
                     True, account.proxy.username, account.proxy.password)
        else:
            proxy = ("mtproto", account.proxy.host, account.proxy.port, account.proxy.secret)
    return TelegramClient(account.session, account.api_id, account.api_hash, proxy=proxy)


_POOL: Optional[ClientPool] = None


def get_client_pool(
    config: Optional[AppConfig] = None,
    client_factory: Optional[Callable[[AccountConfig], Any]] = None,
) -> ClientPool:
    """Return a process-wide ClientPool. Initializes on first call."""
    global _POOL
    if _POOL is None:
        if config is None:
            raise RuntimeError("client pool not initialized; pass a config on first call")
        _POOL = ClientPool(config, client_factory=client_factory)
    return _POOL


def reset_client_pool() -> None:
    """Test helper: drop the cached singleton."""
    global _POOL
    _POOL = None
