"""Configuration loader for accounts and proxies."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class ProxyConfig:
    kind: str
    host: str
    port: int
    username: Optional[str] = None
    password: Optional[str] = None
    secret: Optional[str] = None


@dataclass
class AccountConfig:
    name: str
    api_id: int
    api_hash: str
    session: str
    rate_per_minute: int = 30
    proxy: Optional[ProxyConfig] = None


@dataclass
class AppConfig:
    accounts: List[AccountConfig] = field(default_factory=list)
    default_window_days: int = 30

    def account_names(self) -> List[str]:
        return [a.name for a in self.accounts]


def _proxy_from_dict(data: Optional[Dict[str, Any]]) -> Optional[ProxyConfig]:
    if not data:
        return None
    kind = str(data.get("kind", "socks5")).lower()
    if kind not in ("socks5", "mtproto"):
        raise ValueError(f"unsupported proxy kind: {kind}")
    return ProxyConfig(
        kind=kind,
        host=str(data["host"]),
        port=int(data["port"]),
        username=data.get("username"),
        password=data.get("password"),
        secret=data.get("secret"),
    )


def load_config(path: str | os.PathLike[str]) -> AppConfig:
    """Load an AppConfig from a JSON file."""
    p = Path(path)
    raw = json.loads(p.read_text(encoding="utf-8"))
    accounts_raw = raw.get("accounts", [])
    if not isinstance(accounts_raw, list) or not accounts_raw:
        raise ValueError("config must declare a non-empty 'accounts' list")
    accounts: List[AccountConfig] = []
    seen_names: set[str] = set()
    for a in accounts_raw:
        name = str(a["name"])
        if name in seen_names:
            raise ValueError(f"duplicate account name: {name}")
        seen_names.add(name)
        accounts.append(
            AccountConfig(
                name=name,
                api_id=int(a["api_id"]),
                api_hash=str(a["api_hash"]),
                session=str(a["session"]),
                rate_per_minute=int(a.get("rate_per_minute", 30)),
                proxy=_proxy_from_dict(a.get("proxy")),
            )
        )
    return AppConfig(
        accounts=accounts,
        default_window_days=int(raw.get("default_window_days", 30)),
    )
