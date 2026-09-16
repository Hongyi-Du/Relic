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
        raise NotImplementedError('account_names is not implemented yet')

def _proxy_from_dict(data: Optional[Dict[str, Any]]) -> Optional[ProxyConfig]:
    raise NotImplementedError('_proxy_from_dict is not implemented yet')

def load_config(path: str | os.PathLike[str]) -> AppConfig:
    """Load an AppConfig from a JSON file."""
    raise NotImplementedError('load_config is not implemented yet')
