"""Tests for tg_automation.config and tg_automation.client."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tg_automation.client import ClientPool, reset_client_pool
from tg_automation.config import load_config


def test_load_config_parses_accounts_and_proxy(tmp_config: Path):
    cfg = load_config(tmp_config)
    assert cfg.account_names() == ["primary", "secondary"]
    assert cfg.accounts[0].api_id == 12345
    assert cfg.accounts[0].proxy is not None
    assert cfg.accounts[0].proxy.kind == "socks5"
    assert cfg.accounts[1].proxy is None


def test_load_config_rejects_empty_accounts(tmp_path: Path):
    p = tmp_path / "bad.json"
    p.write_text(json.dumps({"accounts": []}), encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(p)


def test_load_config_rejects_duplicate_names(tmp_path: Path):
    p = tmp_path / "dup.json"
    p.write_text(json.dumps({
        "accounts": [
            {"name": "x", "api_id": 1, "api_hash": "a", "session": "s"},
            {"name": "x", "api_id": 2, "api_hash": "b", "session": "t"},
        ]
    }), encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(p)


def test_client_pool_rotates_across_accounts(tmp_config: Path):
    reset_client_pool()
    cfg = load_config(tmp_config)
    pool = ClientPool(cfg, client_factory=lambda a: {"name": a.name})
    seq = [pool.acquire()[0].name for _ in range(4)]
    # primary rate_per_minute=2, secondary=3 → expect interleaving:
    # primary, secondary, primary, secondary
    assert seq[:2] == ["primary", "secondary"]
    assert "secondary" in seq


def test_client_pool_caches_clients(tmp_config: Path):
    reset_client_pool()
    cfg = load_config(tmp_config)
    created = []

    def factory(a):
        created.append(a.name)
        return object()

    pool = ClientPool(cfg, client_factory=factory)
    for _ in range(4):
        pool.acquire()
    # Each account instantiated once.
    assert sorted(created) == ["primary", "secondary"]


def test_client_pool_respects_rate_budget(tmp_config: Path):
    reset_client_pool()
    cfg = load_config(tmp_config)
    now = [0.0]
    pool = ClientPool(cfg, client_factory=lambda a: a.name, clock=lambda: now[0])
    # primary=2, secondary=3 -> 5 acquisitions in <60s, the 6th should raise.
    for _ in range(5):
        pool.acquire()
    with pytest.raises(RuntimeError):
        pool.acquire()
    # Advance past the minute and pool should be usable again.
    now[0] = 120.0
    pool.acquire()
