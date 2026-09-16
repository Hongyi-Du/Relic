"""Shared fixtures: fake Telegram source / growth client and sample data."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


def _dt(s: str) -> datetime:
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)


@dataclass
class FakeSource:
    messages_by_group: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    posts_by_group: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    calls: List[tuple] = field(default_factory=list)
    raise_once: Optional[Exception] = None

    def iter_messages(self, group_id: str, since: datetime, until: datetime) -> Iterable[Dict[str, Any]]:
        self.calls.append(("iter_messages", group_id))
        if self.raise_once is not None:
            exc = self.raise_once
            self.raise_once = None
            raise exc
        return list(self.messages_by_group.get(group_id, []))

    def iter_posts(self, group_id: str, since: datetime, until: datetime) -> Iterable[Dict[str, Any]]:
        self.calls.append(("iter_posts", group_id))
        return list(self.posts_by_group.get(group_id, []))


@dataclass
class FakeGrowthClient:
    invite_url: str = "https://t.me/+abcXYZ"
    floodwait_remaining: int = 0
    floodwait_seconds: int = 0
    network_errors_remaining: int = 0
    failures: Dict[str, int] = field(default_factory=dict)
    invited: List[tuple] = field(default_factory=list)
    membership_events_by_group: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    create_calls: List[tuple] = field(default_factory=list)

    def create_invite_link(
        self,
        group_id: str,
        expires_at: Optional[datetime],
        join_limit: Optional[int],
    ) -> str:
        self.create_calls.append((group_id, expires_at, join_limit))
        return f"{self.invite_url}?gid={group_id}"

    def invite_user(self, group_id: str, username: str) -> None:
        from tg_automation.retry import FloodWaitError, NetworkError

        if self.floodwait_remaining > 0:
            self.floodwait_remaining -= 1
            raise FloodWaitError(seconds=self.floodwait_seconds or 1)
        if self.network_errors_remaining > 0:
            self.network_errors_remaining -= 1
            raise NetworkError(f"transient drop inviting {username}")
        if self.failures.get(username, 0) > 0:
            self.failures[username] -= 1
            raise NetworkError(f"failing {username}")
        self.invited.append((group_id, username))

    def iter_membership_events(
        self, group_id: str, since: datetime, until: datetime
    ) -> Iterable[Dict[str, Any]]:
        return list(self.membership_events_by_group.get(group_id, []))


@pytest.fixture
def sample_messages() -> List[Dict[str, Any]]:
    return [
        {"sender_id": 1, "username": "alice", "kind": "message",
         "date": "2026-05-01T10:00:00"},
        {"sender_id": 1, "username": "alice", "kind": "message",
         "date": "2026-05-01T11:00:00"},
        {"sender_id": 1, "username": "alice", "kind": "reaction", "count": 1,
         "date": "2026-05-02T09:00:00"},
        {"sender_id": 2, "username": "bob", "kind": "message",
         "date": "2026-05-01T12:00:00"},
        {"sender_id": 2, "username": "bob", "kind": "reaction", "count": 2,
         "date": "2026-05-03T15:00:00"},
        {"sender_id": 3, "username": "carol", "kind": "message",
         "date": "2026-05-05T08:00:00"},
        # outside window — must be filtered out
        {"sender_id": 1, "username": "alice", "kind": "message",
         "date": "2024-01-01T00:00:00"},
    ]


@pytest.fixture
def fake_source(sample_messages: List[Dict[str, Any]]) -> FakeSource:
    src = FakeSource()
    src.messages_by_group["g1"] = list(sample_messages)
    src.messages_by_group["g2"] = [
        {"sender_id": 9, "username": "dave", "kind": "message",
         "date": "2026-05-04T10:00:00"},
    ]
    src.posts_by_group["g1"] = [
        {
            "post_id": 101, "author_id": 1, "author_username": "alice",
            "date": "2026-05-01T10:00:00",
            "reactions": {"👍": 3, "❤️": 1}, "replies": 2, "comments": 0,
        },
        {
            "post_id": 102, "author_id": 2, "author_username": "bob",
            "date": "2026-05-02T10:00:00",
            "reactions": {"👍": 1}, "replies": 0, "comments": 5,
        },
    ]
    src.posts_by_group["g2"] = [
        {
            "post_id": 201, "author_id": 9, "author_username": "dave",
            "date": "2026-05-03T10:00:00",
            "reactions": {"🔥": 2}, "replies": 1, "comments": 1,
        },
    ]
    return src


@pytest.fixture
def fake_growth_client() -> FakeGrowthClient:
    return FakeGrowthClient()


@pytest.fixture
def window():
    return _dt("2026-04-15T00:00:00"), _dt("2026-05-15T00:00:00")


@pytest.fixture
def tmp_config(tmp_path: Path) -> Path:
    cfg = {
        "default_window_days": 30,
        "accounts": [
            {
                "name": "primary",
                "api_id": 12345,
                "api_hash": "deadbeef",
                "session": "/tmp/primary.session",
                "rate_per_minute": 2,
                "proxy": {"kind": "socks5", "host": "127.0.0.1", "port": 1080},
            },
            {
                "name": "secondary",
                "api_id": 67890,
                "api_hash": "cafef00d",
                "session": "/tmp/secondary.session",
                "rate_per_minute": 3,
            },
        ],
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(cfg), encoding="utf-8")
    return path


@pytest.fixture
def invite_csv(tmp_path: Path) -> Path:
    p = tmp_path / "invites.csv"
    p.write_text("username\nalice\nbob\ncarol\n", encoding="utf-8")
    return p
