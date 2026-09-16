"""Tests for tg_automation.growth."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from tg_automation.growth import (
    create_invite_link,
    schedule_bulk_invites,
    trend_dashboard,
)


def _dt(s):
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)


def test_create_invite_link_passes_options(fake_growth_client):
    expires = _dt("2026-06-01T00:00:00")
    link = create_invite_link("g1", fake_growth_client, expires_at=expires, join_limit=10)
    assert link.group_id == "g1"
    assert "g1" in link.url
    assert link.expires_at == expires
    assert link.join_limit == 10
    assert fake_growth_client.create_calls == [("g1", expires, 10)]


def test_create_invite_link_rejects_nonpositive_limit(fake_growth_client):
    with pytest.raises(ValueError):
        create_invite_link("g1", fake_growth_client, join_limit=0)


def test_bulk_invite_invites_all(fake_growth_client, invite_csv: Path):
    results = schedule_bulk_invites("g1", str(invite_csv), fake_growth_client)
    assert [r.status for r in results] == ["invited"] * 3
    assert {u for _, u in fake_growth_client.invited} == {"alice", "bob", "carol"}


def test_bulk_invite_honors_floodwait(fake_growth_client, invite_csv: Path, monkeypatch):
    fake_growth_client.floodwait_remaining = 1
    fake_growth_client.floodwait_seconds = 7
    sleeps = []
    monkeypatch.setattr("tg_automation.retry.time.sleep", lambda s: sleeps.append(s))
    results = schedule_bulk_invites("g1", str(invite_csv), fake_growth_client)
    assert 7.0 in sleeps  # waited the requested duration
    assert all(r.status == "invited" for r in results)


def test_bulk_invite_skips_after_max_attempts(fake_growth_client, tmp_path: Path, monkeypatch):
    csv_path = tmp_path / "single.csv"
    csv_path.write_text("username\nflaky\n", encoding="utf-8")
    fake_growth_client.failures = {"flaky": 99}  # always fails
    monkeypatch.setattr("tg_automation.retry.time.sleep", lambda s: None)
    results = schedule_bulk_invites("g1", str(csv_path), fake_growth_client)
    assert len(results) == 1
    assert results[0].status == "skipped"
    assert "flaky" in results[0].error


def test_bulk_invite_retries_then_succeeds(fake_growth_client, tmp_path: Path, monkeypatch):
    csv_path = tmp_path / "single.csv"
    csv_path.write_text("username\nflaky\n", encoding="utf-8")
    fake_growth_client.network_errors_remaining = 2
    monkeypatch.setattr("tg_automation.retry.time.sleep", lambda s: None)
    results = schedule_bulk_invites("g1", str(csv_path), fake_growth_client)
    assert results[0].status == "invited"


def test_bulk_invite_requires_username_column(tmp_path: Path, fake_growth_client):
    bad = tmp_path / "bad.csv"
    bad.write_text("name\nalice\n", encoding="utf-8")
    with pytest.raises(ValueError):
        schedule_bulk_invites("g1", str(bad), fake_growth_client)


def test_trend_dashboard_emits_daily_rows(fake_growth_client):
    fake_growth_client.membership_events_by_group["g1"] = [
        {"date": "2026-05-01T10:00:00", "kind": "join", "user_id": 1},
        {"date": "2026-05-01T11:00:00", "kind": "message", "user_id": 2},
        {"date": "2026-05-02T09:00:00", "kind": "join", "user_id": 3},
        {"date": "2026-05-02T12:00:00", "kind": "message", "user_id": 3},
    ]
    rows = trend_dashboard(
        "g1", fake_growth_client,
        since=_dt("2026-05-01T00:00:00"),
        until=_dt("2026-05-03T00:00:00"),
    )
    by_day = {r["date"]: r for r in rows}
    assert by_day["2026-05-01"]["new_members"] == 1
    assert by_day["2026-05-01"]["active_members"] == 2
    assert by_day["2026-05-02"]["new_members"] == 1
    assert by_day["2026-05-02"]["active_members"] == 1


def test_trend_dashboard_counts_any_event_kind_as_active(fake_growth_client):
    fake_growth_client.membership_events_by_group["g1"] = [
        {"date": "2026-05-01T10:00:00", "kind": "reaction", "user_id": 7},
        {"date": "2026-05-01T11:00:00", "kind": "leave", "user_id": 8},
    ]
    rows = trend_dashboard(
        "g1", fake_growth_client,
        since=_dt("2026-05-01T00:00:00"),
        until=_dt("2026-05-02T00:00:00"),
    )
    assert rows == [{"date": "2026-05-01", "new_members": 0, "active_members": 2}]


def test_bulk_invite_unexpected_invite_errors_are_skipped(tmp_path: Path):
    csv_path = tmp_path / "single.csv"
    csv_path.write_text("username\nalice\n", encoding="utf-8")

    class BrokenClient:
        def invite_user(self, group_id, username):
            raise ValueError(f"bad target {username}")

    results = schedule_bulk_invites("g1", str(csv_path), BrokenClient())
    assert len(results) == 1
    assert results[0].status == "skipped"
    assert "alice" in results[0].error
