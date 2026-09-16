"""Tests for tg_automation.analysis.analyze_interactions."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from tg_automation.analysis import analyze_interactions
from tg_automation.retry import NetworkError


def _dt(s: str):
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)


def test_returns_per_user_records(fake_source, window):
    since, until = window
    recs = analyze_interactions("g1", since=since, until=until, source=fake_source)
    by_user = {r.user_id: r for r in recs}
    assert set(by_user) == {1, 2, 3}
    alice = by_user[1]
    assert alice.messages_sent == 2
    assert alice.reactions_given == 1
    assert alice.active_days == 2
    assert alice.last_active_at == _dt("2026-05-02T09:00:00")
    bob = by_user[2]
    assert bob.messages_sent == 1
    assert bob.reactions_given == 2
    assert bob.active_days == 2


def test_window_defaults_to_trailing_30_days(fake_source):
    now = _dt("2026-05-15T00:00:00")
    recs = analyze_interactions("g1", source=fake_source, now=now)
    # All in-window samples should be present; the 2024-01-01 entry filtered.
    for r in recs:
        if r.user_id == 1:
            assert r.messages_sent == 2  # 2024-01-01 excluded


def test_records_have_expected_fields(fake_source, window):
    since, until = window
    recs = analyze_interactions("g1", since=since, until=until, source=fake_source)
    keys = recs[0].as_dict().keys()
    for required in ("user_id", "username", "messages_sent",
                     "reactions_given", "last_active_at", "active_days"):
        assert required in keys


def test_invalid_window_raises(fake_source):
    same = _dt("2026-05-01T00:00:00")
    with pytest.raises(ValueError):
        analyze_interactions("g1", since=same, until=same, source=fake_source)


def test_source_required():
    with pytest.raises(ValueError):
        analyze_interactions("g1")


def test_messages_outside_window_ignored(fake_source):
    since = _dt("2026-05-04T00:00:00")
    until = _dt("2026-05-10T00:00:00")
    recs = analyze_interactions("g1", since=since, until=until, source=fake_source)
    ids = {r.user_id for r in recs}
    # Only carol (2026-05-05) falls in [04, 10)
    assert ids == {3}


def test_retries_network_then_succeeds(fake_source, monkeypatch, window):
    since, until = window
    fake_source.raise_once = NetworkError("transient")
    sleeps = []
    monkeypatch.setattr("tg_automation.retry.time.sleep", lambda s: sleeps.append(s))
    recs = analyze_interactions("g1", since=since, until=until, source=fake_source)
    assert len(recs) == 3
    assert sleeps  # at least one backoff sleep happened
