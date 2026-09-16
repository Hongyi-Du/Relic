"""Tests for tg_automation.cli."""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from tg_automation.cli import run


def test_analyze_subcommand(fake_source, tmp_config: Path):
    buf = io.StringIO()
    rc = run(
        ["--config", str(tmp_config),
         "--since", "2026-04-15T00:00:00",
         "--until", "2026-05-15T00:00:00",
         "analyze", "--group", "g1"],
        source=fake_source, out=buf,
    )
    assert rc == 0
    data = json.loads(buf.getvalue())
    assert any(r["user_id"] == 1 for r in data)


def test_filter_users_active(fake_source, tmp_config: Path):
    buf = io.StringIO()
    rc = run(
        ["--config", str(tmp_config),
         "--since", "2026-04-15T00:00:00",
         "--until", "2026-05-15T00:00:00",
         "filter-users", "--group", "g1", "--mode", "active"],
        source=fake_source, out=buf,
    )
    assert rc == 0
    data = json.loads(buf.getvalue())
    assert all(r["messages_sent"] + r["reactions_given"] >= 1 for r in data)


def test_filter_users_custom(fake_source, tmp_config: Path):
    buf = io.StringIO()
    rc = run(
        ["--config", str(tmp_config),
         "--since", "2026-04-15T00:00:00",
         "--until", "2026-05-15T00:00:00",
         "filter-users", "--group", "g1",
         "--mode", "custom", "--predicate", "messages_sent>=2"],
        source=fake_source, out=buf,
    )
    assert rc == 0
    data = json.loads(buf.getvalue())
    assert {r["user_id"] for r in data} == {1}


def test_engagement_subcommand(fake_source, tmp_config: Path):
    buf = io.StringIO()
    rc = run(
        ["--config", str(tmp_config),
         "--since", "2026-04-15T00:00:00",
         "--until", "2026-05-15T00:00:00",
         "engagement", "--group", "g1", "--group", "g2"],
        source=fake_source, out=buf,
    )
    assert rc == 0
    payload = json.loads(buf.getvalue())
    assert "posts" in payload
    assert "user_frequency" in payload


def test_growth_invite_link_subcommand(fake_growth_client, tmp_config: Path):
    buf = io.StringIO()
    rc = run(
        ["--config", str(tmp_config),
         "growth", "invite-link", "--group", "g1", "--join-limit", "5"],
        growth_client=fake_growth_client, out=buf,
    )
    assert rc == 0
    data = json.loads(buf.getvalue())
    assert data["group_id"] == "g1"
    assert data["join_limit"] == 5


def test_growth_bulk_invite_subcommand(fake_growth_client, tmp_config: Path, invite_csv: Path):
    buf = io.StringIO()
    rc = run(
        ["--config", str(tmp_config),
         "growth", "bulk-invite", "--group", "g1", "--csv", str(invite_csv)],
        growth_client=fake_growth_client, out=buf,
    )
    assert rc == 0
    rows = json.loads(buf.getvalue())
    assert {r["username"] for r in rows} == {"alice", "bob", "carol"}


def test_growth_trend_subcommand(fake_growth_client, tmp_config: Path):
    fake_growth_client.membership_events_by_group["g1"] = [
        {"date": "2026-05-01T10:00:00", "kind": "join", "user_id": 1},
    ]
    buf = io.StringIO()
    rc = run(
        ["--config", str(tmp_config),
         "--since", "2026-05-01T00:00:00",
         "--until", "2026-05-03T00:00:00",
         "growth", "trend", "--group", "g1"],
        growth_client=fake_growth_client, out=buf,
    )
    assert rc == 0
    rows = json.loads(buf.getvalue())
    assert any(r["date"] == "2026-05-01" and r["new_members"] == 1 for r in rows)


def test_export_writes_csv(tmp_path: Path, fake_source, tmp_config: Path):
    out_csv = tmp_path / "out.csv"
    buf = io.StringIO()
    rc = run(
        ["--config", str(tmp_config),
         "--since", "2026-04-15T00:00:00",
         "--until", "2026-05-15T00:00:00",
         "analyze", "--group", "g1", "--export", str(out_csv)],
        source=fake_source, out=buf,
    )
    assert rc == 0
    text = out_csv.read_text(encoding="utf-8")
    assert "user_id" in text


def test_failed_group_does_not_abort_others(fake_source, tmp_config: Path):
    """One unreachable group is logged but processing continues for the rest."""
    seen = {"calls": []}

    def connector(gid):
        seen["calls"].append(gid)
        if gid == "broken":
            raise RuntimeError("can't connect")

    buf = io.StringIO()
    rc = run(
        ["--config", str(tmp_config),
         "--since", "2026-04-15T00:00:00",
         "--until", "2026-05-15T00:00:00",
         "analyze", "--group", "broken", "--group", "g1"],
        source=fake_source, out=buf, connector=connector,
    )
    assert rc == 0
    data = json.loads(buf.getvalue())
    assert all(r["group_id"] == "g1" for r in data)
    assert "broken" in seen["calls"] and "g1" in seen["calls"]


def test_growth_invite_link_accepts_expires_at_flag(fake_growth_client, tmp_config: Path):
    buf = io.StringIO()
    rc = run(
        ["--config", str(tmp_config),
         "growth", "invite-link", "--group", "g1",
         "--expires-at", "2026-06-01T00:00:00", "--join-limit", "5"],
        growth_client=fake_growth_client, out=buf,
    )
    assert rc == 0
    data = json.loads(buf.getvalue())
    assert data["expires_at"].startswith("2026-06-01T00:00:00")
    assert fake_growth_client.create_calls[0][1].isoformat().startswith("2026-06-01T00:00:00")
