"""Tests for tg_automation.engagement.engagement_report."""

from __future__ import annotations

from datetime import datetime, timezone

from tg_automation.engagement import engagement_report


def _dt(s):
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)


def test_reports_posts_across_groups(fake_source, window):
    since, until = window
    report = engagement_report(["g1", "g2"], since=since, until=until, source=fake_source)
    posts = report["posts"]
    assert len(posts) == 3
    ids = {(p["group_id"], p["post_id"]) for p in posts}
    assert ids == {("g1", 101), ("g1", 102), ("g2", 201)}


def test_reaction_breakdown_by_emoji(fake_source, window):
    since, until = window
    report = engagement_report(["g1"], since=since, until=until, source=fake_source)
    post = next(p for p in report["posts"] if p["post_id"] == 101)
    assert post["reactions_by_emoji"] == {"👍": 3, "❤️": 1}
    assert post["reactions_total"] == 4
    assert post["replies"] == 2
    assert post["comments"] == 0


def test_user_message_frequency(fake_source, window):
    since, until = window
    report = engagement_report(["g1", "g2"], since=since, until=until, source=fake_source)
    by_user = {u["user_id"]: u for u in report["user_frequency"]}
    assert by_user[1]["messages_sent"] == 1
    assert by_user[2]["messages_sent"] == 1
    assert by_user[9]["messages_sent"] == 1
    assert by_user[1]["messages_per_day"] >= 0


def test_window_clips_old_posts(fake_source):
    fake_source.posts_by_group["g1"].append(
        {"post_id": 999, "author_id": 1, "author_username": "alice",
         "date": "2024-01-01T00:00:00", "reactions": {}, "replies": 0, "comments": 0}
    )
    since = _dt("2026-04-15T00:00:00")
    until = _dt("2026-05-15T00:00:00")
    report = engagement_report(["g1"], since=since, until=until, source=fake_source)
    assert 999 not in {p["post_id"] for p in report["posts"]}


def test_post_rows_include_author_username_and_date(fake_source, window):
    since, until = window
    report = engagement_report(["g1"], since=since, until=until, source=fake_source)
    post = next(p for p in report["posts"] if p["post_id"] == 101)
    assert post["author_username"] == "alice"
    assert post["date"] == "2026-05-01T10:00:00+00:00"
