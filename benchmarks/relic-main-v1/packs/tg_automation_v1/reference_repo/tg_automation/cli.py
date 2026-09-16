"""CLI entry point: ``python -m tg_automation`` or the ``tg-automation`` script."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Sequence

from tg_automation.analysis import analyze_interactions
from tg_automation.config import load_config
from tg_automation.engagement import engagement_report
from tg_automation.export import export_report
from tg_automation.filters import filter_users
from tg_automation.growth import (
    create_invite_link,
    schedule_bulk_invites,
    trend_dashboard,
)

LOG = logging.getLogger("tg_automation.cli")


def _parse_dt(value: Optional[str]) -> Optional[datetime]:
    if value is None:
        return None
    return datetime.fromisoformat(value).replace(tzinfo=timezone.utc)


def _maybe_export(rows: List[Dict[str, Any]], path: Optional[str]) -> None:
    if path:
        export_report(rows, path)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="tg-automation", description="Telegram automation toolkit")
    p.add_argument("--config", required=False, default=None, help="Path to config JSON")
    p.add_argument("--since", default=None, help="ISO datetime (UTC) lower bound")
    p.add_argument("--until", default=None, help="ISO datetime (UTC) upper bound")
    sub = p.add_subparsers(dest="command", required=True)

    a = sub.add_parser("analyze", help="Per-user interaction analysis")
    a.add_argument("--group", action="append", required=True, dest="groups")
    a.add_argument("--export", default=None)

    f = sub.add_parser("filter-users", help="Filter analysis output")
    f.add_argument("--group", action="append", required=True, dest="groups")
    f.add_argument("--mode", choices=["active", "inactive", "custom"], required=True)
    f.add_argument("--predicate", action="append", default=[], dest="predicates")
    f.add_argument("--export", default=None)

    e = sub.add_parser("engagement", help="Engagement report across posts")
    e.add_argument("--group", action="append", required=True, dest="groups")
    e.add_argument("--export", default=None)

    g = sub.add_parser("growth", help="Group growth utilities")
    g_sub = g.add_subparsers(dest="growth_cmd", required=True)
    g_invite = g_sub.add_parser("invite-link")
    g_invite.add_argument("--group", required=True)
    g_invite.add_argument("--expires-at", "--expires", dest="expires", default=None)
    g_invite.add_argument("--join-limit", type=int, default=None)
    g_bulk = g_sub.add_parser("bulk-invite")
    g_bulk.add_argument("--group", required=True)
    g_bulk.add_argument("--csv", required=True)
    g_bulk.add_argument("--export", default=None)
    g_trend = g_sub.add_parser("trend")
    g_trend.add_argument("--group", required=True)
    g_trend.add_argument("--export", default=None)

    return p


def _connect_groups(groups: Sequence[str], connector) -> List[str]:
    ok: List[str] = []
    for gid in groups:
        try:
            connector(gid)
            ok.append(gid)
        except Exception as exc:  # noqa: BLE001
            LOG.error("connect to %s failed: %s", gid, exc)
    return ok


def run(
    argv: Sequence[str],
    *,
    source=None,
    growth_client=None,
    connector=None,
    out=sys.stdout,
) -> int:
    parser = build_parser()
    ns = parser.parse_args(list(argv))

    if ns.config:
        load_config(ns.config)

    since = _parse_dt(ns.since)
    until = _parse_dt(ns.until)

    if connector is None:
        def connector(_gid: str) -> None:
            return None

    if ns.command in ("analyze", "filter-users", "engagement"):
        groups = _connect_groups(ns.groups, connector)
        if not groups:
            print(json.dumps({"error": "no reachable groups"}), file=out)
            return 1

    if ns.command == "analyze":
        all_rows: List[Dict[str, Any]] = []
        for gid in groups:
            recs = analyze_interactions(gid, since=since, until=until, source=source)
            for r in recs:
                row = r.as_dict()
                row["group_id"] = gid
                all_rows.append(row)
        _maybe_export(all_rows, ns.export)
        print(json.dumps(all_rows, default=str), file=out)
        return 0

    if ns.command == "filter-users":
        all_rows = []
        for gid in groups:
            recs = analyze_interactions(gid, since=since, until=until, source=source)
            filtered = filter_users(recs, mode=ns.mode, custom=ns.predicates or None)
            for r in filtered:
                row = r.as_dict()
                row["group_id"] = gid
                all_rows.append(row)
        _maybe_export(all_rows, ns.export)
        print(json.dumps(all_rows, default=str), file=out)
        return 0

    if ns.command == "engagement":
        report = engagement_report(groups, since=since, until=until, source=source)
        rows = report["posts"]
        _maybe_export(rows, ns.export)
        print(json.dumps(report, default=str), file=out)
        return 0

    if ns.command == "growth":
        if ns.growth_cmd == "invite-link":
            expires = _parse_dt(ns.expires)
            link = create_invite_link(
                ns.group, growth_client, expires_at=expires, join_limit=ns.join_limit
            )
            print(json.dumps(link.as_dict()), file=out)
            return 0
        if ns.growth_cmd == "bulk-invite":
            results = schedule_bulk_invites(ns.group, ns.csv, growth_client)
            rows = [r.as_dict() for r in results]
            _maybe_export(rows, ns.export)
            print(json.dumps(rows), file=out)
            return 0
        if ns.growth_cmd == "trend":
            rows = trend_dashboard(ns.group, growth_client, since=since, until=until)
            _maybe_export(rows, ns.export)
            print(json.dumps(rows), file=out)
            return 0

    parser.error(f"unknown command: {ns.command}")
    return 2


def main() -> int:  # pragma: no cover - thin wrapper
    logging.basicConfig(level=logging.INFO)
    return run(sys.argv[1:])
