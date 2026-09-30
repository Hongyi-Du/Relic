"""Read-only evidence-pack builders for paper-grade Society-Core runs."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .experiments import write_json
from .platform_calibration import (
    PlatformCalibrationReport,
    PlatformSource,
    build_platform_calibration_report,
    records_from_raw_payloads,
)


DEFAULT_REDDIT_SUBREDDITS = ("MachineLearning", "programming", "LocalLLaMA", "artificial")
DEFAULT_GITHUB_REPOS = ("pydantic/pydantic-ai", "microsoft/autogen", "langchain-ai/langchain")


@dataclass(frozen=True)
class CommandCapture:
    source_id: str
    command: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str


@dataclass(frozen=True)
class PlatformEvidencePack:
    report: PlatformCalibrationReport
    output_dir: str
    raw_payload_paths: dict[str, str]
    command_captures: tuple[CommandCapture, ...] = field(default_factory=tuple)


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _json_rows(stdout: str) -> list[dict[str, Any]]:
    if not stdout.strip():
        return []
    parsed = json.loads(stdout)
    if isinstance(parsed, list):
        return [row for row in parsed if isinstance(row, dict)]
    if isinstance(parsed, dict):
        for key in ("items", "data", "results", "nodes"):
            value = parsed.get(key)
            if isinstance(value, list):
                return [row for row in value if isinstance(row, dict)]
        return [parsed]
    return []


def _run_capture(command: tuple[str, ...], *, timeout_seconds: float) -> CommandCapture:
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )
    return CommandCapture(
        source_id="",
        command=command,
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def _capture_with_source(
    source_id: str,
    command: tuple[str, ...],
    *,
    timeout_seconds: float,
) -> CommandCapture:
    try:
        capture = _run_capture(command, timeout_seconds=timeout_seconds)
        return CommandCapture(
            source_id=source_id,
            command=capture.command,
            returncode=capture.returncode,
            stdout=capture.stdout,
            stderr=capture.stderr,
        )
    except Exception as exc:  # pragma: no cover - exercised by integration failures
        return CommandCapture(
            source_id=source_id,
            command=command,
            returncode=124,
            stdout="",
            stderr=f"{type(exc).__name__}: {exc}",
        )


def _source_url(platform: str, query: str) -> str:
    if platform == "hacker_news":
        return "https://news.ycombinator.com"
    if platform == "reddit":
        return f"https://www.reddit.com/r/{query}/hot/"
    if platform == "github":
        return f"https://github.com/{query}/issues"
    return ""


def _write_raw_payloads(
    *,
    output_dir: Path,
    raw_payloads: dict[str, list[dict[str, Any]]],
    captures: tuple[CommandCapture, ...],
) -> dict[str, str]:
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    for source_id, rows in sorted(raw_payloads.items()):
        path = raw_dir / f"{source_id}.json"
        write_json(path, rows)
        paths[source_id] = str(path)
    capture_path = raw_dir / "command_captures.json"
    write_json(capture_path, captures)
    paths["command_captures"] = str(capture_path)
    return paths


def collect_platform_evidence_pack(
    *,
    output_dir: Path,
    reddit_subreddits: tuple[str, ...] = DEFAULT_REDDIT_SUBREDDITS,
    github_repos: tuple[str, ...] = DEFAULT_GITHUB_REPOS,
    hn_limit: int = 30,
    reddit_limit: int = 25,
    github_limit: int = 30,
    timeout_seconds: float = 45.0,
    collected_at_utc: str | None = None,
) -> PlatformEvidencePack:
    output_dir.mkdir(parents=True, exist_ok=True)
    collected_at = collected_at_utc or _utc_now()
    command_specs: list[tuple[str, str, str, tuple[str, ...]]] = [
        (
            "hn_top",
            "hacker_news",
            "top",
            ("opencli", "hackernews", "top", "--limit", str(hn_limit), "-f", "json"),
        ),
        (
            "hn_show",
            "hacker_news",
            "show",
            ("opencli", "hackernews", "show", "--limit", str(hn_limit), "-f", "json"),
        ),
    ]
    for subreddit in reddit_subreddits:
        source_id = f"reddit_{subreddit.lower().replace('-', '_')}"
        command_specs.append(
            (
                source_id,
                "reddit",
                subreddit,
                ("opencli", "reddit", "hot", "--subreddit", subreddit, "--limit", str(reddit_limit), "-f", "json"),
            )
        )
    for repo in github_repos:
        source_id = f"github_{repo.replace('/', '_').replace('-', '_')}"
        command_specs.append(
            (
                source_id,
                "github",
                repo,
                (
                    "gh",
                    "issue",
                    "list",
                    "--repo",
                    repo,
                    "--state",
                    "all",
                    "--limit",
                    str(github_limit),
                    "--json",
                    "number,title,author,comments,state,labels,url",
                ),
            )
        )
    captures: list[CommandCapture] = []
    sources: list[PlatformSource] = []
    raw_payloads: dict[str, list[dict[str, Any]]] = {}
    for source_id, platform, query, command in command_specs:
        capture = _capture_with_source(source_id, command, timeout_seconds=timeout_seconds)
        captures.append(capture)
        caveats: list[str] = []
        rows: list[dict[str, Any]] = []
        if capture.returncode != 0:
            caveats.append(f"command_failed:{capture.returncode}")
        else:
            try:
                rows = _json_rows(capture.stdout)
            except json.JSONDecodeError:
                caveats.append("json_decode_failed")
        if not rows:
            caveats.append("empty_source_rows")
        raw_payloads[source_id] = rows
        sources.append(
            PlatformSource(
                source_id=source_id,
                platform=platform,
                query=query,
                source_url=_source_url(platform, query),
                collected_at_utc=collected_at,
                command=" ".join(command),
                caveats=tuple(caveats),
            )
        )
    forum_records, feedback_records = records_from_raw_payloads(raw_payloads, tuple(sources))
    report = build_platform_calibration_report(
        sources=tuple(sources),
        forum_records=forum_records,
        feedback_records=feedback_records,
    )
    raw_paths = _write_raw_payloads(output_dir=output_dir, raw_payloads=raw_payloads, captures=tuple(captures))
    write_json(output_dir / "platform_samples.json", {
        "sources": sources,
        "forum_records": forum_records,
        "feedback_records": feedback_records,
    })
    write_json(output_dir / "platform_calibration_report.json", report)
    write_json(output_dir / "calibration_targets.json", report.calibration_targets)
    write_json(output_dir / "platform_evidence_index.json", {
        "report_hash": report.hash(),
        "raw_payload_paths": raw_paths,
        "platform_count": len(sources),
        "forum_record_count": len(forum_records),
        "feedback_record_count": len(feedback_records),
        "caveats": report.caveats,
    })
    return PlatformEvidencePack(
        report=report,
        output_dir=str(output_dir),
        raw_payload_paths=raw_paths,
        command_captures=tuple(captures),
    )
