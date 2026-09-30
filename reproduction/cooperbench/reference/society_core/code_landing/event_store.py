"""Durable append-only event store for resumable code-landing runs."""

from __future__ import annotations

import json
import re
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

from society_core.hashing import canonicalize, stable_hash


@dataclass(frozen=True)
class RunRecord:
    run_id: str
    idempotency_key: str
    task_id: str
    repo_digest: str
    spec_hash: str
    model_snapshot: str
    config_hash: str
    attempt: int
    status: str
    result_hash: str | None
    created_at: str
    updated_at: str
    created: bool = False


@dataclass(frozen=True)
class EventRecord:
    event_id: str
    run_id: str
    sequence: int
    event_type: str
    payload: dict[str, Any]
    payload_hash: str
    previous_hash: str | None
    event_hash: str
    idempotency_key: str
    created_at: str


class SQLiteEventStore:
    def __init__(self, path: Path) -> None:
        self.path = path.resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def start_run(
        self,
        *,
        task_id: str,
        repo_digest: str,
        spec_hash: str,
        model_snapshot: str,
        config_hash: str,
        attempt: int,
    ) -> RunRecord:
        payload = {
            "task_id": task_id,
            "repo_digest": repo_digest,
            "spec_hash": spec_hash,
            "model_snapshot": model_snapshot,
            "config_hash": config_hash,
            "attempt": attempt,
        }
        idempotency_key = stable_hash(payload)
        run_id = f"landing_run_{idempotency_key[:24]}"
        now = _now()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            cursor = connection.execute(
                """
                INSERT OR IGNORE INTO runs (
                    run_id, idempotency_key, task_id, repo_digest, spec_hash,
                    model_snapshot, config_hash, attempt, status, result_hash,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'running', NULL, ?, ?)
                """,
                (
                    run_id,
                    idempotency_key,
                    task_id,
                    repo_digest,
                    spec_hash,
                    model_snapshot,
                    config_hash,
                    attempt,
                    now,
                    now,
                ),
            )
            created = cursor.rowcount == 1
            row = connection.execute(
                "SELECT * FROM runs WHERE idempotency_key = ?",
                (idempotency_key,),
            ).fetchone()
            connection.commit()
        assert row is not None
        return _run_record(row, created=created)

    def append_event(
        self,
        *,
        run_id: str,
        event_type: str,
        payload: Mapping[str, Any],
        idempotency_key: str | None = None,
    ) -> EventRecord:
        event_key = idempotency_key or f"event_{uuid.uuid4().hex}"
        sanitized = redact_sensitive_payload(payload)
        payload_json = json.dumps(
            canonicalize(sanitized),
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        payload_hash = stable_hash(payload_json)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT * FROM events WHERE run_id = ? AND idempotency_key = ?",
                (run_id, event_key),
            ).fetchone()
            if existing is not None:
                if (
                    str(existing["event_type"]) != event_type
                    or str(existing["payload_hash"]) != payload_hash
                ):
                    connection.rollback()
                    raise ValueError("event_idempotency_conflict")
                connection.commit()
                return _event_record(existing)
            run_row = connection.execute(
                "SELECT status FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
            if run_row is None:
                connection.rollback()
                raise KeyError(f"unknown_run:{run_id}")
            if str(run_row["status"]) != "running":
                connection.rollback()
                raise ValueError("terminal_run_rejects_new_event")
            previous = connection.execute(
                "SELECT sequence, event_hash FROM events WHERE run_id = ? ORDER BY sequence DESC LIMIT 1",
                (run_id,),
            ).fetchone()
            sequence = int(previous["sequence"]) + 1 if previous is not None else 1
            previous_hash = (
                str(previous["event_hash"]) if previous is not None else None
            )
            event_payload = {
                "run_id": run_id,
                "sequence": sequence,
                "event_type": event_type,
                "payload_hash": payload_hash,
                "previous_hash": previous_hash,
                "idempotency_key": event_key,
            }
            event_hash = stable_hash(event_payload)
            event_id = f"landing_event_{event_hash[:24]}"
            created_at = _now()
            connection.execute(
                """
                INSERT INTO events (
                    event_id, run_id, sequence, event_type, payload_json,
                    payload_hash, previous_hash, event_hash, idempotency_key,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event_id,
                    run_id,
                    sequence,
                    event_type,
                    payload_json,
                    payload_hash,
                    previous_hash,
                    event_hash,
                    event_key,
                    created_at,
                ),
            )
            row = connection.execute(
                "SELECT * FROM events WHERE event_id = ?", (event_id,)
            ).fetchone()
            connection.commit()
        assert row is not None
        return _event_record(row)

    def append_terminal_event(
        self,
        *,
        run_id: str,
        event_type: str,
        payload: Mapping[str, Any],
        idempotency_key: str,
        status: str,
    ) -> tuple[EventRecord, RunRecord]:
        if not event_type or not idempotency_key:
            raise ValueError("terminal_event_identity_required")
        if not status or status == "running":
            raise ValueError("terminal_run_status_required")
        sanitized = redact_sensitive_payload(payload)
        payload_json = json.dumps(
            canonicalize(sanitized),
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        payload_hash = stable_hash(payload_json)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            run_row = connection.execute(
                "SELECT * FROM runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            if run_row is None:
                connection.rollback()
                raise KeyError(f"unknown_run:{run_id}")
            event_row = connection.execute(
                "SELECT * FROM events WHERE run_id = ? AND idempotency_key = ?",
                (run_id, idempotency_key),
            ).fetchone()
            if event_row is not None:
                if (
                    str(event_row["event_type"]) != event_type
                    or str(event_row["payload_hash"]) != payload_hash
                ):
                    connection.rollback()
                    raise ValueError("event_idempotency_conflict")
                latest = connection.execute(
                    "SELECT event_id FROM events WHERE run_id = ? "
                    "ORDER BY sequence DESC LIMIT 1",
                    (run_id,),
                ).fetchone()
                if latest is None or str(latest["event_id"]) != str(
                    event_row["event_id"]
                ):
                    connection.rollback()
                    raise ValueError("terminal_event_not_final")
            else:
                if str(run_row["status"]) != "running":
                    connection.rollback()
                    raise ValueError("terminal_run_rejects_new_event")
                previous = connection.execute(
                    "SELECT sequence, event_hash FROM events WHERE run_id = ? "
                    "ORDER BY sequence DESC LIMIT 1",
                    (run_id,),
                ).fetchone()
                sequence = int(previous["sequence"]) + 1 if previous is not None else 1
                previous_hash = (
                    str(previous["event_hash"]) if previous is not None else None
                )
                event_payload = {
                    "run_id": run_id,
                    "sequence": sequence,
                    "event_type": event_type,
                    "payload_hash": payload_hash,
                    "previous_hash": previous_hash,
                    "idempotency_key": idempotency_key,
                }
                event_hash = stable_hash(event_payload)
                event_id = f"landing_event_{event_hash[:24]}"
                connection.execute(
                    """
                    INSERT INTO events (
                        event_id, run_id, sequence, event_type, payload_json,
                        payload_hash, previous_hash, event_hash, idempotency_key,
                        created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        event_id,
                        run_id,
                        sequence,
                        event_type,
                        payload_json,
                        payload_hash,
                        previous_hash,
                        event_hash,
                        idempotency_key,
                        _now(),
                    ),
                )
                event_row = connection.execute(
                    "SELECT * FROM events WHERE event_id = ?",
                    (event_id,),
                ).fetchone()
            assert event_row is not None
            existing_hash = run_row["result_hash"]
            existing_status = str(run_row["status"])
            if existing_hash is not None and (
                str(existing_hash) != payload_hash or existing_status != status
            ):
                connection.rollback()
                raise ValueError("run_already_completed_with_different_result")
            if existing_hash is None and existing_status != "running":
                connection.rollback()
                raise ValueError("terminal_run_missing_result_hash")
            connection.execute(
                "UPDATE runs SET status = ?, result_hash = ?, updated_at = ? "
                "WHERE run_id = ?",
                (status, payload_hash, _now(), run_id),
            )
            completed_row = connection.execute(
                "SELECT * FROM runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            connection.commit()
        assert completed_row is not None
        return _event_record(event_row), _run_record(completed_row, created=False)

    def complete_run(self, run_id: str, *, status: str, result_hash: str) -> RunRecord:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
            if row is None:
                connection.rollback()
                raise KeyError(f"unknown_run:{run_id}")
            existing_hash = row["result_hash"]
            existing_status = str(row["status"])
            if existing_hash is not None and (
                str(existing_hash) != result_hash or existing_status != status
            ):
                connection.rollback()
                raise ValueError("run_already_completed_with_different_result")
            latest_event = connection.execute(
                """
                SELECT payload_hash FROM events
                WHERE run_id = ? ORDER BY sequence DESC LIMIT 1
                """,
                (run_id,),
            ).fetchone()
            if latest_event is None or str(latest_event["payload_hash"]) != result_hash:
                connection.rollback()
                raise ValueError("run_result_hash_must_match_latest_event")
            connection.execute(
                "UPDATE runs SET status = ?, result_hash = ?, updated_at = ? WHERE run_id = ?",
                (status, result_hash, _now(), run_id),
            )
            updated = connection.execute(
                "SELECT * FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
            connection.commit()
        assert updated is not None
        return _run_record(updated, created=False)

    def get_run(self, run_id: str) -> RunRecord:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
        if row is None:
            raise KeyError(f"unknown_run:{run_id}")
        return _run_record(row, created=False)

    def find_intent_run(
        self,
        *,
        task_id: str,
        spec_hash: str,
        model_snapshot: str,
        config_hash: str,
        attempt: int,
    ) -> RunRecord | None:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM runs
                WHERE task_id = ? AND spec_hash = ? AND model_snapshot = ?
                  AND config_hash = ? AND attempt = ?
                ORDER BY created_at
                """,
                (task_id, spec_hash, model_snapshot, config_hash, attempt),
            ).fetchall()
        if len(rows) > 1:
            raise ValueError("ambiguous_intent_run")
        if not rows:
            return None
        return _run_record(rows[0], created=False)

    def load_events(self, run_id: str) -> tuple[EventRecord, ...]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM events WHERE run_id = ? ORDER BY sequence",
                (run_id,),
            ).fetchall()
        return tuple(_event_record(row) for row in rows)

    def load_verified_events(
        self,
        run_id: str,
        *,
        terminal_event_type: str | None = None,
    ) -> tuple[EventRecord, ...]:
        if not self.verify_chain(run_id):
            raise ValueError("event_chain_integrity_failure")
        events = self.load_events(run_id)
        run = self.get_run(run_id)
        if terminal_event_type is not None and run.result_hash is not None:
            if not events or events[-1].event_type != terminal_event_type:
                raise ValueError("terminal_event_not_final")
            terminal = next(
                (
                    event
                    for event in reversed(events)
                    if event.event_type == terminal_event_type
                ),
                None,
            )
            if terminal is None or terminal.payload_hash != run.result_hash:
                raise ValueError("terminal_result_hash_mismatch")
        return events

    def verify_chain(self, run_id: str) -> bool:
        previous_hash: str | None = None
        for event in self.load_events(run_id):
            payload_json = json.dumps(
                canonicalize(event.payload),
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            )
            payload_hash = stable_hash(payload_json)
            expected_hash = stable_hash(
                {
                    "run_id": event.run_id,
                    "sequence": event.sequence,
                    "event_type": event.event_type,
                    "payload_hash": payload_hash,
                    "previous_hash": previous_hash,
                    "idempotency_key": event.idempotency_key,
                }
            )
            if (
                payload_hash != event.payload_hash
                or event.previous_hash != previous_hash
                or event.event_hash != expected_hash
            ):
                return False
            previous_hash = event.event_hash
        return True

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    idempotency_key TEXT NOT NULL UNIQUE,
                    task_id TEXT NOT NULL,
                    repo_digest TEXT NOT NULL,
                    spec_hash TEXT NOT NULL,
                    model_snapshot TEXT NOT NULL,
                    config_hash TEXT NOT NULL,
                    attempt INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    result_hash TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS events (
                    event_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL REFERENCES runs(run_id),
                    sequence INTEGER NOT NULL,
                    event_type TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    payload_hash TEXT NOT NULL,
                    previous_hash TEXT,
                    event_hash TEXT NOT NULL,
                    idempotency_key TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE(run_id, sequence),
                    UNIQUE(run_id, idempotency_key)
                );
                CREATE INDEX IF NOT EXISTS events_run_sequence
                    ON events(run_id, sequence);
                CREATE INDEX IF NOT EXISTS runs_intent_lookup
                    ON runs(task_id, spec_hash, model_snapshot, config_hash, attempt);
                """
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA synchronous = FULL")
        return connection


def _run_record(row: sqlite3.Row, *, created: bool) -> RunRecord:
    return RunRecord(
        run_id=str(row["run_id"]),
        idempotency_key=str(row["idempotency_key"]),
        task_id=str(row["task_id"]),
        repo_digest=str(row["repo_digest"]),
        spec_hash=str(row["spec_hash"]),
        model_snapshot=str(row["model_snapshot"]),
        config_hash=str(row["config_hash"]),
        attempt=int(row["attempt"]),
        status=str(row["status"]),
        result_hash=str(row["result_hash"]) if row["result_hash"] is not None else None,
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
        created=created,
    )


def _event_record(row: sqlite3.Row) -> EventRecord:
    payload = json.loads(str(row["payload_json"]))
    return EventRecord(
        event_id=str(row["event_id"]),
        run_id=str(row["run_id"]),
        sequence=int(row["sequence"]),
        event_type=str(row["event_type"]),
        payload=payload,
        payload_hash=str(row["payload_hash"]),
        previous_hash=str(row["previous_hash"])
        if row["previous_hash"] is not None
        else None,
        event_hash=str(row["event_hash"]),
        idempotency_key=str(row["idempotency_key"]),
        created_at=str(row["created_at"]),
    )


def redact_sensitive_payload(value: Any, *, key: str = "") -> Any:
    """Remove credential-like material before durable or prompt-visible use."""

    lowered = key.lower()
    sensitive = (
        "api_key" in lowered
        or lowered == "token"
        or lowered.endswith("_token")
        or "secret" in lowered
        or "password" in lowered
        or "authorization" in lowered
    )
    if sensitive:
        return "[REDACTED]"
    if isinstance(value, Mapping):
        return {
            str(item_key): redact_sensitive_payload(
                item_value,
                key=str(item_key),
            )
            for item_key, item_value in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [redact_sensitive_payload(item) for item in value]
    if isinstance(value, str):
        redacted = value
        for pattern in _SECRET_TEXT_PATTERNS:
            redacted = pattern.sub("[REDACTED]", redacted)
        return redacted
    return value


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds")


_SECRET_TEXT_PATTERNS = (
    re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/-]{20,}=*"),
)
