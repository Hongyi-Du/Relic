"""Persistent storage for violation records.

`ViolationStore` is the abstract interface; `SQLiteViolationStore` is the
default backend. MySQL/MongoDB backends would implement the same interface.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from .violations import ViolationRecord, ViolationType


class ViolationStore:
    def save(self, record: ViolationRecord) -> None:  # pragma: no cover
        raise NotImplementedError

    def list_recent(self, limit: int = 50) -> List[ViolationRecord]:  # pragma: no cover
        raise NotImplementedError

    def all(self) -> List[ViolationRecord]:  # pragma: no cover
        raise NotImplementedError


class SQLiteViolationStore(ViolationStore):
    SCHEMA = """
    CREATE TABLE IF NOT EXISTS violation_records (
        violation_id TEXT PRIMARY KEY,
        track_id INTEGER NOT NULL,
        violation_type TEXT NOT NULL,
        timestamp TEXT NOT NULL,
        camera_id TEXT NOT NULL,
        location TEXT NOT NULL,
        plate_text TEXT,
        snapshot_path TEXT NOT NULL,
        measured_value REAL
    );
    """

    def __init__(self, db_path: str | Path = ":memory:") -> None:
        self._path = str(db_path)
        if self._path != ":memory:":
            Path(self._path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute(self.SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def save(self, record: ViolationRecord) -> None:
        self._conn.execute(
            """INSERT OR REPLACE INTO violation_records
               (violation_id, track_id, violation_type, timestamp, camera_id,
                location, plate_text, snapshot_path, measured_value)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                record.violation_id,
                record.track_id,
                record.violation_type.value,
                record.timestamp.isoformat(),
                record.camera_id,
                record.location,
                record.plate_text,
                record.snapshot_path,
                record.measured_value,
            ),
        )
        self._conn.commit()

    def _row_to_record(self, row) -> ViolationRecord:
        return ViolationRecord(
            violation_id=row["violation_id"],
            track_id=row["track_id"],
            violation_type=ViolationType(row["violation_type"]),
            timestamp=datetime.fromisoformat(row["timestamp"]),
            camera_id=row["camera_id"],
            location=row["location"],
            plate_text=row["plate_text"],
            snapshot_path=row["snapshot_path"],
            measured_value=row["measured_value"],
        )

    def list_recent(self, limit: int = 50) -> List[ViolationRecord]:
        cur = self._conn.execute(
            "SELECT * FROM violation_records ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        )
        return [self._row_to_record(r) for r in cur.fetchall()]

    def all(self) -> List[ViolationRecord]:
        cur = self._conn.execute("SELECT * FROM violation_records ORDER BY timestamp ASC")
        return [self._row_to_record(r) for r in cur.fetchall()]