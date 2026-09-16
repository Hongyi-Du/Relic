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

    def save(self, record: ViolationRecord) -> None:
        raise NotImplementedError('save is not implemented yet')

    def list_recent(self, limit: int=50) -> List[ViolationRecord]:
        raise NotImplementedError('list_recent is not implemented yet')

    def all(self) -> List[ViolationRecord]:
        raise NotImplementedError('all is not implemented yet')

class SQLiteViolationStore(ViolationStore):
    SCHEMA = '\n    CREATE TABLE IF NOT EXISTS violation_records (\n        violation_id TEXT PRIMARY KEY,\n        track_id INTEGER NOT NULL,\n        violation_type TEXT NOT NULL,\n        timestamp TEXT NOT NULL,\n        camera_id TEXT NOT NULL,\n        location TEXT NOT NULL,\n        plate_text TEXT,\n        snapshot_path TEXT NOT NULL,\n        measured_value REAL\n    );\n    '

    def __init__(self, db_path: str | Path=':memory:') -> None:
        raise NotImplementedError('__init__ is not implemented yet')

    def close(self) -> None:
        raise NotImplementedError('close is not implemented yet')

    def save(self, record: ViolationRecord) -> None:
        raise NotImplementedError('save is not implemented yet')

    def _row_to_record(self, row) -> ViolationRecord:
        raise NotImplementedError('_row_to_record is not implemented yet')

    def list_recent(self, limit: int=50) -> List[ViolationRecord]:
        raise NotImplementedError('list_recent is not implemented yet')

    def all(self) -> List[ViolationRecord]:
        raise NotImplementedError('all is not implemented yet')
