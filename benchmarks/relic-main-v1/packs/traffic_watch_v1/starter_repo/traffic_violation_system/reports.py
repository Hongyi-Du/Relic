"""Report exporter: CSV violation rows + per-violation snapshot copies."""
from __future__ import annotations
import csv
import shutil
from pathlib import Path
from typing import Iterable
from .violations import ViolationRecord
CSV_COLUMNS = ['violation_id', 'track_id', 'violation_type', 'timestamp', 'camera_id', 'location', 'plate_text', 'snapshot_path', 'measured_value']

class ReportExporter:

    def export_csv(self, records: Iterable[ViolationRecord], path: str | Path) -> Path:
        raise NotImplementedError('export_csv is not implemented yet')

    def export_snapshots(self, records: Iterable[ViolationRecord], dest_dir: str | Path) -> list[Path]:
        raise NotImplementedError('export_snapshots is not implemented yet')
