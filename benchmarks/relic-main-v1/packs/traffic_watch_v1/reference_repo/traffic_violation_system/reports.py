"""Report exporter: CSV violation rows + per-violation snapshot copies."""

from __future__ import annotations

import csv
import shutil
from pathlib import Path
from typing import Iterable

from .violations import ViolationRecord


CSV_COLUMNS = [
    "violation_id",
    "track_id",
    "violation_type",
    "timestamp",
    "camera_id",
    "location",
    "plate_text",
    "snapshot_path",
    "measured_value",
]


class ReportExporter:
    def export_csv(self, records: Iterable[ViolationRecord], path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            for r in records:
                row = r.to_dict()
                writer.writerow({k: row.get(k, "") for k in CSV_COLUMNS})
        return out

    def export_snapshots(
        self, records: Iterable[ViolationRecord], dest_dir: str | Path
    ) -> list[Path]:
        dest = Path(dest_dir)
        dest.mkdir(parents=True, exist_ok=True)
        copied: list[Path] = []
        for r in records:
            src = Path(r.snapshot_path)
            if src.exists():
                target = dest / f"{r.violation_id}{src.suffix or '.jpg'}"
                shutil.copy2(src, target)
                copied.append(target)
        return copied
