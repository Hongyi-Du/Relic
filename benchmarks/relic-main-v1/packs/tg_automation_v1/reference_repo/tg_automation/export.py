"""CSV / Excel report export."""

from __future__ import annotations

import csv
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

_SNAKE_RE = re.compile(r"[^a-z0-9_]+")


def _normalize_field(name: str) -> str:
    """Coerce ``name`` to a stable snake_case identifier."""
    s = str(name).strip().lower().replace("-", "_").replace(" ", "_")
    s = _SNAKE_RE.sub("_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    if not s:
        raise ValueError("empty field name after normalization")
    return s


def _stringify(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return str(value)
    return value


def _coerce_rows(rows: Iterable[Any]) -> tuple[List[Dict[str, Any]], List[str]]:
    coerced: List[Dict[str, Any]] = []
    fields: List[str] = []
    seen: set[str] = set()
    for row in rows:
        if hasattr(row, "as_dict"):
            row = row.as_dict()
        if not isinstance(row, dict):
            raise TypeError(f"unsupported row type: {type(row).__name__}")
        norm: Dict[str, Any] = {}
        for k, v in row.items():
            nk = _normalize_field(k)
            norm[nk] = _stringify(v)
            if nk not in seen:
                seen.add(nk)
                fields.append(nk)
        coerced.append(norm)
    return coerced, fields


def export_report(rows: Iterable[Any], path: str | os.PathLike[str]) -> str:
    """Write ``rows`` to ``path`` as CSV or XLSX based on extension."""
    p = Path(path)
    ext = p.suffix.lower()
    if ext not in (".csv", ".xlsx"):
        raise ValueError(f"unsupported export extension: {ext!r}; use .csv or .xlsx")
    coerced, fields = _coerce_rows(rows)
    p.parent.mkdir(parents=True, exist_ok=True)

    if ext == ".csv":
        with open(p, "w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            for row in coerced:
                writer.writerow({f: row.get(f, "") for f in fields})
        return str(p)

    # .xlsx
    from openpyxl import Workbook  # type: ignore

    wb = Workbook()
    ws = wb.active
    ws.title = "report"
    ws.append(fields)
    for row in coerced:
        ws.append([row.get(f, "") for f in fields])
    wb.save(p)
    return str(p)
