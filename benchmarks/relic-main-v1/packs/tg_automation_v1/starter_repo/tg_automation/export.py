"""CSV / Excel report export."""
from __future__ import annotations
import csv
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence
_SNAKE_RE = re.compile('[^a-z0-9_]+')

def _normalize_field(name: str) -> str:
    """Coerce ``name`` to a stable snake_case identifier."""
    raise NotImplementedError('_normalize_field is not implemented yet')

def _stringify(value: Any) -> Any:
    raise NotImplementedError('_stringify is not implemented yet')

def _coerce_rows(rows: Iterable[Any]) -> tuple[List[Dict[str, Any]], List[str]]:
    raise NotImplementedError('_coerce_rows is not implemented yet')

def export_report(rows: Iterable[Any], path: str | os.PathLike[str]) -> str:
    """Write ``rows`` to ``path`` as CSV or XLSX based on extension."""
    raise NotImplementedError('export_report is not implemented yet')
