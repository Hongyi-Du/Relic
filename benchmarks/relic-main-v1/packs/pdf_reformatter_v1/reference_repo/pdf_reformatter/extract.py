"""Extract table rows from a source PDF.

The source PDFs are laid out as a fixed-column table. We extract one
``Row`` per data row preserving column order; empty cells are kept as
empty strings rather than dropped so the column contract is stable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence

import pdfplumber


@dataclass
class Row:
    """One table row. ``cells`` is in source column order."""

    cells: List[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.cells)

    def __getitem__(self, idx: int) -> str:
        return self.cells[idx]


def _normalize_cell(value: object) -> str:
    if value is None:
        return ""
    text = str(value)
    # pdfplumber leaves newlines in multi-line cells; keep them so the
    # right-hand column's variable-length text can be measured per line.
    return text.replace("\r\n", "\n").replace("\r", "\n").strip("\n").strip()


def extract_rows(pdf_path: str, n_columns: int | None = None) -> List[Row]:
    """Extract rows from every page of ``pdf_path``.

    The PDF is parsed page by page; each page's table is concatenated.
    When ``n_columns`` is provided every emitted row is padded or
    truncated to that width so downstream writers can rely on a uniform
    schema. When omitted the widest extracted row determines the width
    and shorter rows are padded with empty strings.
    """

    raw_rows: List[List[str]] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables() or []
            for table in tables:
                for row in table:
                    raw_rows.append([_normalize_cell(c) for c in row])

    if not raw_rows:
        return []

    width = n_columns if n_columns is not None else max(len(r) for r in raw_rows)
    normalized: List[Row] = []
    for cells in raw_rows:
        if len(cells) < width:
            cells = cells + [""] * (width - len(cells))
        elif len(cells) > width:
            cells = cells[:width]
        normalized.append(Row(cells=list(cells)))
    return normalized


def rows_from_records(records: Sequence[Sequence[str]]) -> List[Row]:
    """Build ``Row`` objects from in-memory records.

    Provided so tests and callers that already hold the cell grid (from
    a non-PDF source, or a fixture) can drive ``paginate`` and the
    writers without round-tripping through pdfplumber.
    """

    if not records:
        return []
    width = max(len(r) for r in records)
    out: List[Row] = []
    for rec in records:
        cells = [_normalize_cell(c) for c in rec]
        if len(cells) < width:
            cells = cells + [""] * (width - len(cells))
        out.append(Row(cells=cells))
    return out
