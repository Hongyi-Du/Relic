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
        raise NotImplementedError('__len__ is not implemented yet')

    def __getitem__(self, idx: int) -> str:
        raise NotImplementedError('__getitem__ is not implemented yet')

def _normalize_cell(value: object) -> str:
    raise NotImplementedError('_normalize_cell is not implemented yet')

def extract_rows(pdf_path: str, n_columns: int | None=None) -> List[Row]:
    """Extract rows from every page of ``pdf_path``.

    The PDF is parsed page by page; each page's table is concatenated.
    When ``n_columns`` is provided every emitted row is padded or
    truncated to that width so downstream writers can rely on a uniform
    schema. When omitted the widest extracted row determines the width
    and shorter rows are padded with empty strings.
    """
    raise NotImplementedError('extract_rows is not implemented yet')

def rows_from_records(records: Sequence[Sequence[str]]) -> List[Row]:
    """Build ``Row`` objects from in-memory records.

    Provided so tests and callers that already hold the cell grid (from
    a non-PDF source, or a fixture) can drive ``paginate`` and the
    writers without round-tripping through pdfplumber.
    """
    raise NotImplementedError('rows_from_records is not implemented yet')
