"""Emit reformatted output as PDF, DOCX, and CSV.

Each writer takes the already-paginated rows (or raw rows for the
formats that do not paginate) and produces a file at the requested
path. Writers never re-measure or re-paginate — that responsibility
lives in ``pdf_reformatter.layout``.
"""
from __future__ import annotations
import csv
from pathlib import Path
from typing import List, Sequence
from docx import Document
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from pdf_reformatter.extract import Row
from pdf_reformatter.layout import DEFAULT_BOTTOM_MARGIN, DEFAULT_FONT_SIZE, DEFAULT_LEADING, DEFAULT_PAGE_HEIGHT, DEFAULT_PAGE_WIDTH, DEFAULT_TOP_MARGIN, Page, _wrap_cell

def write_pdf(pages: Sequence[Page], output_path: str | Path, col_widths: Sequence[float], page_width: float=DEFAULT_PAGE_WIDTH, page_height: float=DEFAULT_PAGE_HEIGHT, top_margin: float=DEFAULT_TOP_MARGIN, bottom_margin: float=DEFAULT_BOTTOM_MARGIN, font_size: float=DEFAULT_FONT_SIZE, leading: float=DEFAULT_LEADING, avg_char_width: float=5.0) -> Path:
    """Write paginated rows to a PDF file.

    Each ``Page`` becomes one PDF page. Rows are drawn with the same
    column widths used to paginate them so the rendered geometry
    matches what ``paginate`` measured. A box is drawn around every
    cell to keep the table appearance of the source PDF.
    """
    raise NotImplementedError('write_pdf is not implemented yet')

def write_docx(rows: Sequence[Row], output_path: str | Path) -> Path:
    """Write rows to a DOCX file as a single table.

    Column order matches the source; empty cells are preserved as empty
    string cells (not skipped) so the DOCX schema is row-cell-uniform.
    """
    raise NotImplementedError('write_docx is not implemented yet')

def write_csv(rows: Sequence[Row], output_path: str | Path) -> Path:
    """Write rows to a CSV file using the stdlib ``csv`` module.

    Empty cells are written as empty fields. The CSV is round-trippable
    by ``csv.reader`` without errors.
    """
    raise NotImplementedError('write_csv is not implemented yet')
