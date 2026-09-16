"""Command-line entry point for the reformatter."""
from __future__ import annotations
import argparse
from pathlib import Path
from typing import List, Sequence
from pdf_reformatter.extract import Row, extract_rows
from pdf_reformatter.layout import DEFAULT_BOTTOM_MARGIN, DEFAULT_FONT_SIZE, DEFAULT_LEADING, DEFAULT_PAGE_HEIGHT, DEFAULT_PAGE_WIDTH, DEFAULT_TOP_MARGIN, paginate
from pdf_reformatter.writers import write_csv, write_docx, write_pdf

def _parse_widths(spec: str | None, n_columns: int, page_width: float) -> List[float]:
    raise NotImplementedError('_parse_widths is not implemented yet')

def reformat(input_pdf: str | Path, output_pdf: str | Path, output_docx: str | Path, output_csv: str | Path, col_widths: Sequence[float] | None=None, page_width: float=DEFAULT_PAGE_WIDTH, page_height: float=DEFAULT_PAGE_HEIGHT, top_margin: float=DEFAULT_TOP_MARGIN, bottom_margin: float=DEFAULT_BOTTOM_MARGIN, font_size: float=DEFAULT_FONT_SIZE, leading: float=DEFAULT_LEADING, repeat_header: bool=False) -> dict:
    """Library entry point. Returns a dict of the four output paths."""
    raise NotImplementedError('reformat is not implemented yet')

def build_parser() -> argparse.ArgumentParser:
    raise NotImplementedError('build_parser is not implemented yet')

def main(argv: Sequence[str] | None=None) -> int:
    raise NotImplementedError('main is not implemented yet')
if __name__ == '__main__':
    raise SystemExit(main())
