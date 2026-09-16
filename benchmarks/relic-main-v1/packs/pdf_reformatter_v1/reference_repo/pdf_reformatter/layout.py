"""Page layout: measure rows and paginate them."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

from pdf_reformatter.extract import Row

DEFAULT_PAGE_WIDTH = 595.0
DEFAULT_PAGE_HEIGHT = 842.0
DEFAULT_TOP_MARGIN = 50.0
DEFAULT_BOTTOM_MARGIN = 50.0
DEFAULT_FONT_SIZE = 10.0
DEFAULT_LEADING = 12.0
_DEFAULT_AVG_CHAR_WIDTH = 5.0
_CELL_PADDING = 4.0
COL_WIDTHS = [60.0, 120.0, 70.0, 250.0]


@dataclass
class Page:
    rows: List[Row]
    heights: List[float]
    repeated_header: bool = False

    @property
    def total_height(self) -> float:
        return sum(self.heights)


def _wrap_cell(text: str, max_width: float, avg_char_width: float) -> List[str]:
    if not text:
        return [""]
    usable = max(max_width - _CELL_PADDING, 1.0)
    chars_per_line = max(int(usable / max(avg_char_width, 0.1)), 1)

    out: List[str] = []
    for segment in str(text).split("\n"):
        if segment == "":
            out.append("")
            continue
        words = segment.split(" ")
        current = ""
        for word in words:
            candidate = word if not current else current + " " + word
            if len(candidate) <= chars_per_line:
                current = candidate
                continue
            if current:
                out.append(current)
                current = ""
            while len(word) > chars_per_line:
                out.append(word[:chars_per_line])
                word = word[chars_per_line:]
            current = word
        out.append(current)
    return out


def measure_row_height(
    row: Row,
    col_widths: Sequence[float],
    font_size: float = DEFAULT_FONT_SIZE,
    leading: float = DEFAULT_LEADING,
    avg_char_width: float = _DEFAULT_AVG_CHAR_WIDTH,
) -> float:
    if len(col_widths) != len(row.cells):
        raise ValueError(
            f"col_widths has {len(col_widths)} entries but row has {len(row.cells)} cells"
        )
    max_lines = 1
    for cell, width in zip(row.cells, col_widths):
        max_lines = max(max_lines, len(_wrap_cell(cell, width, avg_char_width)))
    return max_lines * leading


def paginate(
    rows: Sequence[Row],
    col_widths: Sequence[float],
    page_height: float = DEFAULT_PAGE_HEIGHT,
    top_margin: float = DEFAULT_TOP_MARGIN,
    bottom_margin: float = DEFAULT_BOTTOM_MARGIN,
    font_size: float = DEFAULT_FONT_SIZE,
    leading: float = DEFAULT_LEADING,
    avg_char_width: float = _DEFAULT_AVG_CHAR_WIDTH,
    repeat_header: bool = False,
    min_body_rows_after_header: int = 1,
) -> List[Page]:
    body_height = page_height - top_margin - bottom_margin
    if body_height <= 0:
        raise ValueError("page_height must exceed top_margin + bottom_margin")
    if min_body_rows_after_header < 1:
        raise ValueError("min_body_rows_after_header must be >= 1")
    if not rows:
        return []

    all_rows = list(rows)
    header = all_rows[0] if repeat_header else None
    header_h = (
        measure_row_height(header, col_widths, font_size, leading, avg_char_width)
        if header is not None
        else 0.0
    )

    pages: List[Page] = []
    current_rows: List[Row] = []
    current_heights: List[float] = []
    used = 0.0
    current_repeated = False

    def start_page(with_repeated_header: bool) -> None:
        nonlocal current_rows, current_heights, used, current_repeated
        current_rows = []
        current_heights = []
        used = 0.0
        current_repeated = False
        if with_repeated_header and header is not None:
            current_rows.append(header)
            current_heights.append(header_h)
            used = header_h
            current_repeated = True

    def close_page() -> None:
        nonlocal current_rows, current_heights, used, current_repeated
        if current_rows:
            pages.append(
                Page(
                    rows=current_rows,
                    heights=current_heights,
                    repeated_header=current_repeated,
                )
            )
        current_rows = []
        current_heights = []
        used = 0.0
        current_repeated = False

    start_page(False)
    body_rows = all_rows[1:] if repeat_header else all_rows

    if repeat_header:
        h0 = header_h
        current_rows.append(header)
        current_heights.append(h0)
        used = h0

    for row in body_rows:
        h = measure_row_height(row, col_widths, font_size, leading, avg_char_width)
        # Normal page break: row would not fit.
        if current_rows and used + h > body_height + 1e-6:
            close_page()
            start_page(repeat_header)

        # If repeated header itself leaves no room, still place the oversized
        # row after it; the "oversized alone" rule wins over dropping data.
        current_rows.append(row)
        current_heights.append(h)
        used += h

        # Oversized body rows are allowed, but they should be isolated on
        # their page so they cannot hide following rows.
        if h > body_height + 1e-6 or (
            repeat_header and header_h + h > body_height + 1e-6
        ):
            close_page()
            start_page(repeat_header)

    close_page()

    # Drop a dangling repeated-header-only page created after an oversized
    # row or final close.
    if pages and pages[-1].repeated_header and len(pages[-1].rows) == 1:
        pages.pop()
    return pages
