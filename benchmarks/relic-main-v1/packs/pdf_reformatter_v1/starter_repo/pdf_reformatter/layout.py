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
        raise NotImplementedError('total_height is not implemented yet')

def _wrap_cell(text: str, max_width: float, avg_char_width: float) -> List[str]:
    raise NotImplementedError('_wrap_cell is not implemented yet')

def measure_row_height(row: Row, col_widths: Sequence[float], font_size: float=DEFAULT_FONT_SIZE, leading: float=DEFAULT_LEADING, avg_char_width: float=_DEFAULT_AVG_CHAR_WIDTH) -> float:
    raise NotImplementedError('measure_row_height is not implemented yet')

def paginate(rows: Sequence[Row], col_widths: Sequence[float], page_height: float=DEFAULT_PAGE_HEIGHT, top_margin: float=DEFAULT_TOP_MARGIN, bottom_margin: float=DEFAULT_BOTTOM_MARGIN, font_size: float=DEFAULT_FONT_SIZE, leading: float=DEFAULT_LEADING, avg_char_width: float=_DEFAULT_AVG_CHAR_WIDTH, repeat_header: bool=False, min_body_rows_after_header: int=1) -> List[Page]:
    raise NotImplementedError('paginate is not implemented yet')
