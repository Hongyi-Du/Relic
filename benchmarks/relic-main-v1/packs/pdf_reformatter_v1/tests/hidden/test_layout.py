"""Pagination: rows sized by actual cell height, no row straddles a page."""

from __future__ import annotations

import pytest

from pdf_reformatter.extract import Row, rows_from_records
from pdf_reformatter.layout import (
    DEFAULT_BOTTOM_MARGIN,
    DEFAULT_TOP_MARGIN,
    measure_row_height,
    paginate,
)


COL_WIDTHS = [60, 120, 70, 250]


def test_measure_row_height_grows_with_long_right_column():
    short = Row(cells=["1", "A", "1", ""])
    tall = Row(
        cells=[
            "1",
            "A",
            "1",
            "word " * 60,  # a long entry only in the right-hand column
        ]
    )
    short_h = measure_row_height(short, COL_WIDTHS)
    tall_h = measure_row_height(tall, COL_WIDTHS)
    assert tall_h > short_h, (
        "right-hand column with longer content must produce a taller row"
    )


def test_measure_row_height_grows_with_any_long_column():
    # The same growth must trigger for any column, not just the right.
    left_heavy = Row(cells=["x " * 40, "", "", ""])
    h_left = measure_row_height(left_heavy, COL_WIDTHS)
    h_short = measure_row_height(Row(cells=["x", "", "", ""]), COL_WIDTHS)
    assert h_left > h_short


def test_measure_row_height_mismatched_columns_raises():
    with pytest.raises(ValueError):
        measure_row_height(Row(cells=["a", "b"]), COL_WIDTHS)


def test_paginate_no_row_straddles_page_boundary():
    # Mix of short and tall rows.
    records = []
    for i in range(40):
        notes = ("long " * 30) if i in (5, 17, 33) else ""
        records.append([f"{i}", f"N{i}", f"{i*2}", notes])
    rows = rows_from_records(records)

    page_height = 400.0  # small page to force many breaks
    top = 50.0
    bottom = 50.0
    body = page_height - top - bottom

    pages = paginate(
        rows,
        col_widths=COL_WIDTHS,
        page_height=page_height,
        top_margin=top,
        bottom_margin=bottom,
    )
    assert sum(len(p.rows) for p in pages) == len(rows)
    for page in pages:
        assert page.total_height <= body + 1e-6, (
            f"page total height {page.total_height} exceeds body {body}"
        )


def test_paginate_preserves_row_order_and_count():
    rows = rows_from_records([[f"r{i}", "", "", ""] for i in range(25)])
    pages = paginate(rows, col_widths=COL_WIDTHS, page_height=200.0)
    flat = [row for p in pages for row in p.rows]
    assert [r.cells[0] for r in flat] == [r.cells[0] for r in rows]


def test_paginate_starts_new_page_when_row_would_not_fit():
    # Build rows whose heights force exactly two-per-page.
    short = Row(cells=["a", "b", "c", ""])
    rows = [short] * 10
    h = measure_row_height(short, COL_WIDTHS)
    # Body just big enough for two rows.
    pages = paginate(
        rows,
        col_widths=COL_WIDTHS,
        page_height=2 * h + DEFAULT_TOP_MARGIN + DEFAULT_BOTTOM_MARGIN,
    )
    assert all(len(p.rows) <= 2 for p in pages)
    assert sum(len(p.rows) for p in pages) == 10


def test_paginate_handles_row_taller_than_body():
    # A pathologically long entry should still be placed (alone) rather
    # than dropped silently.
    huge = Row(cells=["", "", "", "tok " * 1000])
    pages = paginate([huge], col_widths=COL_WIDTHS, page_height=200.0)
    assert len(pages) == 1
    assert pages[0].rows[0] is huge



def test_paginate_repeats_header_on_later_pages_without_losing_body_order():
    rows = rows_from_records(
        [["ID", "Name", "Qty", "Notes"]]
        + [[f"{i}", f"Item {i}", str(i), "long " * 14] for i in range(1, 9)]
    )
    page_height = 170.0
    pages = paginate(
        rows,
        col_widths=COL_WIDTHS,
        page_height=page_height,
        repeat_header=True,
    )

    assert len(pages) >= 3
    assert pages[0].rows[0].cells == ["ID", "Name", "Qty", "Notes"]
    assert pages[0].repeated_header is False

    for page in pages[1:]:
        assert page.repeated_header is True
        assert page.rows[0].cells == rows[0].cells
        assert len(page.rows) >= 2

    body_ids = [
        row.cells[0]
        for page in pages
        for idx, row in enumerate(page.rows)
        if not (idx == 0 and row.cells == rows[0].cells)
    ]
    assert body_ids == [str(i) for i in range(1, 9)]


def test_repeated_header_height_counts_against_page_body():
    rows = rows_from_records(
        [["ID", "Name", "Qty", "Notes"]]
        + [[f"{i}", "Item", "1", "wide note " * 8] for i in range(1, 6)]
    )
    page_height = 150.0
    body = page_height - DEFAULT_TOP_MARGIN - DEFAULT_BOTTOM_MARGIN
    pages = paginate(
        rows,
        col_widths=COL_WIDTHS,
        page_height=page_height,
        repeat_header=True,
    )

    for page in pages:
        assert page.total_height <= body + 1e-6
        if page.repeated_header:
            assert page.heights[0] == measure_row_height(rows[0], COL_WIDTHS)
