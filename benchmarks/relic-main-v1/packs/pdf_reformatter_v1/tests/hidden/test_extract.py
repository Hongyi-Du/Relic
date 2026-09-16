"""Extraction layer: ``extract_rows`` reads the source PDF table."""

from __future__ import annotations

from pdf_reformatter.extract import Row, extract_rows, rows_from_records


def test_extract_rows_returns_row_objects(sample_pdf):
    rows = extract_rows(str(sample_pdf))
    assert rows, "expected at least one row from the sample PDF"
    assert all(isinstance(r, Row) for r in rows)


def test_extract_rows_uniform_column_count(sample_pdf):
    rows = extract_rows(str(sample_pdf))
    widths = {len(r.cells) for r in rows}
    assert len(widths) == 1, f"rows have mismatched widths: {widths}"
    assert widths.pop() == 4


def test_extract_preserves_long_right_hand_cell(sample_pdf):
    rows = extract_rows(str(sample_pdf))
    long_cells = [r.cells[3] for r in rows if "Very long note" in r.cells[3]]
    assert long_cells, "long right-hand column entry was dropped"


def test_extract_empty_right_hand_cells_kept(sample_pdf):
    rows = extract_rows(str(sample_pdf))
    # The bulk of rows carry an empty Notes column; they must still be
    # represented with an empty string, not omitted from the row.
    empty_count = sum(1 for r in rows if r.cells[3] == "")
    assert empty_count > 5


def test_rows_from_records_pads_short_rows():
    rows = rows_from_records([["a", "b", "c"], ["d"]])
    assert len(rows) == 2
    assert rows[1].cells == ["d", "", ""]
