"""Shared fixtures.

We build the sample input PDF on the fly with reportlab. This keeps the
test suite hermetic — no checked-in binary that drifts away from the
expected schema — and the generator is also a clear specification of
the source layout: four columns per row, the right-hand column
deliberately receives a long entry on one row so the variable-length
behaviour can be asserted.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import pytest
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

# Make ``solution/`` importable when running the tests directly from
# the repository checkout. Harbor mounts /solution and tests/test.sh
# sets PYTHONPATH=/solution — this fallback is only needed for the
# `pytest tests/` invocation a developer runs locally.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_LOCAL_SOLUTION = _REPO_ROOT / "solution"
if _LOCAL_SOLUTION.exists() and str(_LOCAL_SOLUTION) not in sys.path:
    sys.path.insert(0, str(_LOCAL_SOLUTION))


SAMPLE_HEADER = ["ID", "Name", "Quantity", "Notes"]

# 30 data rows. Two of them carry deliberately long ``Notes`` so the
# right-hand column drives the row height. The remainder are short so
# the page-fit algorithm has to mix tall and short rows.
SAMPLE_ROWS: List[List[str]] = []
for i in range(1, 31):
    notes = ""
    if i == 7:
        notes = (
            "Very long note explaining the entry in much more detail than "
            "the other rows so the right hand column drives a tall row "
            "that would otherwise straddle the page boundary."
        )
    elif i == 22:
        notes = (
            "Another extended remark with several sentences. It mentions "
            "several details that wrap across multiple lines inside the "
            "right hand column of the source PDF."
        )
    SAMPLE_ROWS.append([f"{i:03d}", f"Item {i}", f"{i * 3}", notes])


def _build_sample_pdf(path: Path) -> None:
    """Generate a sample PDF whose extracted text reproduces ``SAMPLE_ROWS``.

    The pdfplumber default extractor finds tables via ruled lines, so we
    draw an explicit grid: one box per cell, text inside each box. We
    deliberately let rows overflow page boundaries — that is the
    misbehaviour the tool fixes.
    """

    page_w, page_h = A4
    c = canvas.Canvas(str(path), pagesize=A4)
    col_widths = [60, 120, 70, 250]
    left = (page_w - sum(col_widths)) / 2.0
    row_height = 30.0
    top = page_h - 50
    y = top

    # Header.
    rows = [SAMPLE_HEADER] + SAMPLE_ROWS
    for row in rows:
        # Compute per-row height from the longest cell's wrapped length.
        max_lines = 1
        for cell, w in zip(row, col_widths):
            chars_per_line = max(int((w - 4) / 5.0), 1)
            words = cell.split(" ")
            line = ""
            count = 1
            for word in words:
                candidate = (line + " " + word).strip()
                if len(candidate) <= chars_per_line:
                    line = candidate
                else:
                    count += 1
                    line = word
            max_lines = max(max_lines, count)
        h = max(row_height, max_lines * 12 + 6)

        if y - h < 30:
            # NOTE: the bug being fixed — we keep drawing instead of
            # starting a new page, so the row straddles the boundary.
            # The reformatter will paginate correctly. For the fixture
            # we DO start a new page here so the extractor still reads
            # every row, but we leave only a tiny margin to mimic the
            # "no space left" condition.
            c.showPage()
            y = top
        x = left
        for cell, w in zip(row, col_widths):
            c.rect(x, y - h, w, h, stroke=1, fill=0)
            # Wrap text inside the cell.
            chars_per_line = max(int((w - 4) / 5.0), 1)
            words = cell.split(" ") if cell else [""]
            line = ""
            ty = y - 12
            for word in words:
                candidate = (line + " " + word).strip()
                if len(candidate) <= chars_per_line:
                    line = candidate
                else:
                    c.drawString(x + 2, ty, line)
                    ty -= 12
                    line = word
            if line:
                c.drawString(x + 2, ty, line)
            x += w
        y -= h
    c.save()


@pytest.fixture(scope="session")
def sample_pdf(tmp_path_factory) -> Path:
    p = tmp_path_factory.mktemp("input") / "sample.pdf"
    _build_sample_pdf(p)
    return p


@pytest.fixture
def out_dir(tmp_path: Path) -> Path:
    d = tmp_path / "out"
    d.mkdir()
    return d


@pytest.fixture
def sample_rows():
    """In-memory mirror of the rows the sample PDF carries."""

    from pdf_reformatter.extract import rows_from_records

    return rows_from_records([SAMPLE_HEADER] + SAMPLE_ROWS)
