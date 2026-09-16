"""Withheld contract: a rendered text table sizes each column from that column's cells.

The table is only inspected through its rendered output, so the contract holds for any
implementation that measures the right thing.
"""

from __future__ import annotations

from typing import List

from boltons.tableutils import Table

HEADERS = ["id", "name", "note"]
ROWS = [
    {"id": "1", "name": "alpha", "note": "a very long note indeed"},
    {"id": "22", "name": "b", "note": "x"},
    {"id": "333", "name": "gamma-long-name", "note": "y"},
]


def rendered() -> List[str]:
    return Table.from_data(ROWS).to_text().splitlines()


def widest_cell(position: int) -> int:
    column = [HEADERS[position]] + [row[HEADERS[position]] for row in ROWS]
    return max(len(value) for value in column)


def test_every_rendered_line_has_the_same_width() -> None:
    assert len({len(line) for line in rendered()}) == 1


def test_column_separators_line_up_across_header_rule_and_body() -> None:
    layouts = {
        tuple(index for index, char in enumerate(line) if char == "|")
        for line in rendered()
    }

    assert len(layouts) == 1
    assert len(next(iter(layouts))) == len(HEADERS) - 1


def test_each_column_is_wide_enough_for_its_own_widest_cell() -> None:
    for line in rendered():
        cells = line.split("|")
        assert len(cells) == len(HEADERS)
        for position, cell in enumerate(cells):
            assert len(cell) >= widest_cell(position)


def test_cell_text_is_never_dropped_or_clipped() -> None:
    lines = rendered()
    cells = [[part.strip() for part in line.split("|")] for line in lines]

    assert cells[0] == HEADERS
    assert cells[2] == ["1", "alpha", "a very long note indeed"]
    assert cells[3] == ["22", "b", "x"]
    assert cells[4] == ["333", "gamma-long-name", "y"]
    assert set(lines[1]) == {"-", "|"}
