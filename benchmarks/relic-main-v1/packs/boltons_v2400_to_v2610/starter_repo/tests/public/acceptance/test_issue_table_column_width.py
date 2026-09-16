"""Acceptance check for issue_table_column_width.

Written from the report the engineers were given, not from the fix.
It states in code what the issue states in prose: In the rendered text table every column is sized by its own widest cell (header included), so all rendered lines have equal width, the column separators line up on every row, and no cell text is clipped.
"""
from boltons.tableutils import Table


def test_to_text_sizes_each_column_by_its_widest_cell():
    table = Table(
        [['short', 'a much longer value'], ['widest', 'x']],
        headers=['first', 'second'],
    )

    lines = table.to_text().splitlines()

    assert len({len(line) for line in lines}) == 1
    assert {
        tuple(index for index, character in enumerate(line) if character == '|')
        for line in lines
    } == {tuple(index for index, character in enumerate(lines[0]) if character == '|')}
    assert 'short' in table.to_text()
    assert 'a much longer value' in table.to_text()
    assert 'widest' in table.to_text()
    assert 'x' in table.to_text()
