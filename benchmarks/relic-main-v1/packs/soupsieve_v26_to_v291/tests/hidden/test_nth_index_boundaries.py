"""Withheld contract: An+B index arithmetic agrees with the CSS selector specification.

Exercised only through ``soupsieve.select``, so any implementation that computes the
sequence correctly satisfies the contract regardless of how it is structured internally.
"""

from __future__ import annotations

from bs4 import BeautifulSoup

import soupsieve


def selected_ids(selector: str, markup: str) -> list[str]:
    root = BeautifulSoup(markup, "html.parser").div
    assert root is not None
    return [element["id"] for element in soupsieve.select(selector, root)]


def test_forward_sequence_continues_after_zero_boundary() -> None:
    markup = "<div>" + "".join(f'<i id="i{index}"></i>' for index in range(1, 13)) + "</div>"

    assert selected_ids("*:nth-child(5n-10)", markup) == ["i5", "i10"]


def test_descending_of_type_sequence_includes_terminal_sibling() -> None:
    markup = "<div>" + "".join(f'<i id="i{index}"></i>' for index in range(1, 7)) + "</div>"

    assert selected_ids("i:nth-of-type(-6n+12)", markup) == ["i6"]


def test_reverse_of_type_sequence_handles_negative_offset() -> None:
    markup = (
        "<div>"
        + "".join(
            f'<em id="e{index}"></em><span id="s{index}"></span>' for index in range(1, 8)
        )
        + "</div>"
    )

    assert selected_ids("em:nth-last-of-type(3n-6)", markup) == ["e2", "e5"]


def test_established_nth_forms_remain_stable() -> None:
    markup = """
    <div>
      <i id="a"></i>text<b id="b"></b>
      <i id="c"></i><b id="d"></b><i id="e"></i>
    </div>
    """

    assert selected_ids("*:nth-last-child(odd)", markup) == ["a", "c", "e"]
    assert selected_ids("b:nth-of-type(2)", markup) == ["d"]
