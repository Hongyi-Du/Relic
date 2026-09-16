"""Withheld contract: prefix/suffix/substring attribute tests with an empty operand match nothing.

Exercised only through ``soupsieve.select``; the contract says what the selector must select,
not how the comparison is built.
"""

from __future__ import annotations

from bs4 import BeautifulSoup

import soupsieve

MARKUP = """
<div>
  <a id="filled" href="alpha">one</a>
  <a id="blank" href="">two</a>
  <a id="absent">three</a>
</div>
"""


def selected_ids(selector: str) -> list[str]:
    root = BeautifulSoup(MARKUP, "html.parser").div
    assert root is not None
    return [element["id"] for element in soupsieve.select(selector, root)]


def test_empty_operand_selects_nothing_for_every_substring_operator() -> None:
    assert selected_ids('[href^=""]') == []
    assert selected_ids('[href$=""]') == []
    assert selected_ids('[href*=""]') == []


def test_case_folding_flags_do_not_revive_the_empty_operand() -> None:
    assert selected_ids('[href^="" i]') == []
    assert selected_ids('[href$="" s]') == []
    assert selected_ids("[href*='' i]") == []


def test_empty_operand_stays_empty_when_combined_with_other_conditions() -> None:
    assert selected_ids('a[href][href*=""]') == []
    assert selected_ids(':is([href^=""], [href$="a"])') == ["filled"]


def test_non_empty_and_presence_forms_keep_their_meaning() -> None:
    assert selected_ids("[href^=a]") == ["filled"]
    assert selected_ids("[href$=a]") == ["filled"]
    assert selected_ids("[href*=ph]") == ["filled"]
    assert selected_ids("[href]") == ["filled", "blank"]
    assert selected_ids('[href=""]') == ["blank"]
    assert selected_ids('[href~=""]') == []
