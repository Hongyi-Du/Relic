"""Withheld contract: the expanded state of disclosure and dialog elements is selectable.

Exercised only through ``soupsieve.select``, so any implementation that reports the state
correctly satisfies the contract regardless of how the state check is structured.
"""

from __future__ import annotations

from bs4 import BeautifulSoup

import soupsieve

MARKUP = """
<div>
  <details id="shut_details"><summary>one</summary><p>first</p></details>
  <details id="wide_details" open><summary>two</summary><p>second</p></details>
  <dialog id="shut_dialog"><p>third</p></dialog>
  <dialog id="wide_dialog" open><p>fourth</p></dialog>
  <div id="carrier" open>fifth</div>
</div>
"""


def selected_ids(selector: str, markup: str = MARKUP) -> list[str]:
    root = BeautifulSoup(markup, "html.parser").div
    assert root is not None
    return [element["id"] for element in soupsieve.select(selector, root)]


def test_expanded_state_matches_only_disclosure_and_dialog_elements() -> None:
    assert selected_ids(":open") == ["wide_details", "wide_dialog"]


def test_expanded_state_narrows_to_the_requested_element_type() -> None:
    assert selected_ids("details:open") == ["wide_details"]
    assert selected_ids("dialog:open") == ["wide_dialog"]


def test_expanded_state_negates_like_any_other_state_check() -> None:
    assert selected_ids(":is(details, dialog):not(:open)") == ["shut_details", "shut_dialog"]


def test_plain_attribute_presence_keeps_its_broader_meaning() -> None:
    assert selected_ids("[open]") == ["wide_details", "wide_dialog", "carrier"]
    assert selected_ids("div[open]") == ["carrier"]
