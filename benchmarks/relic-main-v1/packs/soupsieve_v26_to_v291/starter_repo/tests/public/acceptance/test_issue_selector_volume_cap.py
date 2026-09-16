"""Withheld contract: an absurdly large selector list is refused instead of compiled.

Exercised through the top-level ``compile`` entry point. Where the ceiling is enforced does
not matter; what matters is that a caller gets a ``ValueError`` rather than a matcher whose
every use costs a proportional amount of work.
"""

from __future__ import annotations

import pytest
from bs4 import BeautifulSoup

import soupsieve

MARKUP = '<div><i id="target">x</i><b id="other">y</b></div>'
EXCESSIVE = 30000
ORDINARY = 512


def selector_list(count: int) -> str:
    return ",".join("i" for _ in range(count))


def selected_ids(selector: str) -> list[str]:
    root = BeautifulSoup(MARKUP, "html.parser").div
    assert root is not None
    return [element["id"] for element in soupsieve.select(selector, root)]


def test_a_direct_selector_list_beyond_the_ceiling_is_refused() -> None:
    with pytest.raises(ValueError):
        soupsieve.compile(selector_list(EXCESSIVE))


def test_selectors_reached_through_a_matches_any_group_also_count() -> None:
    with pytest.raises(ValueError):
        soupsieve.compile(f":is({selector_list(EXCESSIVE)})")


def test_selectors_reached_through_a_caller_supplied_alias_also_count() -> None:
    with pytest.raises(ValueError):
        soupsieve.compile("i:--bulk", custom={":--bulk": selector_list(EXCESSIVE)})


def test_ordinary_selector_lists_still_compile_and_select() -> None:
    assert selected_ids(selector_list(ORDINARY)) == ["target"]
    assert selected_ids(f":is({selector_list(ORDINARY)}, b)") == ["target", "other"]

    root = BeautifulSoup(MARKUP, "html.parser").div
    assert root is not None
    compiled = soupsieve.compile(":--few", custom={":--few": selector_list(ORDINARY)})
    assert [element["id"] for element in compiled.select(root)] == ["target"]
